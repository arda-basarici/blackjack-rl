"""Reconstructed ground-truth references for Problem B (DESIGN D17, build stage B1).

B has "no clean table" (§3), but we still audit against reconstructed truth:
- ``edge_by_count``   — empirical player edge vs Hi-Lo true count (basic strategy, many hands),
- ``kelly_bet_curve`` — the analytic full-Kelly bet fraction implied by edge-by-count,
- ``index_plays``     — the known count-deviation index plays, to audit learned deviations.

The edge measurement runs flat-bet basic strategy through the **same** Problem-B session env the bet
agent trains in (``session.env``), so the reference and the agent are measured on identical terms
(D17). ``edge_by_count`` / ``kelly_bet_curve`` are *measured*; ``index_plays`` is *literature*.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

from simulator.game_state import Action
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import EDGE_REFERENCE_PATH
from blackjack_rl.session.bet_agent import FlatBet
from blackjack_rl.session.env import SessionConfig, run_sessions


@dataclass(frozen=True)
class CountEdge:
    """Measured player return per unit wagered at one (integer) Hi-Lo true-count bucket.

    ``mean_return`` is the player's expected net per unit bet — **positive = player advantage**, the
    *negative* of the house edge in ``evaluation.metrics.EdgeResult``. ``variance`` is the per-hand
    outcome variance in the bucket (~1.3 for blackjack), the denominator of the Kelly fraction.
    ``std_error`` is the SE of ``mean_return`` so a bucket's edge can be read with a CI — rare extreme
    counts are noisy until measured over many hands.
    """

    true_count: int
    mean_return: float
    variance: float
    std_error: float
    n: int


@dataclass
class CountAccumulator:
    """Per-bucket Welford moments ``[n, mean, M2]`` keyed by integer Hi-Lo true count — the shared
    accumulation primitive behind ``edge_by_count`` (single stream) and its parallel runner (B2a).

    Holds *raw* running moments, not finalized ``CountEdge``s, for two reasons: (1) partials from
    independent workers combine **losslessly** via ``merge`` (Chan's parallel variance), which needs
    M2, not a finished variance; (2) the ``n < 2`` drop is deferred to ``edges`` so a bucket that is
    tiny in one worker but populated overall is not lost mid-merge. ``add`` mutates owned local state
    (Welford is inherently accumulative); ``merge`` and ``edges`` are pure.
    """

    buckets: dict[int, list[float]] = field(default_factory=dict)  # tc -> [n, mean, M2]

    def add(self, true_count: int, value: float) -> None:
        """Fold one per-unit return into its bucket — Welford's online mean+variance update."""
        a = self.buckets.setdefault(true_count, [0.0, 0.0, 0.0])
        a[0] += 1.0
        delta = value - a[1]
        a[1] += delta / a[0]
        a[2] += delta * (value - a[1])

    def merge(self, other: CountAccumulator) -> CountAccumulator:
        """Combine two partials into a new accumulator (pure) via Chan et al.'s parallel variance.

        Per shared bucket: ``n = nA + nB``, ``mean = meanA + δ·nB/n``, ``M2 = M2A + M2B + δ²·nA·nB/n``
        (``δ = meanB − meanA``); buckets present on only one side carry over unchanged. Exact —
        merging worker partials yields the same moments as folding every hand into one stream, which
        is what lets the runner fan out across cores without changing the measured curve.
        """
        out = CountAccumulator({tc: list(m) for tc, m in self.buckets.items()})
        for tc, (n_b, mean_b, m2_b) in other.buckets.items():
            if tc not in out.buckets:
                out.buckets[tc] = [n_b, mean_b, m2_b]
                continue
            n_a, mean_a, m2_a = out.buckets[tc]
            n = n_a + n_b
            delta = mean_b - mean_a
            out.buckets[tc] = [
                n,
                mean_a + delta * n_b / n,
                m2_a + m2_b + delta * delta * n_a * n_b / n,
            ]
        return out

    def edges(self) -> dict[int, CountEdge]:
        """Finalize to one ``CountEdge`` per bucket with ``n >= 2`` (variance needs two), keyed and
        ordered by true count. Single-hand buckets (rare extreme counts) are dropped as unmeasurable.
        """
        result: dict[int, CountEdge] = {}
        for true_count in sorted(self.buckets):
            n_raw, mean, m2 = self.buckets[true_count]
            n = int(n_raw)
            if n < 2:
                continue
            variance = m2 / (n - 1)
            result[true_count] = CountEdge(
                true_count=true_count,
                mean_return=mean,
                variance=variance,
                std_error=sqrt(variance / n),
                n=n,
            )
        return result

    @property
    def n_total(self) -> int:
        """Total hands folded in, across all buckets (incl. ``n < 2`` ones)."""
        return int(sum(m[0] for m in self.buckets.values()))

    @property
    def pooled_mean(self) -> float:
        """Frequency-weighted mean per-unit return over *all* buckets (incl. ``n < 2``) — the exact
        flat-bet edge across every hand seen, the anchor-check quantity (B2a)."""
        total = sum(m[0] for m in self.buckets.values())
        return sum(m[0] * m[1] for m in self.buckets.values()) / total if total else 0.0


def accumulate_edges(
    *, n_hands: int, seed: int, max_hands_per_session: int = 1000
) -> CountAccumulator:
    """Play ~``n_hands`` of flat-bet basic strategy through the Problem-B session env off ``seed`` and
    return the raw per-bucket ``CountAccumulator`` (unfinalized, mergeable) — the shared core of
    ``edge_by_count`` and the parallel runner (one call per worker, with distinct seeds).

    Counting on, shoe persists, reshuffle at penetration — the *same* env the bet agent trains in
    (D17 "identical terms"); each hand is bucketed by ``round(true_count)`` and folded with Welford.
    The starting bankroll sits far above any reachable loss so a session never ruins; ruin timing is
    count-independent anyway, so it could not bias the per-count buckets regardless.
    """
    if n_hands < 1:
        raise ValueError(f"n_hands must be >= 1, got {n_hands}")

    n_sessions = max(1, -(-n_hands // max_hands_per_session))  # ceil division
    config = SessionConfig(
        starting_bankroll=float(max_hands_per_session) * 10.0 + 1000.0,  # ruin unreachable at flat 1
        max_hands=max_hands_per_session,
        seed=seed,
    )

    acc = CountAccumulator()
    collected = 0
    for capture in run_sessions(config, BasicStrategy(), FlatBet(1.0), n_sessions):
        for rec in capture.hands:
            if collected >= n_hands:
                break
            ret = rec.payout / rec.bet  # per-unit return (flat bet 1; robust to any spread clamp)
            acc.add(round(rec.true_count), ret)
            collected += 1
        if collected >= n_hands:
            break
    return acc


def edge_by_count(
    *, n_hands: int, seed: int = 0, max_hands_per_session: int = 1000
) -> dict[int, CountEdge]:
    """Empirically measure player edge as a function of Hi-Lo true count (DESIGN D17, B1).

    A thin single-stream finalize over ``accumulate_edges``: returns one ``CountEdge`` per bucket
    that saw >= 2 hands (variance needs two), keyed and ordered by integer true count. The high-n
    parallel measurement (B2a) reuses ``accumulate_edges`` per worker and merges before finalizing.
    """
    return accumulate_edges(
        n_hands=n_hands, seed=seed, max_hands_per_session=max_hands_per_session
    ).edges()


def kelly_bet_curve(edges: dict[int, CountEdge]) -> dict[int, float]:
    """Full-Kelly bet fraction implied by an ``edge_by_count`` measurement (DESIGN D17, B1).

    For each count bucket the Kelly fraction of bankroll to wager is ``mean_return / variance`` (the
    mean-over-variance optimum for log-growth), floored at 0 — at a non-positive edge Kelly says do
    not bet. This is the *continuous, unbounded* reference curve: B2 discretizes it into the bet
    spread (D15) and real play caps it (fractional Kelly / table max). Keyed and ordered by true count.
    """
    curve: dict[int, float] = {}
    for true_count in sorted(edges):
        edge = edges[true_count]
        fraction = edge.mean_return / edge.variance if edge.variance > 0 else 0.0
        curve[true_count] = max(0.0, fraction)
    return curve


@dataclass(frozen=True)
class EdgeReference:
    """The committed edge-by-count reference: per-count measured ``edges`` + the implied full-Kelly
    ``kelly_curve``, plus the ``provenance`` of the run that produced them (DESIGN D17, B2c).

    This is the **single canonical source** the Problem-B Kelly baseline (``KellyBet``) sizes from and
    the signature figure plots — frozen and committed (``core.paths.EDGE_REFERENCE_PATH``) rather than
    read from a git-ignored ``runs/`` artifact, so a reference everything is measured against cannot
    silently drift. Regenerate in place with ``scripts/measure_edge_by_count.py``.
    """

    edges: dict[int, CountEdge]
    kelly_curve: dict[int, float]
    provenance: dict


def load_edge_reference(path: Path | str = EDGE_REFERENCE_PATH) -> EdgeReference:
    """Load the committed edge-by-count reference JSON (DESIGN D17, B2c).

    Reconstructs the int-keyed ``edges`` and ``kelly_curve`` (the JSON keys them by string) and keeps
    the provenance fields (run id / timestamp / git hash / config / anchor check) so any run that uses
    the reference can record exactly which measurement it sized from.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edges = {
        int(e["true_count"]): CountEdge(
            true_count=int(e["true_count"]),
            mean_return=e["mean_return"],
            variance=e["variance"],
            std_error=e["std_error"],
            n=int(e["n"]),
        )
        for e in data["edges"]
    }
    kelly_curve = {int(tc): f for tc, f in data["kelly_curve"].items()}
    provenance = {
        k: data[k]
        for k in ("run_id", "timestamp", "git_hash", "config", "n_total", "anchor_check")
        if k in data
    }
    return EdgeReference(edges=edges, kelly_curve=kelly_curve, provenance=provenance)


@dataclass(frozen=True)
class IndexPlay:
    """One Hi-Lo count-deviation from basic strategy.

    Semantics: take ``action_at_or_above`` when ``true_count >= index``, otherwise ``action_below``
    (the basic-strategy play). Pairs use ``is_pair`` with ``player_total`` = the pair's total (e.g.
    T,T = 20). ``dealer_upcard`` is 2..11 (11 = ace).
    """

    label: str
    player_total: int
    is_pair: bool
    dealer_upcard: int
    index: float
    action_below: Action
    action_at_or_above: Action


@dataclass(frozen=True)
class IndexPlayTable:
    """The reconstructed play-side reference (literature, not measured): the Illustrious 18 playing
    deviations and the Fab 4 surrenders (Schlesinger, *Blackjack Attack*; Hi-Lo).

    Caveats (intellectual honesty):
    - **Insurance** — the single most valuable deviation (take at TC >= +3) is omitted: it is a side
      bet, not a ``GameState`` ``Action`` the engine models, so it is unauditable here.
    - **Surrenders** require ``surrender_allowed=True`` (off in ``problem_b_config`` by default), so
      they are reference-only until enabled. **15 v 10** appears in both groups: surrender at TC >= 0
      takes precedence *when surrender is available*; otherwise the playing deviation (stand at
      TC >= +4) applies.
    - Exact indices and the ``>=`` vs ``>`` boundary vary slightly by source and rule set (S17/H17,
      DAS). B3 audits learned deviations against this table and should cross-check ``action_below``
      against the engine's actual ``BasicStrategy`` for the live config.
    """

    playing: tuple[IndexPlay, ...]
    surrender: tuple[IndexPlay, ...]


def index_plays() -> IndexPlayTable:
    """The known Hi-Lo index plays — the play-side reference B3 audits learned deviations against
    (DESIGN D17, B1). Literature-sourced; see ``IndexPlayTable`` for caveats and semantics."""
    playing = (
        IndexPlay("16 v 10", 16, False, 10, 0.0, "hit", "stand"),
        IndexPlay("15 v 10", 15, False, 10, 4.0, "hit", "stand"),
        IndexPlay("T,T v 5", 20, True, 5, 5.0, "stand", "split"),
        IndexPlay("T,T v 6", 20, True, 6, 4.0, "stand", "split"),
        IndexPlay("10 v 10", 10, False, 10, 4.0, "hit", "double"),
        IndexPlay("12 v 3", 12, False, 3, 2.0, "hit", "stand"),
        IndexPlay("12 v 2", 12, False, 2, 3.0, "hit", "stand"),
        IndexPlay("11 v A", 11, False, 11, 1.0, "hit", "double"),
        IndexPlay("9 v 2", 9, False, 2, 1.0, "hit", "double"),
        IndexPlay("10 v A", 10, False, 11, 4.0, "hit", "double"),
        IndexPlay("9 v 7", 9, False, 7, 3.0, "hit", "double"),
        IndexPlay("16 v 9", 16, False, 9, 5.0, "hit", "stand"),
        IndexPlay("13 v 2", 13, False, 2, -1.0, "hit", "stand"),
        IndexPlay("12 v 4", 12, False, 4, 0.0, "hit", "stand"),
        IndexPlay("12 v 5", 12, False, 5, -2.0, "hit", "stand"),
        IndexPlay("12 v 6", 12, False, 6, -1.0, "hit", "stand"),
        IndexPlay("13 v 3", 13, False, 3, -2.0, "hit", "stand"),
    )
    surrender = (
        IndexPlay("14 v 10", 14, False, 10, 3.0, "hit", "surrender"),
        IndexPlay("15 v 10", 15, False, 10, 0.0, "hit", "surrender"),
        IndexPlay("15 v 9", 15, False, 9, 2.0, "hit", "surrender"),
        IndexPlay("15 v A", 15, False, 11, 1.0, "hit", "surrender"),
    )
    return IndexPlayTable(playing=playing, surrender=surrender)
