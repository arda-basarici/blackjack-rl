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

from dataclasses import dataclass
from math import sqrt

from simulator.game_state import Action
from strategies.basic_strategy import BasicStrategy

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


def edge_by_count(
    *, n_hands: int, seed: int = 0, max_hands_per_session: int = 1000
) -> dict[int, CountEdge]:
    """Empirically measure player edge as a function of Hi-Lo true count (DESIGN D17, B1).

    Plays flat-bet basic strategy through the Problem-B session env (counting on, shoe persists,
    reshuffle at penetration) — the *same* env the bet agent trains in (D17 "identical terms") — and
    buckets each hand by ``round(true_count)``, accumulating the per-unit return with Welford's
    algorithm (stable single-pass mean + variance). The starting bankroll is set far above any
    reachable loss so a session never ruins; ruin timing would be count-independent anyway, so it
    could not bias the per-count buckets regardless. Only the edge is measured here.

    Returns one ``CountEdge`` per bucket that saw >= 2 hands (variance needs two), keyed and ordered
    by integer true count. Single-hand buckets (the rare extreme counts) are dropped as unmeasurable.
    """
    if n_hands < 1:
        raise ValueError(f"n_hands must be >= 1, got {n_hands}")

    n_sessions = max(1, -(-n_hands // max_hands_per_session))  # ceil division
    config = SessionConfig(
        starting_bankroll=float(max_hands_per_session) * 10.0 + 1000.0,  # ruin unreachable at flat 1
        max_hands=max_hands_per_session,
        seed=seed,
    )

    acc: dict[int, list[float]] = {}  # bucket -> [n, mean, m2] (Welford)
    collected = 0
    for capture in run_sessions(config, BasicStrategy(), FlatBet(1.0), n_sessions):
        for rec in capture.hands:
            if collected >= n_hands:
                break
            ret = rec.payout / rec.bet  # per-unit return (flat bet 1; robust to any spread clamp)
            a = acc.setdefault(round(rec.true_count), [0.0, 0.0, 0.0])
            a[0] += 1
            delta = ret - a[1]
            a[1] += delta / a[0]
            a[2] += delta * (ret - a[1])
            collected += 1
        if collected >= n_hands:
            break

    edges: dict[int, CountEdge] = {}
    for true_count in sorted(acc):
        n_raw, mean, m2 = acc[true_count]
        n = int(n_raw)
        if n < 2:
            continue
        variance = m2 / (n - 1)
        edges[true_count] = CountEdge(
            true_count=true_count,
            mean_return=mean,
            variance=variance,
            std_error=sqrt(variance / n),
            n=n,
        )
    return edges


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
