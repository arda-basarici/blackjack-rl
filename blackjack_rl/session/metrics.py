"""Outcome & risk metrics for Problem B (DESIGN D17, build stage B2c).

Two axes, never collapsed (mirrors §7's discipline, adapted to B):
- **outcome** — log-growth rate per hand; final-bankroll distribution,
- **risk**    — probability of ruin (headline for a finite bankroll); drawdown distribution.

All are **pure functions over an already-played ``list[SessionCapture]``** (functional core, decision
D1): the caller plays the batch once via ``session.env.run_sessions`` and scores every rung of the
D17 ladder on the same captures. The unit of replication is the **session** (independent: fresh shoe,
reset bankroll), so confidence intervals aggregate over sessions, never over the (correlated) hands
within a session — that would be pseudo-replication and an overconfident CI. Rates near 0 (ruin in
the growth regime) use the **Wilson** interval, which Wald collapses on. Full derivations:
WHITEBOARD.md 2026-06-29.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

from blackjack_rl.session.env import SessionCapture

Z95 = 1.959963984540054  # standard-normal 97.5th percentile (two-sided 95%)


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a two-sided 95% CI and the sample size it rests on (``n`` = sessions)."""

    value: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class Proportion:
    """A measured rate ``k/n`` with a **Wilson** 95% CI — for ruin / drawdown-breach (a Bernoulli over
    sessions). Wilson, not Wald, because these rates sit near 0 where Wald gives a zero-width
    (falsely certain) interval."""

    estimate: float
    low: float
    high: float
    k: int
    n: int


@dataclass(frozen=True)
class Distribution:
    """Shape of a per-session quantity across the batch: ``n``, ``mean``, ``minimum``, ``maximum`` and
    the requested ``quantiles`` (``{q: value}``). Quantiles because final bankroll / drawdown are
    right-skewed, so the mean alone misleads."""

    n: int
    mean: float
    minimum: float
    maximum: float
    quantiles: dict[float, float]


# --- pure helpers --------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = Z95) -> Proportion:
    """Wilson score 95% CI for a proportion of ``k`` successes in ``n`` trials (DESIGN D17).

    Stays inside [0, 1] and keeps a positive width at ``k=0`` or ``k=n``, where the Wald interval
    collapses to zero — the right tool for rare-event rates like ruin. Formula: WHITEBOARD 2026-06-29.
    """
    if n <= 0:
        raise ValueError(f"wilson_interval needs n >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return Proportion(estimate=p, low=max(0.0, center - half), high=min(1.0, center + half), k=k, n=n)


def _quantile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated quantile of a pre-sorted, non-empty list (``q`` in [0, 1])."""
    if q <= 0:
        return sorted_xs[0]
    if q >= 1:
        return sorted_xs[-1]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    if lo + 1 >= len(sorted_xs):
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[lo + 1] * frac


def _mean_ci(values: list[float], z: float = Z95) -> Estimate:
    """Mean of ``values`` with a normal-approximation two-sided 95% CI (SE = s/sqrt(n)). Sessions are
    the independent unit and ``n`` is in the hundreds+, so the normal approx ~ Student-t. ``n < 2``
    yields a zero-width CI; empty yields ``-inf`` (every session was a wipeout)."""
    n = len(values)
    if n == 0:
        return Estimate(value=float("-inf"), low=float("-inf"), high=float("-inf"), n=0)
    m = sum(values) / n
    if n < 2:
        return Estimate(value=m, low=m, high=m, n=n)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    half = z * sqrt(var / n)
    return Estimate(value=m, low=m - half, high=m + half, n=n)


def _summarize(values: list[float], quantiles: tuple[float, ...]) -> Distribution:
    """Mean / min / max / requested quantiles of a non-empty batch of per-session values."""
    if not values:
        raise ValueError("cannot summarize an empty batch")
    xs = sorted(values)
    return Distribution(
        n=len(xs),
        mean=sum(xs) / len(xs),
        minimum=xs[0],
        maximum=xs[-1],
        quantiles={q: _quantile(xs, q) for q in quantiles},
    )


# --- per-session primitives (scalar; the building blocks of the batch aggregates) ----

def session_growth_rate(capture: SessionCapture) -> float:
    """One session's per-hand log-growth rate ``g_i = log(W_final / W_0) / N`` (the per-hand log
    increments telescope). ``-inf`` on a total wipeout (final bankroll 0); ``0`` for an empty session.
    """
    if capture.n_hands == 0:
        return 0.0
    if capture.final_bankroll <= 0.0:
        return float("-inf")
    return log(capture.final_bankroll / capture.starting_bankroll) / capture.n_hands


def session_max_drawdown(capture: SessionCapture) -> float:
    """One session's drawdown vs its **initial** bankroll: ``1 - min(W_t) / W_0`` in [0, 1] (hard ruin
    is the limiting case ``1``). Initial-anchored, not peak-to-trough, because the risk here is
    survival — distance toward the barrier — see WHITEBOARD 2026-06-29."""
    w0 = capture.starting_bankroll
    lo = min([w0, *(h.bankroll_after for h in capture.hands)])
    return max(0.0, 1.0 - lo / w0)


# --- scalar aggregation cores (over per-session values) ------------------------------
# A parallel runner reduces each SessionCapture to these per-session scalars *in-worker* (so it ships
# tiny arrays across processes, not full captures) and aggregates here. The capture-based functions
# below are thin wrappers over these, so single-process and parallel paths share one implementation.

def growth_rate_of(per_session_growth: list[float]) -> Estimate:
    """Growth-rate Estimate from per-session ``g_i``. Wipeouts (``-inf``) are dropped from the
    estimate; ``Estimate.n`` is the count kept, so ``len(input) - n`` is the dropped-wipeout count
    (accounted for on the risk axis, D2 option a)."""
    return _mean_ci([g for g in per_session_growth if isfinite(g)])


def bankroll_distribution_of(
    final_bankrolls: list[float], quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> Distribution:
    """Final-bankroll Distribution from per-session final bankrolls (skew-aware quantiles)."""
    if not final_bankrolls:
        raise ValueError("bankroll_distribution_of needs at least one session")
    return _summarize(final_bankrolls, quantiles)


def drawdown_distribution_of(
    drawdowns: list[float], quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
) -> Distribution:
    """Drawdown Distribution from per-session drawdowns ``D_i``."""
    if not drawdowns:
        raise ValueError("drawdown_distribution_of needs at least one session")
    return _summarize(drawdowns, quantiles)


# --- batch aggregates over captures (single-process / tests; delegate to the cores) --

def growth_rate(captures: list[SessionCapture]) -> Estimate:
    """Per-hand log-growth rate across the batch: mean +/- 95% CI over the per-session ``g_i``
    (sessions are the independent unit, D1). Wiped sessions are dropped from the estimate and
    accounted for on the risk axis (D2 option a); ``len(captures) - Estimate.n`` is the dropped count.
    """
    if not captures:
        raise ValueError("growth_rate needs at least one session")
    return growth_rate_of([session_growth_rate(c) for c in captures])


def bankroll_distribution(
    captures: list[SessionCapture], quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> Distribution:
    """Final-bankroll distribution across sessions (D17): quantiles because final wealth is
    right-skewed (a few count-favorable shoes run up big), so the mean alone misleads."""
    if not captures:
        raise ValueError("bankroll_distribution needs at least one session")
    return bankroll_distribution_of([c.final_bankroll for c in captures], quantiles)


def ruin_probability(captures: list[SessionCapture]) -> Proportion:
    """Fraction of sessions that hit the hard ruin barrier, with a Wilson 95% CI (D17)."""
    if not captures:
        raise ValueError("ruin_probability needs at least one session")
    k = sum(1 for c in captures if c.ruined)
    return wilson_interval(k, len(captures))


def drawdown_distribution(
    captures: list[SessionCapture], quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
) -> Distribution:
    """Distribution of per-session drawdown ``D_i`` across the batch (D17) — the threshold-free risk
    shape, and the informative risk signal in the growth regime where hard ruin ~ 0."""
    if not captures:
        raise ValueError("drawdown_distribution needs at least one session")
    return drawdown_distribution_of([session_max_drawdown(c) for c in captures], quantiles)


def drawdown_breach_probability(captures: list[SessionCapture], level: float) -> Proportion:
    """``P(session drawdown >= level)`` with a Wilson 95% CI — a legible handle on the drawdown
    distribution (e.g. ``level=0.5`` = 'lost half the roll at some point'). D17."""
    if not 0.0 < level <= 1.0:
        raise ValueError(f"drawdown level must be in (0, 1], got {level}")
    if not captures:
        raise ValueError("drawdown_breach_probability needs at least one session")
    k = sum(1 for c in captures if session_max_drawdown(c) >= level)
    return wilson_interval(k, len(captures))
