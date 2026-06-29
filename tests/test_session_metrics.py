"""Tests for the Problem-B outcome & risk metrics (DESIGN D17, B2c).

Covers the contract each metric promises, with the edge cases that motivated the design:
- ``wilson_interval`` — behaves at the boundary (p=0/1) where Wald collapses; symmetric; validated.
- ``growth_rate``     — telescopes to log(W_final/W_0)/N; wipeouts dropped from growth, counted as ruin.
- the distributions  — quantiles/mean/min/max; the drawdown anchor is the *initial* bankroll.
All built on hand-constructed ``SessionCapture``s — pure, no engine.
"""
from math import isinf, log

import pytest

from blackjack_rl.session.env import HandRecord, SessionCapture
from blackjack_rl.session.metrics import (
    bankroll_distribution,
    bankroll_distribution_of,
    drawdown_breach_probability,
    drawdown_distribution,
    growth_rate,
    growth_rate_of,
    ruin_probability,
    session_growth_rate,
    session_max_drawdown,
    wilson_interval,
)


def _session(path: list[float], ruined: bool = False) -> SessionCapture:
    """Build a SessionCapture from a bankroll path ``[W0, W1, ..., WN]`` (N hands)."""
    hands = [
        HandRecord(
            true_count=0.0,
            decks_remaining=3.0,
            bankroll_before=before,
            bet=1.0,
            payout=after - before,
            bankroll_after=after,
            log_reward=log(after / before) if after > 0 else float("-inf"),
            done=False,
        )
        for before, after in zip(path, path[1:])
    ]
    return SessionCapture(
        hands=hands, ruined=ruined, final_bankroll=path[-1], starting_bankroll=path[0]
    )


# --- wilson_interval -----------------------------------------------------------------

def test_wilson_does_not_collapse_at_zero():
    """The whole reason we use Wilson: at k=0 it keeps a positive upper bound (~z^2/n), where Wald
    would give 0 +/- 0 (false certainty)."""
    w = wilson_interval(0, 100)
    assert w.estimate == 0.0
    assert w.low == pytest.approx(0.0, abs=1e-9)  # 0 up to float rounding, never negative
    assert w.low >= 0.0
    assert 0.02 < w.high < 0.05  # ~3.84/100, the honest 'at most this often'


def test_wilson_does_not_collapse_at_full():
    w = wilson_interval(100, 100)
    assert w.estimate == 1.0
    assert w.high == 1.0  # clamped at 1
    assert 0.95 < w.low < 1.0


def test_wilson_is_symmetric():
    """k/n and (n-k)/n are mirror images about 0.5."""
    a = wilson_interval(30, 100)
    b = wilson_interval(70, 100)
    assert a.low == pytest.approx(1 - b.high)
    assert a.high == pytest.approx(1 - b.low)


def test_wilson_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)  # k > n
    with pytest.raises(ValueError):
        wilson_interval(0, 0)  # n < 1


# --- growth rate (outcome) -----------------------------------------------------------

def test_session_growth_rate_telescopes():
    """g_i = mean per-hand log-increment = log(W_final/W_0)/N, regardless of the path between."""
    assert session_growth_rate(_session([100, 50, 200])) == pytest.approx(log(2) / 2)


def test_session_growth_rate_is_minus_inf_on_wipeout():
    assert isinf(session_growth_rate(_session([100, 0]))) is True
    assert session_growth_rate(_session([100, 0])) < 0


def test_session_growth_rate_empty_session_is_zero():
    assert session_growth_rate(_session([100])) == 0.0  # no hands played


def test_growth_rate_aggregates_over_sessions_with_ci():
    captures = [_session([100, 200]), _session([100, 110])]
    est = growth_rate(captures)
    assert est.value == pytest.approx((log(2) + log(1.1)) / 2)
    assert est.n == 2
    assert est.low < est.value < est.high  # a real, non-degenerate CI


def test_growth_rate_drops_wipeouts_from_estimate():
    """A wiped session (g_i = -inf) is excluded from growth; Estimate.n reveals the drop count, so the
    caller can surface it (it's accounted for on the ruin axis, D2 option a)."""
    captures = [_session([100, 200]), _session([100, 0], ruined=True)]
    est = growth_rate(captures)
    assert est.n == 1  # only the finite session
    assert est.value == pytest.approx(log(2))
    assert len(captures) - est.n == 1  # one wipeout dropped, not silently averaged to -inf


def test_growth_rate_empty_raises():
    with pytest.raises(ValueError):
        growth_rate([])


def test_scalar_cores_match_capture_path():
    """The scalar entry points (used by the parallel runner) agree with the capture-based functions —
    same single implementation, so a worker reducing in-process gives identical results."""
    captures = [_session([100, 200]), _session([100, 0], ruined=True), _session([100, 150])]
    assert growth_rate_of([session_growth_rate(c) for c in captures]) == growth_rate(captures)
    assert bankroll_distribution_of([c.final_bankroll for c in captures]) == bankroll_distribution(
        captures
    )


def test_growth_rate_of_drops_minus_inf():
    from math import log

    est = growth_rate_of([log(2), float("-inf"), log(1.1)])
    assert est.n == 2  # the -inf wipeout is dropped
    assert est.value == pytest.approx((log(2) + log(1.1)) / 2)


# --- ruin probability (risk) ---------------------------------------------------------

def test_ruin_probability_counts_ruined_with_wilson():
    captures = [_session([100, 50], ruined=True), _session([100, 200]), _session([100, 150])]
    p = ruin_probability(captures)
    assert p.k == 1
    assert p.n == 3
    assert p.estimate == pytest.approx(1 / 3)
    assert p.low < p.estimate < p.high


def test_ruin_probability_empty_raises():
    with pytest.raises(ValueError):
        ruin_probability([])


# --- drawdown (risk) -----------------------------------------------------------------

def test_session_drawdown_is_trough_vs_initial():
    assert session_max_drawdown(_session([100, 50, 200])) == pytest.approx(0.5)


def test_session_drawdown_zero_when_never_below_start():
    """Initial-anchored: a run-up that never dips below W_0 has zero drawdown (peak-to-trough would
    score the 200->150 dip; we don't, because the risk is survival vs the barrier)."""
    assert session_max_drawdown(_session([100, 200, 150])) == 0.0


def test_drawdown_breach_probability_thresholds():
    captures = [_session([100, 100]), _session([100, 50, 100]), _session([100, 20])]  # D = 0, .5, .8
    p = drawdown_breach_probability(captures, level=0.5)
    assert p.k == 2  # the 0.5 and 0.8 sessions breach
    assert p.estimate == pytest.approx(2 / 3)


def test_drawdown_breach_rejects_bad_level():
    caps = [_session([100, 100])]
    with pytest.raises(ValueError):
        drawdown_breach_probability(caps, level=0.0)
    with pytest.raises(ValueError):
        drawdown_breach_probability(caps, level=1.5)


def test_drawdown_distribution_shape():
    caps = [_session([100, 100]), _session([100, 50, 100]), _session([100, 20])]  # D = 0, .5, .8
    d = drawdown_distribution(caps, quantiles=(0.5,))
    assert d.n == 3
    assert d.minimum == pytest.approx(0.0)
    assert d.maximum == pytest.approx(0.8)
    assert d.quantiles[0.5] == pytest.approx(0.5)  # median of {0, .5, .8}


# --- bankroll distribution (outcome shape) -------------------------------------------

def test_bankroll_distribution_quantiles_and_summary():
    caps = [_session([100, 100]), _session([100, 200]), _session([100, 300])]
    d = bankroll_distribution(caps, quantiles=(0.1, 0.5, 0.9))
    assert d.n == 3
    assert d.mean == pytest.approx(200.0)
    assert d.minimum == 100.0
    assert d.maximum == 300.0
    assert d.quantiles[0.5] == pytest.approx(200.0)
    assert set(d.quantiles) == {0.1, 0.5, 0.9}


def test_distributions_empty_raise():
    with pytest.raises(ValueError):
        bankroll_distribution([])
    with pytest.raises(ValueError):
        drawdown_distribution([])
