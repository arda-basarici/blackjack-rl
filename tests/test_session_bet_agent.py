"""Tests for the Problem-B bet policies (DESIGN D17): ``FlatBet`` (rung 1) and ``KellyBet`` (rung 2).

``KellyBet`` is the analytic baseline the learned bettor is audited against, so its contract matters:
- discrete mode snaps to the *nearest* spread level (the DQN's menu → fair comparison, decision A),
- continuous mode returns the raw ``f*·bankroll`` ceiling,
- ``kelly_fraction`` scales the wager (fractional / over-Kelly),
- ``f*`` reads the nearest measured bucket and clamps outside the range,
- a non-positive edge (``f*=0``) still bets the table minimum (the mandatory bet).
Unit tests use synthetic curves (deterministic); one integration test plugs it into the real env.
"""
import pytest
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.session.bet_agent import FlatBet, KellyBet
from blackjack_rl.session.env import BET_SPREAD, SessionConfig, run_sessions
from blackjack_rl.session.references import load_edge_reference


def _f(true_count: float, kb: KellyBet) -> float:
    """Read KellyBet's f* at a count by betting on a 1-unit bankroll in continuous mode."""
    return kb.bet(true_count=true_count, decks_remaining=3.0, bankroll=1.0)


# --- FlatBet -------------------------------------------------------------------------

def test_flat_bet_is_constant():
    fb = FlatBet(3.0)
    assert fb.bet(true_count=8.0, decks_remaining=1.0, bankroll=500.0) == 3.0
    assert fb.bet(true_count=-5.0, decks_remaining=5.0, bankroll=10.0) == 3.0


def test_flat_bet_rejects_nonpositive():
    with pytest.raises(ValueError):
        FlatBet(0.0)
    with pytest.raises(ValueError):
        FlatBet(-1.0)


# --- KellyBet: sizing ----------------------------------------------------------------

def test_kelly_discrete_snaps_to_nearest_level():
    """desired = f*·bankroll, then snap to the nearest spread level (the DQN's menu)."""
    kb = KellyBet({4: 0.0124})  # default spread 1..8
    # 0.0124 * 400 = 4.96 -> nearest level is 5
    assert kb.bet(true_count=4.0, decks_remaining=3.0, bankroll=400.0) == 5.0
    # 0.0124 * 100 = 1.24 -> nearest level is 1
    assert kb.bet(true_count=4.0, decks_remaining=3.0, bankroll=100.0) == 1.0


def test_kelly_continuous_returns_raw_fraction():
    """The analytic ceiling: f*·bankroll unrounded (the env, not the bettor, bounds it)."""
    kb = KellyBet({4: 0.0124}, discretize=False)
    assert kb.bet(true_count=4.0, decks_remaining=3.0, bankroll=400.0) == pytest.approx(4.96)


def test_kelly_fraction_scales_the_bet():
    kb_full = KellyBet({4: 0.01}, discretize=False)
    kb_half = KellyBet({4: 0.01}, kelly_fraction=0.5, discretize=False)
    assert kb_half.bet(true_count=4.0, decks_remaining=3.0, bankroll=400.0) == pytest.approx(
        0.5 * kb_full.bet(true_count=4.0, decks_remaining=3.0, bankroll=400.0)
    )


def test_kelly_over_betting_allowed():
    """c > 1 (over-Kelly) is permitted — used for the ruin-config over-betting experiments."""
    kb = KellyBet({4: 0.0124}, kelly_fraction=2.0, discretize=False)
    assert kb.bet(true_count=4.0, decks_remaining=3.0, bankroll=400.0) == pytest.approx(9.92)


def test_kelly_nonpositive_edge_bets_the_minimum():
    """f* = 0 at a non-positive edge -> the snap lands on the spread minimum: the mandatory table bet
    a counter must place even when Kelly says don't (the underbet that makes full-Kelly net-negative
    at modest bankrolls)."""
    kb = KellyBet({0: 0.0})
    assert kb.bet(true_count=0.0, decks_remaining=3.0, bankroll=400.0) == float(min(BET_SPREAD))
    kb_custom = KellyBet({0: 0.0}, spread=(2, 4, 6))
    assert kb_custom.bet(true_count=0.0, decks_remaining=3.0, bankroll=400.0) == 2.0


# --- KellyBet: f* lookup -------------------------------------------------------------

def test_kelly_fstar_uses_nearest_bucket_and_clamps():
    kb = KellyBet({2: 0.01, 6: 0.02}, discretize=False)
    assert _f(2.0, kb) == pytest.approx(0.01)   # exact
    assert _f(2.4, kb) == pytest.approx(0.01)   # rounds to 2
    assert _f(3.0, kb) == pytest.approx(0.01)   # nearest key to 3 is 2 (not 6)
    assert _f(10.0, kb) == pytest.approx(0.02)  # above range -> clamp to top bucket
    assert _f(-5.0, kb) == pytest.approx(0.01)  # below range -> clamp to bottom bucket


# --- KellyBet: validation ------------------------------------------------------------

def test_kelly_rejects_bad_construction():
    with pytest.raises(ValueError):
        KellyBet({})  # empty curve
    with pytest.raises(ValueError):
        KellyBet({0: 0.0}, kelly_fraction=0.0)
    with pytest.raises(ValueError):
        KellyBet({0: 0.0}, kelly_fraction=-1.0)
    with pytest.raises(ValueError):
        KellyBet({0: 0.0}, spread=())


# --- KellyBet: integration (plugs into the real env as a BetPolicy) ------------------

def test_kelly_runs_as_a_bet_policy():
    """The committed curve + KellyBet drive real sessions through run_sessions without error — confirms
    the BetPolicy protocol fit end to end."""
    kb = KellyBet(load_edge_reference().kelly_curve)
    caps = list(
        run_sessions(
            SessionConfig(starting_bankroll=400.0, max_hands=20, seed=1), BasicStrategy(), kb, 2
        )
    )
    assert len(caps) == 2
    assert all(c.n_hands > 0 for c in caps)
