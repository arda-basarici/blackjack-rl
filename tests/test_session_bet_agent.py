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
import torch
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.session.bet_agent import (
    FEATURE_DIM,
    BetAgent,
    FlatBet,
    KellyBet,
    bet_feature_dim,
    encode_bet_state,
    session_to_transitions,
)
from blackjack_rl.session.env import (
    BET_SPREAD,
    HandRecord,
    IndexedBetPolicy,
    SessionCapture,
    SessionConfig,
    SessionEnv,
    run_sessions,
)
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


# --- BetAgent (B2d): encoding -------------------------------------------------------

def test_encode_bet_state_shape_and_normalization():
    """Three normalized scalars: tc/TC_SCALE, remaining-shoe fraction (decks/num_decks), bankroll/scale."""
    feats = encode_bet_state(10.0, 6.0, 400.0, num_decks=6.0, bankroll_scale=400.0)
    assert len(feats) == FEATURE_DIM == 3
    assert feats == pytest.approx([1.0, 1.0, 1.0])
    # remaining fraction is decks_remaining / num_decks in (0, 1]; bankroll on the fixed absolute scale
    half = encode_bet_state(-5.0, 3.0, 200.0, num_decks=6.0, bankroll_scale=400.0)
    assert half == pytest.approx([-0.5, 0.5, 0.5])


# --- BetAgent (B2d): the bankroll-feature encoding ablation ---------------------------

def test_bet_feature_dim_follows_the_flag():
    assert bet_feature_dim("raw") == 3
    assert bet_feature_dim("logratio") == 3
    assert bet_feature_dim("none") == 2  # bankroll dropped -> count + depth only
    with pytest.raises(ValueError, match="unknown bankroll_feature"):
        bet_feature_dim("bogus")


def test_encode_bet_state_bankroll_feature_variants():
    import math

    kw = dict(num_decks=6.0, bankroll_scale=400.0)
    raw = encode_bet_state(0.0, 3.0, 200.0, bankroll_feature="raw", **kw)
    assert raw == pytest.approx([0.0, 0.5, 0.5])                       # bankroll/scale = 200/400
    logr = encode_bet_state(0.0, 3.0, 200.0, bankroll_feature="logratio", **kw)
    assert logr == pytest.approx([0.0, 0.5, math.log(0.5)])            # log(200/400)
    dropped = encode_bet_state(0.0, 3.0, 200.0, bankroll_feature="none", **kw)
    assert dropped == pytest.approx([0.0, 0.5])                        # bankroll gone -> 2 features
    with pytest.raises(ValueError, match="unknown bankroll_feature"):
        encode_bet_state(0.0, 3.0, 200.0, bankroll_feature="bogus", **kw)


def test_bet_agent_none_feature_has_two_input_net():
    """A dropped-bankroll agent's net takes 2 inputs, and encode_state returns 2 features — still runs."""
    agent = BetAgent(levels=(1, 2, 3), bankroll_feature="none")
    assert agent.q_net.net[0].in_features == 2
    assert len(agent.encode_state(2.0, 3.0, 400.0)) == 2
    assert agent.q_values(true_count=2.0, decks_remaining=3.0, bankroll=400.0).shape == (3,)


# --- BetAgent (B2d): action selection ------------------------------------------------

def test_bet_agent_q_values_width_matches_levels():
    agent = BetAgent(levels=(1, 2, 3))
    q = agent.q_values(true_count=2.0, decks_remaining=3.0, bankroll=400.0)
    assert q.shape == (3,)


def test_bet_agent_greedy_and_bet_agree():
    """bet() returns levels[greedy_level]; greedy_level is the argmax of the Q-vector."""
    torch.manual_seed(0)
    agent = BetAgent()
    kw = dict(true_count=3.0, decks_remaining=2.5, bankroll=400.0)
    idx = agent.greedy_level(**kw)
    q = agent.q_values(**kw)
    assert idx == int(torch.argmax(q).item())
    assert agent.bet(**kw) == agent.levels[idx]


def test_bet_agent_select_level_greedy_when_epsilon_zero():
    torch.manual_seed(0)
    agent = BetAgent(epsilon=0.0)
    kw = dict(true_count=4.0, decks_remaining=1.0, bankroll=300.0)
    assert agent.select_level(**kw) == agent.greedy_level(**kw)


def test_bet_agent_select_level_explores_within_range():
    """epsilon=1 always explores; the random pick is still a valid index over the menu."""
    import random as _random

    torch.manual_seed(0)
    _random.seed(1)
    agent = BetAgent(levels=(1, 2, 3, 4), epsilon=1.0)
    picks = {agent.select_level(true_count=0.0, decks_remaining=3.0, bankroll=400.0) for _ in range(50)}
    assert picks  # non-empty
    assert all(0 <= p < len(agent.levels) for p in picks)
    assert len(picks) > 1  # genuinely exploring, not stuck


def test_bet_agent_weights_reproducible_from_seed():
    """Same torch seed -> identical weights -> identical greedy policy (reproducible by construction)."""
    torch.manual_seed(7)
    a = BetAgent()
    torch.manual_seed(7)
    b = BetAgent()
    kw = dict(true_count=5.0, decks_remaining=2.0, bankroll=400.0)
    assert a.greedy_level(**kw) == b.greedy_level(**kw)
    assert torch.equal(a.q_values(**kw), b.q_values(**kw))


def test_bet_agent_rejects_empty_levels():
    with pytest.raises(ValueError):
        BetAgent(levels=())


# --- IndexedBetPolicy: who is indexed ------------------------------------------------

def test_indexed_bet_policy_membership():
    """BetAgent is an IndexedBetPolicy (it can report its chosen index); the analytic baselines are not."""
    assert isinstance(BetAgent(), IndexedBetPolicy)
    assert not isinstance(FlatBet(1.0), IndexedBetPolicy)
    assert not isinstance(KellyBet({0: 0.0}), IndexedBetPolicy)


# --- env records the chosen index ----------------------------------------------------

def test_env_records_bet_level_for_indexed_bettor():
    """Driven through select_level, every HandRecord carries a valid level index (and, absent clamping,
    the wager equals levels[bet_level])."""
    torch.manual_seed(0)
    agent = BetAgent(epsilon=0.0)
    cap = next(run_sessions(SessionConfig(starting_bankroll=400.0, max_hands=10, seed=1), BasicStrategy(), agent, 1))
    assert cap.n_hands > 0
    for rec in cap.hands:
        assert rec.bet_level is not None
        assert 0 <= rec.bet_level < len(agent.levels)
        assert rec.bet == agent.levels[rec.bet_level]  # growth bankroll: no clamp


def test_env_leaves_bet_level_none_for_plain_bettor():
    cap = next(run_sessions(SessionConfig(starting_bankroll=400.0, max_hands=10, seed=1), BasicStrategy(), FlatBet(2.0), 1))
    assert all(rec.bet_level is None for rec in cap.hands)


class _CountingIndexedBettor:
    """A minimal IndexedBetPolicy that counts how often each method is called — to prove the env draws
    the action exactly once per hand (via select_level) and never double-draws through bet()."""

    levels = (1.0, 2.0)

    def __init__(self) -> None:
        self.select_calls = 0
        self.bet_calls = 0

    def select_level(self, *, true_count: float, decks_remaining: float, bankroll: float) -> int:
        self.select_calls += 1
        return 0

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        self.bet_calls += 1
        return 1.0


def test_env_draws_action_once_per_hand_no_double_draw():
    bettor = _CountingIndexedBettor()
    assert isinstance(bettor, IndexedBetPolicy)
    cap = SessionEnv(SessionConfig(starting_bankroll=400.0, max_hands=5, seed=1)).run(BasicStrategy(), bettor)
    assert bettor.select_calls == cap.n_hands  # one ε-greedy draw per hand
    assert bettor.bet_calls == 0  # never reached the plain-bettor path


# --- session_to_transitions (B2d): reconstruction ------------------------------------

def _rec(bet_level, bet, *, log_reward=0.0, done=False, bankroll_before=400.0):
    return HandRecord(
        true_count=2.0, decks_remaining=3.0, bankroll_before=bankroll_before, bet=bet,
        payout=0.0, bankroll_after=bankroll_before, log_reward=log_reward, done=done, bet_level=bet_level,
    )


def test_session_to_transitions_from_real_capture():
    torch.manual_seed(0)
    agent = BetAgent(epsilon=0.0)
    cap = next(run_sessions(SessionConfig(starting_bankroll=400.0, max_hands=8, seed=1), BasicStrategy(), agent, 1))
    trs = session_to_transitions(cap, encode=agent.encode_state, n_levels=len(agent.levels))
    assert len(trs) == cap.n_hands
    assert [t.action for t in trs] == [rec.bet_level for rec in cap.hands]
    assert all(t.state.shape == (FEATURE_DIM,) for t in trs)
    # exactly the last hand is terminal; its mask is all-False, others all-True (every level always legal)
    assert trs[-1].done and not trs[-1].next_legal_mask.any()
    assert all(not t.done and t.next_legal_mask.all() for t in trs[:-1])


def test_session_to_transitions_action_is_chosen_index_not_clamped_wager():
    """The clamp-zone guarantee (decision 1a): with bankroll < top level the env clamps the *wager*, but
    the transition's action is the *chosen index*, not a reverse-map of the clamped wager."""
    agent = BetAgent()  # 8 levels (1..8)
    # chose the top level (index 7 -> 8u) but only 5u was held, so bet was clamped to 5.0 (not a level)
    cap = SessionCapture(hands=[_rec(7, 5.0, bankroll_before=5.0, done=True)], ruined=False,
                         final_bankroll=5.0, starting_bankroll=5.0)
    trs = session_to_transitions(cap, encode=agent.encode_state, n_levels=len(agent.levels))
    assert trs[0].action == 7  # the chosen index, exact — not derived from the 5.0 clamped wager


def test_session_to_transitions_clips_wipeout_to_ruin_reward():
    agent = BetAgent()
    cap = SessionCapture(hands=[_rec(0, 1.0, log_reward=float("-inf"), done=True, bankroll_before=1.0)],
                         ruined=True, final_bankroll=0.0, starting_bankroll=100.0)
    trs = session_to_transitions(cap, encode=agent.encode_state, n_levels=len(agent.levels), ruin_reward=-3.0)
    assert trs[0].reward == -3.0


def test_session_to_transitions_requires_indexed_capture():
    """A plain-bettor capture (bet_level None) cannot be reconstructed — fail loud, not silent."""
    agent = BetAgent()
    cap = SessionCapture(
        hands=[HandRecord(true_count=1.0, decks_remaining=3.0, bankroll_before=400.0, bet=2.0,
                          payout=0.0, bankroll_after=400.0, log_reward=0.0, done=True)],
        ruined=False, final_bankroll=400.0, starting_bankroll=400.0,
    )
    with pytest.raises(ValueError, match="IndexedBetPolicy"):
        session_to_transitions(cap, encode=agent.encode_state, n_levels=len(agent.levels))
