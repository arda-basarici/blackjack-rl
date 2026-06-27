"""Tests for blackjack_rl.session.env — the Problem-B session capture driver (B0).

Fast, deterministic wiring checks: config correctness, bankroll bookkeeping, the two terminal
causes (ruin vs horizon), reproducibility, that counting is live, and bet clamping. Statistical
outcome/risk validation belongs to the B2+ metrics, not here.
"""
import math
import random

import pytest
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.session.bet_agent import FlatBet
from blackjack_rl.session.env import (
    BET_SPREAD,
    HandRecord,
    SessionCapture,
    SessionConfig,
    SessionEnv,
    growth_config,
    problem_b_config,
    ruin_config,
    run_sessions,
)


def test_problem_b_config_counting_on_shoe_persists():
    cfg = problem_b_config()
    assert cfg.card_counting_allowed is True   # the whole point of B
    assert cfg.shuffle_every_round is False     # shoe depletes across hands
    assert cfg.min_bet == 1.0                   # bankroll / spread in min-bet units
    assert cfg.num_decks == 6                   # inherits the vegas_strip anchor


def test_session_structure_and_invariants():
    random.seed(0)
    cap = SessionEnv(SessionConfig(max_hands=200)).run(BasicStrategy(), FlatBet(1.0))

    assert isinstance(cap, SessionCapture)
    assert cap.n_hands == len(cap.hands) >= 1
    assert cap.starting_bankroll == 100.0
    for rec in cap.hands:
        assert isinstance(rec, HandRecord)
        assert rec.bankroll_after == max(0.0, rec.bankroll_before + rec.payout)
        if rec.bankroll_after > 0:
            expected = math.log(rec.bankroll_after / rec.bankroll_before)
            assert rec.log_reward == pytest.approx(expected)
        else:
            assert rec.log_reward == float("-inf")
    # exactly the final record is terminal
    assert cap.hands[-1].done is True
    assert all(not rec.done for rec in cap.hands[:-1])


def test_bankroll_is_continuous_across_hands():
    random.seed(1)
    cap = SessionEnv(SessionConfig(max_hands=300)).run(BasicStrategy(), FlatBet(1.0))
    for prev, nxt in zip(cap.hands, cap.hands[1:]):
        assert nxt.bankroll_before == prev.bankroll_after
    assert cap.final_bankroll == cap.hands[-1].bankroll_after


def test_horizon_terminates_without_ruin():
    # huge bankroll, tiny flat bet -> cannot ruin; ends exactly at the horizon
    random.seed(2)
    cfg = SessionConfig(starting_bankroll=10_000.0, max_hands=50)
    cap = SessionEnv(cfg).run(BasicStrategy(), FlatBet(1.0))
    assert cap.ruined is False
    assert cap.n_hands == 50
    assert cap.final_bankroll > cfg.ruin_threshold


def test_ruin_terminates_early():
    # bet the whole (small) bankroll each hand -> a single loss is ruin; statistically certain
    cfg = SessionConfig(starting_bankroll=2.0, ruin_threshold=1.0, max_hands=500)
    saw_ruin = False
    for cap in run_sessions(cfg, BasicStrategy(), FlatBet(100.0), n=50):
        if cap.ruined:
            saw_ruin = True
            assert cap.final_bankroll <= cfg.ruin_threshold
            assert cap.hands[-1].done is True
            assert cap.n_hands < cfg.max_hands     # broke out before the horizon
    assert saw_ruin, "brutal all-in betting never produced a ruin — unexpected"


def test_total_wipeout_gives_neg_inf_log_reward():
    # an all-in loss drives bankroll exactly to 0 -> log(0) is -inf (B2 picks any finite penalty)
    cfg = SessionConfig(starting_bankroll=2.0, ruin_threshold=1.0, max_hands=500)
    saw_wipeout = False
    for cap in run_sessions(cfg, BasicStrategy(), FlatBet(100.0), n=50):
        last = cap.hands[-1]
        if cap.ruined and last.bankroll_after == 0.0:
            saw_wipeout = True
            assert last.log_reward == float("-inf")
    assert saw_wipeout, "all-in betting never wiped out to exactly 0 — expected the -inf branch"


def test_shoe_reshuffles_at_penetration():
    # over a long session the shoe must reshuffle: decks_remaining jumps back up (it otherwise
    # only decreases within a shoe). 6 decks @ 75% penetration -> a reshuffle well within 200 hands.
    random.seed(5)
    cap = SessionEnv(SessionConfig(max_hands=200)).run(BasicStrategy(), FlatBet(1.0))
    depths = [h.decks_remaining for h in cap.hands]
    assert any(nxt > prev for prev, nxt in zip(depths, depths[1:])), "shoe never reshuffled"


def test_run_sessions_is_reproducible_under_seed():
    cfg = SessionConfig(max_hands=100)

    def key(cap):
        return [(h.true_count, h.bet, h.payout, h.bankroll_after) for h in cap.hands]

    a = [key(c) for c in run_sessions(cfg, BasicStrategy(), FlatBet(1.0), n=5)]
    b = [key(c) for c in run_sessions(cfg, BasicStrategy(), FlatBet(1.0), n=5)]
    assert a == b


def test_counting_is_live():
    # a persistent shoe must produce a varying true count (not the frozen 0.0 of Problem A)
    random.seed(3)
    cap = SessionEnv(SessionConfig(max_hands=300)).run(BasicStrategy(), FlatBet(1.0))
    counts = [h.true_count for h in cap.hands]
    assert any(abs(tc) > 0.0 for tc in counts)
    assert min(counts) < 0.0 < max(counts)     # swings both ways as the shoe depletes


def test_bet_is_clamped_to_spread_and_bankroll():
    class _Greedy:
        """Always tries to bet absurdly high — must be clamped to the spread max, then bankroll."""

        def bet(self, *, true_count, decks_remaining, bankroll):
            return 10_000.0

    random.seed(4)
    cfg = SessionConfig(bet_spread=(1, 2, 4, 8), max_hands=80)
    cap = SessionEnv(cfg).run(BasicStrategy(), _Greedy())
    for rec in cap.hands:
        assert rec.bet <= max(cfg.bet_spread)
        assert rec.bet <= rec.bankroll_before


def test_named_configs_share_one_spread_differ_only_in_bankroll():
    # B2b: one fixed ladder across both regimes; only the bankroll sets the growth-vs-ruin axis.
    g, r = growth_config(), ruin_config()
    assert g.bet_spread == r.bet_spread == BET_SPREAD     # the single shared ladder
    assert g.starting_bankroll == 400.0                   # growth: spread top ~= full Kelly @ +6
    assert r.starting_bankroll == 200.0                   # ruin: 8u top ~= 2x Kelly -> over-bet headroom
    assert g.starting_bankroll > r.starting_bankroll      # growth regime is the fatter bankroll
    # the only difference is the bankroll (spread, ruin, horizon, seed all identical)
    assert (g.ruin_threshold, g.max_hands, g.seed) == (r.ruin_threshold, r.max_hands, r.seed)


def test_named_configs_are_valid_and_seedable():
    # construction runs __post_init__ validation; seed override threads through
    assert growth_config(seed=7).seed == 7
    assert ruin_config(seed=7).seed == 7


def test_bet_spread_is_the_fixed_arithmetic_ladder():
    assert BET_SPREAD == (1, 2, 3, 4, 5, 6, 7, 8)
    steps = [b - a for a, b in zip(BET_SPREAD, BET_SPREAD[1:])]
    assert all(s == 1 for s in steps), "BET_SPREAD must be arithmetic (uniform unit steps)"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"starting_bankroll": 1.0, "ruin_threshold": 1.0},  # starts ruined
        {"ruin_threshold": -1.0},
        {"ruin_threshold": 0.5},  # below the spread floor (1) -> could force a sub-minimum bet
        {"bet_spread": ()},
        {"bet_spread": (1, 0, 4)},
        {"max_hands": 0},
    ],
)
def test_session_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        SessionConfig(**kwargs)
