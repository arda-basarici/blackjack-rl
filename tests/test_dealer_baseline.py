"""Tests for the dealer control-variate reward baselines (evaluation/dealer_baseline) and their
wiring into hand_to_transitions: distribution sanity, unbiasedness (mean-zero), score, config,
and that the terminal reward is adjusted correctly (stand baseline drives a stand reward to V_stand)."""
from __future__ import annotations

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.env import CapturedHand, Step
from blackjack_rl.evaluation.dealer_baseline import (
    baseline, dealer_outcome_dist, score, stand_value,
)
from blackjack_rl.training.deep_q import hand_to_transitions


def test_dealer_distribution_normalized_and_plausible() -> None:
    for up in range(2, 12):
        d = dealer_outcome_dist(up)
        assert abs(sum(d.values()) - 1.0) < 1e-9
        assert 0.1 < d[0] < 0.5          # bust probability is in a sane range


def test_baselines_are_mean_zero() -> None:
    for kind in ("bust", "stand"):
        for up in (4, 6, 10, 11):
            d = dealer_outcome_dist(up)
            e = sum(p * baseline(kind, start_total=18, upcard=up, dealer_final=df) for df, p in d.items())
            assert abs(e) < 1e-9          # unbiased: EV preserved


def test_none_is_zero_and_score_values() -> None:
    assert baseline("none", start_total=18, upcard=6, dealer_final=0) == 0.0
    assert score(20, 0) == 1.0           # dealer bust -> win
    assert score(18, 19) == -1.0
    assert score(18, 18) == 0.0
    assert score(22, 17) == -1.0         # player bust -> lose


def test_reward_baseline_config_validation() -> None:
    DQNConfig(num_episodes=10, reward_baseline="stand")
    DQNConfig(num_episodes=10, reward_baseline="bust")
    try:
        DQNConfig(num_episodes=10, reward_baseline="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for reward_baseline='bogus'")


def test_hand_to_transitions_applies_stand_baseline() -> None:
    agent = DQNAgent(epsilon=0.0, encoding="onehot")
    step = Step(player_value=18, player_is_soft=False, dealer_upcard=6, can_split=False,
                can_double=False, action="stand", final_dealer_value=20)
    hand = CapturedHand(steps=[step], reward=-1.0)  # stood on 18, dealer made 20 -> lose

    t_none = hand_to_transitions(hand, agent.actions, encoding="onehot", reward_baseline="none")
    assert abs(float(t_none[0].reward) - (-1.0)) < 1e-6

    # stand baseline: reward - (score(18,20) - V_stand(18,6)); for a stand this is exactly V_stand
    t_stand = hand_to_transitions(hand, agent.actions, encoding="onehot", reward_baseline="stand")
    assert abs(float(t_stand[0].reward) - stand_value(18, 6)) < 1e-6
