"""Tests for blackjack_rl.training.monte_carlo — mechanics only.

Whether the agent learns *good blackjack* is a statistical claim (house edge, policy-diff vs
basic strategy) and is validated in the evaluation sub-unit, not here. These tests check the
credit-assignment maths and that training is deterministic.
"""
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode
from blackjack_rl.training.monte_carlo import _apply_episode, train


def test_apply_episode_assigns_terminal_reward_to_every_step() -> None:
    agent = TabularAgent()
    episode = Episode(steps=[((12, False, 7), "hit"), ((16, False, 7), "stand")], reward=-1.0)
    _apply_episode(agent, episode)
    assert agent.q[((12, False, 7), "hit")] == -1.0
    assert agent.q[((16, False, 7), "stand")] == -1.0
    assert agent.n[((12, False, 7), "hit")] == 1
    assert agent.n[((16, False, 7), "stand")] == 1


def test_apply_episode_averages_repeated_state_action() -> None:
    agent = TabularAgent()
    ka = ((20, False, 10), "stand")
    _apply_episode(agent, Episode(steps=[((20, False, 10), "stand")], reward=1.0))
    _apply_episode(agent, Episode(steps=[((20, False, 10), "stand")], reward=-1.0))
    assert agent.n[ka] == 2
    assert abs(agent.q[ka]) < 1e-12  # mean of {+1, -1}


def test_train_populates_tables() -> None:
    agent = train(ExperimentConfig(num_episodes=2000, seed=42))
    assert agent.q and agent.n
    assert all(count >= 1 for count in agent.n.values())


def test_train_is_reproducible_under_seed() -> None:
    a = train(ExperimentConfig(num_episodes=1000, seed=7))
    b = train(ExperimentConfig(num_episodes=1000, seed=7))
    assert a.q == b.q
    assert a.n == b.n
