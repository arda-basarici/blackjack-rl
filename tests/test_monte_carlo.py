"""Tests for blackjack_rl.training.monte_carlo — mechanics only.

Whether the agent learns *good blackjack* is validated in the evaluation sub-unit, not here.
These check credit assignment (per-step returns), determinism, the learning curve, and that
splits train.
"""
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode
from blackjack_rl.training.monte_carlo import _apply_episode, train


def test_apply_episode_applies_each_step_return() -> None:
    agent = TabularAgent()
    episode = Episode(
        steps=[((12, False, 7), "hit", -1.0), ((16, False, 7), "stand", -1.0)], reward=-1.0
    )
    _apply_episode(agent, episode)
    assert agent.q[((12, False, 7), "hit")] == -1.0
    assert agent.q[((16, False, 7), "stand")] == -1.0
    assert agent.n[((12, False, 7), "hit")] == 1


def test_apply_episode_uses_distinct_per_step_returns() -> None:
    # the split fix: each step gets its OWN return, not one shared total
    agent = TabularAgent()
    ep = Episode(
        steps=[((16, False, 6), "split", 0.5), ((11, False, 6), "stand", 1.0),
               ((9, False, 6), "stand", -0.5)],
        reward=0.5,
    )
    _apply_episode(agent, ep)
    assert agent.q[((16, False, 6), "split")] == 0.5
    assert agent.q[((11, False, 6), "stand")] == 1.0
    assert agent.q[((9, False, 6), "stand")] == -0.5


def test_apply_episode_averages_repeated_state_action() -> None:
    agent = TabularAgent()
    ka = ((20, False, 10), "stand")
    _apply_episode(agent, Episode(steps=[((20, False, 10), "stand", 1.0)], reward=1.0))
    _apply_episode(agent, Episode(steps=[((20, False, 10), "stand", -1.0)], reward=-1.0))
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


def test_train_emits_learning_curve() -> None:
    curve: list[dict] = []
    train(ExperimentConfig(num_episodes=2000, seed=1), progress_every=500, on_checkpoint=curve.append)
    assert [p["episode"] for p in curve] == [500, 1000, 1500, 2000]
    expected_keys = {"episode", "epsilon", "policy_churn", "min_state_visits", "states"}
    for point in curve:
        assert expected_keys <= point.keys()
        assert point["policy_churn"] >= 0


def test_train_with_splits_uses_4tuple_keys_and_learns_split() -> None:
    agent = train(ExperimentConfig(num_episodes=3000, with_splits=True, seed=1))
    assert agent.with_splits is True
    assert any(len(k) == 4 for (k, _a) in agent.n)        # pairs encountered -> 4-tuple keys
    assert any(a == "split" for (_k, a) in agent.n)       # split action explored/learned
