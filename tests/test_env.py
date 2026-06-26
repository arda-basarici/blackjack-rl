"""Tests for blackjack_rl.core.env — the episode-capture wrapper.

Fast, deterministic wiring checks. The precise house-edge validation is statistical and lives
in evaluation; here we only assert structure, determinism, a one-sided sanity bound, and the
per-decision / per-sub-hand return behaviour.
"""
import random

from strategies.basic_strategy import BasicStrategy
from simulator.game_state import Action
from blackjack_rl.tabular.agent import TabularAgent
from blackjack_rl.core.env import Episode, rollout_many

_LEGAL: set[Action] = {"hit", "stand", "double", "split", "surrender"}


def test_episode_structure():
    random.seed(0)
    saw_steps = False
    for ep in rollout_many(BasicStrategy(), 300):
        assert isinstance(ep, Episode)
        assert isinstance(ep.reward, float)
        assert -10.0 < ep.reward < 10.0
        for key, action, step_return in ep.steps:
            assert isinstance(key, tuple) and len(key) == 3   # no-split mode -> 3-tuple
            assert action in _LEGAL
            assert isinstance(step_return, float)
        if ep.steps:
            saw_steps = True
    assert saw_steps


def test_rollout_is_reproducible_under_seed():
    random.seed(123)
    a = [(ep.steps, ep.reward) for ep in rollout_many(BasicStrategy(), 200)]
    random.seed(123)
    b = [(ep.steps, ep.reward) for ep in rollout_many(BasicStrategy(), 200)]
    assert a == b


def test_basic_strategy_does_not_lose_catastrophically():
    random.seed(7)
    n = 10_000
    mean = sum(ep.reward for ep in rollout_many(BasicStrategy(), n)) / n
    assert mean > -0.10, f"mean reward/hand = {mean:.4f}, suspiciously low"


def test_nonsplitting_policy_gives_each_step_the_hand_total():
    # a non-splitting policy -> single-chain hand -> every step's return is the hand total
    # (this is what keeps no-split training identical to before the per-decision refactor)
    random.seed(0)
    agent = TabularAgent(epsilon=1.0)  # explores hit/stand/double, never splits
    for ep in rollout_many(agent, 300):
        for _key, _action, step_return in ep.steps:
            assert step_return == ep.reward


def test_split_episodes_carry_per_subhand_returns():
    random.seed(0)
    saw_split = False
    saw_differing_return = False
    for ep in rollout_many(BasicStrategy(), 4000, with_splits=True):
        for key, action, _r in ep.steps:
            assert len(key) == 4                      # split mode -> 4-tuple keys
        actions = [a for _k, a, _r in ep.steps]
        if "split" in actions:
            saw_split = True
            if any(r != ep.reward for _k, _a, r in ep.steps):
                saw_differing_return = True
    assert saw_split, "basic strategy should split some pairs in 4000 hands"
    assert saw_differing_return, "split hands should show per-sub-hand returns != the total"
