"""Tests for blackjack_rl.env — the episode-capture wrapper.

Fast, deterministic wiring checks. The precise 'house edge ~ 0.45%' validation is a
statistical estimate needing many hands, so it lives in evaluation (Stage 2), not here;
this file only asserts a one-sided sanity bound that catches gross failures without flaking.
"""
import random

from strategies.basic_strategy import BasicStrategy
from simulator.game_state import Action
from blackjack_rl.env import Episode, rollout_many

_LEGAL: set[Action] = {"hit", "stand", "double", "split", "surrender"}


def test_episode_structure():
    random.seed(0)
    saw_steps = False
    for ep in rollout_many(BasicStrategy(), 300):
        assert isinstance(ep, Episode)
        assert isinstance(ep.reward, float)
        assert -10.0 < ep.reward < 10.0
        for key, action in ep.steps:
            assert isinstance(key, tuple) and len(key) == 3
            pv, soft, up = key
            assert isinstance(pv, int) and isinstance(soft, bool) and isinstance(up, int)
            assert action in _LEGAL
        if ep.steps:
            saw_steps = True
    assert saw_steps  # most hands involve at least one decision


def test_rollout_is_reproducible_under_seed():
    random.seed(123)
    a = [(ep.steps, ep.reward) for ep in rollout_many(BasicStrategy(), 200)]
    random.seed(123)
    b = [(ep.steps, ep.reward) for ep in rollout_many(BasicStrategy(), 200)]
    assert a == b


def test_basic_strategy_does_not_lose_catastrophically():
    # Smoke only: basic strategy's true edge is ~ -0.45%/hand, far above this bound. A
    # wiring/strategy failure (e.g. random play) loses far more and trips this. The tight
    # 0.45% check is an evaluation-time concern (Stage 2), not a unit test.
    random.seed(7)
    n = 10_000
    mean = sum(ep.reward for ep in rollout_many(BasicStrategy(), n)) / n
    assert mean > -0.10, f"mean reward/hand = {mean:.4f}, suspiciously low"
