"""Tests for blackjack_rl.config — the ExperimentConfig contract."""
from dataclasses import FrozenInstanceError, asdict

import pytest

from blackjack_rl.config import ExperimentConfig


def test_defaults():
    cfg = ExperimentConfig(num_episodes=1000)
    assert cfg.num_episodes == 1000
    assert cfg.epsilon == 0.1
    assert cfg.seed == 42


def test_is_frozen():
    cfg = ExperimentConfig(num_episodes=1000)
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 7  # type: ignore[misc]


def test_serializes_to_dict():
    cfg = ExperimentConfig(num_episodes=500, epsilon=0.05, seed=7)
    assert asdict(cfg) == {"num_episodes": 500, "epsilon": 0.05, "seed": 7}


def test_rejects_bad_num_episodes():
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=0)


def test_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=1000, epsilon=1.5)
