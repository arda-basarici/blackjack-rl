"""Tests for blackjack_rl.core.config — the ExperimentConfig contract."""
from dataclasses import FrozenInstanceError, asdict

import pytest

from blackjack_rl.core.config import ExperimentConfig


def test_defaults() -> None:
    cfg = ExperimentConfig(num_episodes=1000)
    assert cfg.num_episodes == 1000
    assert cfg.epsilon == 0.1
    assert cfg.epsilon_schedule == "constant"
    assert cfg.epsilon_start == 0.3
    assert cfg.epsilon_end == 0.0
    assert cfg.step_size is None
    assert cfg.with_splits is False
    assert cfg.seed == 42


def test_is_frozen() -> None:
    cfg = ExperimentConfig(num_episodes=1000)
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 7  # type: ignore[misc]


def test_serializes_to_dict() -> None:
    cfg = ExperimentConfig(num_episodes=500, epsilon=0.05, seed=7)
    assert asdict(cfg) == {
        "num_episodes": 500,
        "epsilon": 0.05,
        "epsilon_schedule": "constant",
        "epsilon_start": 0.3,
        "epsilon_end": 0.0,
        "step_size": None,
        "with_splits": False,
        "seed": 7,
    }


def test_with_splits_can_be_enabled() -> None:
    assert ExperimentConfig(num_episodes=10, with_splits=True).with_splits is True


def test_rejects_bad_num_episodes() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=0)


def test_rejects_bad_epsilon() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=1000, epsilon=1.5)


def test_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=1000, epsilon_schedule="bogus")


def test_rejects_bad_step_size() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=1000, step_size=0.0)
    with pytest.raises(ValueError):
        ExperimentConfig(num_episodes=1000, step_size=1.5)
