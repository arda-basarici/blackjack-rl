"""Tests for blackjack_rl.schedules."""
import pytest

from blackjack_rl.config import ExperimentConfig
from blackjack_rl.schedules import make_epsilon_schedule
from blackjack_rl.training.monte_carlo import train


def test_constant_is_flat() -> None:
    f = make_epsilon_schedule("constant", constant=0.1, start=0.3, end=0.0, num_episodes=100)
    assert f(0) == 0.1 and f(50) == 0.1 and f(99) == 0.1


def test_linear_hits_endpoints_and_midpoint() -> None:
    f = make_epsilon_schedule("linear", constant=0.1, start=0.3, end=0.0, num_episodes=101)
    assert f(0) == pytest.approx(0.3)
    assert f(100) == pytest.approx(0.0)
    assert f(50) == pytest.approx(0.15)


def test_exponential_hits_endpoints() -> None:
    f = make_epsilon_schedule("exponential", constant=0.1, start=0.4, end=0.01, num_episodes=101)
    assert f(0) == pytest.approx(0.4)
    assert f(100) == pytest.approx(0.01)
    assert 0.01 < f(50) < 0.4


def test_harmonic_hits_endpoints() -> None:
    f = make_epsilon_schedule("harmonic", constant=0.1, start=0.3, end=0.05, num_episodes=101)
    assert f(0) == pytest.approx(0.3)
    assert f(100) == pytest.approx(0.05)


def test_decay_requires_positive_end() -> None:
    for kind in ("exponential", "harmonic"):
        with pytest.raises(ValueError):
            make_epsilon_schedule(kind, constant=0.1, start=0.3, end=0.0, num_episodes=10)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        make_epsilon_schedule("bogus", constant=0.1, start=0.3, end=0.0, num_episodes=10)


def test_train_applies_linear_decay() -> None:
    agent = train(ExperimentConfig(
        num_episodes=300, epsilon_schedule="linear", epsilon_start=0.3, epsilon_end=0.0, seed=1
    ))
    assert agent.q  # learned something
    assert agent.epsilon == pytest.approx(0.0, abs=1e-9)  # ended at epsilon_end
