"""Tests for the learning-rate schedule: the generic schedule reused for the lr, config validation,
and that train_dqn actually anneals the optimizer step over training (CONCEPTS §26)."""
from __future__ import annotations

from blackjack_rl.config import DQNConfig
from blackjack_rl.schedules import make_epsilon_schedule, make_schedule
from blackjack_rl.training.deep_q import train_dqn


def test_make_schedule_is_generic_and_aliased() -> None:
    # the epsilon name is kept as a back-compat alias of the now-generic builder
    assert make_epsilon_schedule is make_schedule


def test_harmonic_lr_decays_to_end() -> None:
    f = make_schedule("harmonic", constant=1e-3, start=1e-3, end=1e-5, num_episodes=101)
    assert f(0) == 1e-3
    assert f(0) > f(50) > f(100)            # monotone decay
    assert abs(f(100) - 1e-5) < 1e-12       # reaches end exactly at the final episode


def test_lr_schedule_config_validation() -> None:
    DQNConfig(num_episodes=10, lr_schedule="harmonic", lr_end=1e-5)  # ok
    DQNConfig(num_episodes=10, lr_schedule="linear", lr_end=0.0)     # linear may reach exactly 0
    for bad in ("bogus", "Harmonic"):
        try:
            DQNConfig(num_episodes=10, lr_schedule=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for lr_schedule={bad!r}")
    for kind in ("harmonic", "exponential"):  # these need a positive end
        try:
            DQNConfig(num_episodes=10, lr_schedule=kind, lr_end=0.0)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kind} with lr_end=0")


def test_train_dqn_anneals_lr_per_schedule() -> None:
    curve: list[dict] = []
    cfg = DQNConfig(
        num_episodes=300, lr=1e-3, lr_schedule="harmonic", lr_end=1e-5,
        warmup=10, batch_size=8, buffer_capacity=1_000, seed=0,
    )
    train_dqn(cfg, progress_every=100, on_checkpoint=curve.append)
    lrs = [cp["lr"] for cp in curve]
    assert len(lrs) == 3
    assert lrs[0] > lrs[1] > lrs[2]              # the step shrinks over training
    assert abs(lrs[-1] - 1e-5) < 1e-7           # settling toward lr_end


def test_train_dqn_constant_lr_is_unchanged() -> None:
    """Default (constant) schedule keeps the lr fixed — prior runs reproduce exactly."""
    curve: list[dict] = []
    cfg = DQNConfig(
        num_episodes=200, lr=7e-4, warmup=10, batch_size=8, buffer_capacity=1_000, seed=0,
    )
    train_dqn(cfg, progress_every=100, on_checkpoint=curve.append)
    assert all(cp["lr"] == round(7e-4, 8) for cp in curve)
