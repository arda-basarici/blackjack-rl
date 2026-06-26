"""Tests for the learning-rate schedule: the generic schedule reused for the lr, config validation,
and that train_dqn actually anneals the optimizer step over training (CONCEPTS §26)."""
from __future__ import annotations

from blackjack_rl.config import DQNConfig
from blackjack_rl.schedules import make_epsilon_schedule, make_schedule
from blackjack_rl.dqn.deep_q import train_dqn


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


def test_hold_then_decay_keeps_lr_flat_then_anneals() -> None:
    """flat-then-decay: lr stays at ``start`` until ``hold``, then decays start->end over the rest
    (the schedule that pairs with --double-after to keep the step high through the hit/stand phase)."""
    f = make_schedule("harmonic", constant=1e-3, start=1e-3, end=1e-5, num_episodes=201, hold=100)
    assert f(0) == 1e-3
    assert f(50) == 1e-3 == f(99)               # still flat in the hold phase
    assert f(100) == 1e-3                        # decay begins AT hold (shape(0) == start)
    assert f(100) > f(150) > f(200)             # then anneals
    assert abs(f(200) - 1e-5) < 1e-12           # reaches end exactly at the final episode
    # without the hold, the same schedule would already be decaying by episode 50
    g = make_schedule("harmonic", constant=1e-3, start=1e-3, end=1e-5, num_episodes=201)
    assert g(50) < 1e-3


def test_hold_default_is_decay_from_start() -> None:
    """hold=0 (default) must reproduce the original decay-from-episode-0 behavior exactly."""
    a = make_schedule("linear", constant=1e-3, start=1e-3, end=0.0, num_episodes=101)
    b = make_schedule("linear", constant=1e-3, start=1e-3, end=0.0, num_episodes=101, hold=0)
    assert [a(i) for i in (0, 25, 50, 100)] == [b(i) for i in (0, 25, 50, 100)]


def test_lr_hold_until_config_validation() -> None:
    DQNConfig(num_episodes=100, lr_schedule="harmonic", lr_end=1e-5, lr_hold_until=50)  # ok
    DQNConfig(num_episodes=100)                                                          # default 0 ok
    for bad in (-1, 100, 200):  # must be in [0, num_episodes)
        try:
            DQNConfig(num_episodes=100, lr_hold_until=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for lr_hold_until={bad}")


def test_train_dqn_holds_then_anneals_lr() -> None:
    curve: list[dict] = []
    cfg = DQNConfig(
        num_episodes=400, lr=1e-3, lr_schedule="harmonic", lr_end=1e-5, lr_hold_until=200,
        warmup=10, batch_size=8, buffer_capacity=1_000, seed=0,
    )
    train_dqn(cfg, progress_every=100, on_checkpoint=curve.append)
    lrs = [cp["lr"] for cp in curve]            # checkpoints at episodes 100, 200, 300, 400
    assert len(lrs) == 4
    assert lrs[0] == round(1e-3, 8) == lrs[1]   # held flat through the first half
    assert lrs[2] < lrs[1]                       # decaying after the hold
    assert abs(lrs[-1] - 1e-5) < 1e-7           # settling toward lr_end
