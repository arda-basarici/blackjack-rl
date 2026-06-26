"""Tests for the experience-replay buffer (training/replay.py). Synthetic transitions only — no
env wiring yet (that arrives with the training loop)."""
from __future__ import annotations

import random

import torch

from blackjack_rl.dqn.replay import Batch, ReplayBuffer, Transition


def _transition(value: float = 0.0, n_actions: int = 3, in_dim: int = 3) -> Transition:
    return Transition(
        state=torch.full((in_dim,), value),
        action=0,
        reward=value,
        next_state=torch.full((in_dim,), value),
        done=False,
        next_legal_mask=torch.ones(n_actions, dtype=torch.bool),
    )


def test_len_and_push() -> None:
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.push(_transition())
    assert len(buf) == 1


def test_capacity_evicts_oldest() -> None:
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(_transition(value=float(i)))
    assert len(buf) == 3
    # the three most recent (2, 3, 4) survive; 0 and 1 were overwritten
    assert sorted(t.reward for t in buf._buf) == [2.0, 3.0, 4.0]


def test_can_sample() -> None:
    buf = ReplayBuffer(capacity=10)
    for _ in range(4):
        buf.push(_transition())
    assert not buf.can_sample(5)
    assert buf.can_sample(4)


def test_sample_batch_shapes_and_dtypes() -> None:
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(_transition(value=float(i)))
    batch = buf.sample(8)
    assert isinstance(batch, Batch)
    assert batch.states.shape == (8, 3)
    assert batch.actions.shape == (8,) and batch.actions.dtype == torch.long
    assert batch.rewards.shape == (8,) and batch.rewards.dtype == torch.float32
    assert batch.next_states.shape == (8, 3)
    assert batch.dones.shape == (8,) and batch.dones.dtype == torch.bool
    assert batch.next_legal_masks.shape == (8, 3) and batch.next_legal_masks.dtype == torch.bool


def test_sample_is_reproducible_under_seed() -> None:
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(_transition(value=float(i)))
    random.seed(0)
    a = buf.sample(5).rewards
    random.seed(0)
    b = buf.sample(5).rewards
    assert torch.equal(a, b)


def test_zero_capacity_rejected() -> None:
    try:
        ReplayBuffer(capacity=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive capacity")
