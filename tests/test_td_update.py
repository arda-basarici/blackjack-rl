"""Tests for the deep-Q TD update (training/deep_q.py): the target arithmetic (masking, terminal
handling, discount) checked exactly with a stand-in target net, and that an optimizer step
actually reduces the loss on a fixed batch."""
from __future__ import annotations

import torch

from blackjack_rl.agents.dqn import QNetwork
from blackjack_rl.training.deep_q import td_target, td_update
from blackjack_rl.training.replay import Batch


class _ConstQ:
    """A stand-in target net returning a fixed Q-row for every input — lets us check the target
    arithmetic exactly without depending on learned weights."""

    def __init__(self, q_row: list[float]) -> None:
        self._q = torch.tensor(q_row, dtype=torch.float32).unsqueeze(0)  # [1, n]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._q.expand(x.shape[0], -1)


def _batch(reward: float, done: bool, mask: list[bool], in_dim: int = 3) -> Batch:
    return Batch(
        states=torch.zeros(1, in_dim),
        actions=torch.zeros(1, dtype=torch.long),
        rewards=torch.tensor([reward], dtype=torch.float32),
        next_states=torch.zeros(1, in_dim),
        dones=torch.tensor([done], dtype=torch.bool),
        next_legal_masks=torch.tensor([mask], dtype=torch.bool),
    )


def test_terminal_target_is_just_reward() -> None:
    # no bootstrap on terminal — even with an all-illegal next mask (no NaN from -inf * 0)
    y = td_target(_ConstQ([1.0, 5.0, 9.0]), _batch(reward=-1.0, done=True, mask=[False, False, False]))
    assert y.item() == -1.0


def test_nonterminal_target_bootstraps_max_legal() -> None:
    y = td_target(_ConstQ([1.0, 5.0, 9.0]), _batch(reward=0.0, done=False, mask=[True, True, True]))
    assert y.item() == 9.0  # max(1, 5, 9), reward 0, gamma 1


def test_masking_excludes_illegal_next_action() -> None:
    # index 2 (value 9, the would-be best) is illegal -> max over legal {1, 5} = 5
    y = td_target(_ConstQ([1.0, 5.0, 9.0]), _batch(reward=0.0, done=False, mask=[True, True, False]))
    assert y.item() == 5.0


def test_gamma_scales_the_bootstrap() -> None:
    y = td_target(
        _ConstQ([0.0, 0.0, 10.0]), _batch(reward=1.0, done=False, mask=[True, True, True]), gamma=0.5
    )
    assert y.item() == 1.0 + 0.5 * 10.0


def test_update_reduces_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    online = QNetwork(3, 3)
    target = QNetwork(3, 3)  # fixed (never synced here) -> fixed targets to regress toward
    batch = Batch(
        states=torch.randn(16, 3),
        actions=torch.randint(0, 3, (16,)),
        rewards=torch.randn(16),
        next_states=torch.randn(16, 3),
        dones=torch.zeros(16, dtype=torch.bool),
        next_legal_masks=torch.ones(16, 3, dtype=torch.bool),
    )
    opt = torch.optim.Adam(online.parameters(), lr=1e-2)
    first = td_update(online, target, batch, opt)
    last = first
    for _ in range(50):
        last = td_update(online, target, batch, opt)
    assert last < first  # overfitting one batch drives the TD loss down
