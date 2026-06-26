"""Tests for the Double-DQN target (training/deep_q.td_target with an ``online`` net).

The key behavioural difference from vanilla: the next action is *selected* by the online net but
*valued* by the target net. Constructed so vanilla and Double give provably different answers."""
from __future__ import annotations

import torch

from blackjack_rl.dqn.deep_q import td_target
from blackjack_rl.dqn.replay import Batch


class _ConstQ:
    """A stand-in net returning a fixed Q-row for every input."""

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


# online prefers index 2 (its max); target's max is at index 0/1, but its value AT index 2 is low.
_ONLINE = _ConstQ([1.0, 0.0, 5.0])   # argmax -> index 2
_TARGET = _ConstQ([10.0, 10.0, 2.0])  # max -> 10, but value at index 2 -> 2


def test_vanilla_uses_target_max() -> None:
    b = _batch(reward=0.0, done=False, mask=[True, True, True])
    assert td_target(_TARGET, b).item() == 10.0  # max over target


def test_double_selects_with_online_values_with_target() -> None:
    b = _batch(reward=0.0, done=False, mask=[True, True, True])
    # online picks index 2; target's value there is 2.0 (not target's max of 10)
    assert td_target(_TARGET, b, online=_ONLINE).item() == 2.0


def test_double_respects_legal_mask_in_selection() -> None:
    b = _batch(reward=0.0, done=False, mask=[True, True, False])  # index 2 illegal
    # online's legal argmax is index 0 (1.0 > 0.0); target value at 0 is 10.0
    assert td_target(_TARGET, b, online=_ONLINE).item() == 10.0


def test_double_terminal_is_just_reward() -> None:
    b = _batch(reward=-1.0, done=True, mask=[False, False, False])
    assert td_target(_TARGET, b, online=_ONLINE).item() == -1.0  # no bootstrap, no NaN
