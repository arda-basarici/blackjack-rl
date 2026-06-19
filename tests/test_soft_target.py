"""Tests for the soft/Polyak target update (training/deep_q.soft_update) and the config knob."""
from __future__ import annotations

import torch

from blackjack_rl.agents.dqn import QNetwork
from blackjack_rl.config import DQNConfig
from blackjack_rl.training.deep_q import make_target, soft_update


def test_soft_update_moves_target_toward_online_by_tau() -> None:
    torch.manual_seed(0)
    online = QNetwork(3, 3)
    target = make_target(online)            # identical copy, frozen
    # diverge the online net
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    before = [tp.detach().clone() for tp in target.parameters()]
    soft_update(target, online, tau=0.1)
    for b, tp, op in zip(before, target.parameters(), online.parameters()):
        # new target = 0.9*old + 0.1*online
        expected = 0.9 * b + 0.1 * op.detach()
        assert torch.allclose(tp, expected, atol=1e-6)
    # target stays frozen (soft_update must not re-enable grad)
    assert all(not p.requires_grad for p in target.parameters())


def test_target_tau_validation() -> None:
    DQNConfig(num_episodes=10, target_tau=0.005)  # ok
    for bad in (-0.1, 1.0, 1.5):
        try:
            DQNConfig(num_episodes=10, target_tau=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for target_tau={bad}")
