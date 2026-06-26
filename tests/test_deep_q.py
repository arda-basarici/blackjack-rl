"""Tests for the target-network helpers (training/deep_q.py). The TD loop is the next unit; here
we only check that the target is a genuine *frozen* copy and that sync behaves."""
from __future__ import annotations

import torch

from blackjack_rl.dqn.agent import QNetwork
from blackjack_rl.dqn.deep_q import make_target, sync_target


def _params_equal(a: QNetwork, b: QNetwork) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def _one_sgd_step(net: QNetwork) -> None:
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = net(torch.ones(4, 3)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_make_target_is_a_frozen_copy() -> None:
    torch.manual_seed(0)
    online = QNetwork(3, 3)
    target = make_target(online)
    assert _params_equal(online, target)  # same weights at birth
    assert all(not p.requires_grad for p in target.parameters())  # and frozen


def test_training_online_does_not_move_target() -> None:
    torch.manual_seed(0)
    online = QNetwork(3, 3)
    target = make_target(online)
    _one_sgd_step(online)
    assert not _params_equal(online, target)  # online moved; target stayed put


def test_sync_copies_online_into_target() -> None:
    torch.manual_seed(0)
    online = QNetwork(3, 3)
    target = make_target(online)
    _one_sgd_step(online)
    assert not _params_equal(online, target)
    sync_target(target, online)
    assert _params_equal(online, target)  # back in sync
    # target stays frozen after a sync (load_state_dict doesn't re-enable grads)
    assert all(not p.requires_grad for p in target.parameters())
