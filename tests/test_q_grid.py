"""Test for the full-grid Q snapshot (training/deep_q.full_q_grid) used by per-cell trajectories."""
from __future__ import annotations

import torch

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.dqn.deep_q import full_q_grid


def test_full_q_grid_covers_all_cells() -> None:
    torch.manual_seed(0)
    agent = DQNAgent()
    grid = full_q_grid(agent)
    assert len(grid) == 240  # the full enumerate_cells grid
    for _label, qd in grid.items():
        assert set(qd) == set(agent.actions)
        assert all(isinstance(v, float) for v in qd.values())
