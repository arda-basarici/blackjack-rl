"""Tests for the training instrumentation: per-(state, action) experience counts (answers 'which
action was tested more') and probe-cell Q snapshots (answers 'how did Q evolve')."""
from __future__ import annotations

import torch

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.training.deep_q import PROBE_CELLS, probe_q_values, train_dqn


def test_probe_q_values_shape() -> None:
    torch.manual_seed(0)
    agent = DQNAgent()
    pq = probe_q_values(agent)
    assert len(pq) == len(PROBE_CELLS)
    for _label, qd in pq.items():
        assert set(qd) == set(agent.actions)
        assert all(isinstance(v, float) for v in qd.values())


def test_training_populates_sample_counts() -> None:
    agent = train_dqn(DQNConfig(num_episodes=800, warmup=100, batch_size=64, seed=0))
    assert isinstance(agent.sample_counts, dict) and len(agent.sample_counts) > 0
    k = next(iter(agent.sample_counts))
    assert len(k) == 4  # (value, soft, upcard, action)
    assert isinstance(agent.sample_counts[k], int) and agent.sample_counts[k] > 0


def test_checkpoint_carries_probe_q() -> None:
    cps: list[dict] = []
    train_dqn(DQNConfig(num_episodes=600, warmup=100, batch_size=64, seed=0),
              progress_every=600, on_checkpoint=cps.append)
    assert cps and "probe_q" in cps[-1]
    assert len(cps[-1]["probe_q"]) == len(PROBE_CELLS)
