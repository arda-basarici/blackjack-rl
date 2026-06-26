"""Tests for end-of-run representation tooling: the features() accessor, weight save/reload
round-trip, and the cell-embeddings + metadata bundle."""
from __future__ import annotations

import torch

from blackjack_rl.dqn.agent import QNetwork
from blackjack_rl.config import DQNConfig
from blackjack_rl.dqn.experiment import run_dqn
from blackjack_rl.dqn.embedding import cell_embeddings, load_agent
from blackjack_rl.dqn.network_diff import _state_for


def test_features_returns_penultimate_activations() -> None:
    net = QNetwork(in_dim=5, out_dim=3, hidden=(8, 6))
    x = torch.randn(4, 5)
    feats = net.features(x)
    assert feats.shape == (4, 6)          # width of the last hidden layer
    assert net(x).shape == (4, 3)         # full forward still gives one Q per action


def test_save_reload_round_trip(tmp_path) -> None:
    cfg = DQNConfig(
        num_episodes=300, warmup=10, batch_size=8, buffer_capacity=500,
        hidden=(16, 12), encoding="onehot", seed=0,
    )
    res = run_dqn(cfg, eval_hands=200, runs_dir=tmp_path, progress_every=None, save=True)
    assert res.run_dir is not None and (res.run_dir / "model.pt").exists()
    reloaded = load_agent(res.run_dir)
    st = _state_for(16, False, 10)
    assert torch.allclose(res.agent.q_values(st), reloaded.q_values(st), atol=1e-6)


def test_cell_embeddings_shapes_and_metadata(tmp_path) -> None:
    cfg = DQNConfig(
        num_episodes=300, warmup=10, batch_size=8, buffer_capacity=500,
        hidden=(16, 12), encoding="onehot", seed=0,
    )
    res = run_dqn(cfg, eval_hands=200, runs_dir=tmp_path, progress_every=None, save=True)
    ce = cell_embeddings(load_agent(res.run_dir))
    assert len(ce.embeddings) == 240
    assert len(ce.embeddings[0]) == 12             # penultimate width = hidden[-1]
    assert len(ce.cells) == 240
    keys = set(ce.cells[0])
    assert {"player_value", "is_soft", "dealer_upcard", "action", "category", "q_margin"} <= keys
