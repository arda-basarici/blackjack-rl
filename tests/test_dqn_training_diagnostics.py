"""The DQN loop's speed knob and convergence diagnostic: train_every reduces gradient steps, and
each learning-curve checkpoint carries an agreement-with-basic-strategy snapshot."""
from __future__ import annotations

from blackjack_rl.config import DQNConfig
from blackjack_rl.dqn.deep_q import train_dqn


def _final_checkpoint(cfg: DQNConfig) -> dict:
    curve: list[dict] = []
    train_dqn(cfg, progress_every=cfg.num_episodes, on_checkpoint=curve.append)
    assert curve, "expected at least one checkpoint"
    return curve[-1]


def test_train_every_reduces_gradient_steps() -> None:
    base = dict(num_episodes=1200, warmup=100, batch_size=64, seed=0)
    g1 = _final_checkpoint(DQNConfig(train_every=1, **base))["grad_steps"]
    g4 = _final_checkpoint(DQNConfig(train_every=4, **base))["grad_steps"]
    assert g1 > 0 and g4 > 0
    assert g4 < g1 / 3  # ~4x fewer updates (loose bound for warmup edge effects)


def test_checkpoint_includes_agreement_snapshot() -> None:
    cp = _final_checkpoint(DQNConfig(num_episodes=600, warmup=100, batch_size=64, seed=0))
    assert "agreement" in cp
    assert 0.0 <= cp["agreement"] <= 1.0
