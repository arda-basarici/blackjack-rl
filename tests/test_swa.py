"""Test for Stochastic Weight Averaging (DQNConfig.swa): training runs end-to-end and the returned
agent's weights are the back-half average (i.e. SWA actually changed the final net vs no-SWA)."""
from __future__ import annotations

import torch

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.dqn.deep_q import train_dqn


def _final_weights(swa: bool) -> torch.Tensor:
    cfg = DQNConfig(num_episodes=1500, warmup=200, batch_size=64, seed=0, swa=swa)
    agent = train_dqn(cfg, progress_every=100)  # frequent checkpoints so SWA accumulates
    return torch.cat([p.detach().flatten() for p in agent.q_net.parameters()])


def test_swa_runs_and_changes_the_final_net() -> None:
    w_plain = _final_weights(swa=False)
    w_swa = _final_weights(swa=True)
    assert w_plain.shape == w_swa.shape
    # SWA evaluates averaged weights, so the final net must differ from the plain final snapshot
    assert not torch.allclose(w_plain, w_swa)


def test_swa_agent_still_plays_legally() -> None:
    from simulator.game_state import GameState
    cfg = DQNConfig(num_episodes=1500, warmup=200, batch_size=64, seed=0, swa=True)
    agent = train_dqn(cfg, progress_every=100)
    s = GameState(player_value=16, player_is_soft=False, player_card_count=2, dealer_upcard=10,
                  can_hit=True, can_stand=True, can_double=True, can_split=False, can_surrender=False)
    assert isinstance(agent, DQNAgent)
    assert agent.greedy_action(s) in ("hit", "stand", "double")
