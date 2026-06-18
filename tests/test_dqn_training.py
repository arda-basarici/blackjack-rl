"""Smoke test for the DQN training loop (training/deep_q.train_dqn).

Trains briefly on Problem A and checks the loop runs end-to-end with finite (non-NaN) losses —
which exercises the -inf*0 guard in the live loop, where terminal transitions really occur — and
returns an agent with a legal greedy policy. It does NOT assert the agent learned basic strategy;
that's the real run (task 4). Kept small so it stays fast.
"""
from __future__ import annotations

import math

import torch

from simulator.game_state import Action, GameState

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.training.deep_q import train_dqn


def _decision_state(value: int = 16, soft: bool = False, upcard: int = 10) -> GameState:
    return GameState(
        player_value=value,
        player_is_soft=soft,
        player_card_count=2,
        dealer_upcard=upcard,
        can_hit=True,
        can_stand=True,
        can_double=True,
        can_split=False,
        can_surrender=False,
    )


def test_train_dqn_runs_with_finite_losses_and_legal_policy() -> None:
    cfg = DQNConfig(num_episodes=1500, warmup=200, batch_size=64, target_sync_every=100, seed=0)
    losses: list[float | None] = []
    agent = train_dqn(cfg, progress_every=500, on_checkpoint=lambda d: losses.append(d["recent_loss"]))

    assert isinstance(agent, DQNAgent)
    assert agent.greedy_action(_decision_state()) in ("hit", "stand", "double")
    assert losses, "expected at least one learning-curve checkpoint"
    assert all(x is not None and math.isfinite(x) for x in losses)  # the NaN guard holds live


def test_train_dqn_is_reproducible_under_seed() -> None:
    cfg = DQNConfig(num_episodes=400, warmup=100, batch_size=64, seed=7)
    a1 = train_dqn(cfg)
    a2 = train_dqn(cfg)
    assert torch.equal(a1.q_values(_decision_state()), a2.q_values(_decision_state()))
