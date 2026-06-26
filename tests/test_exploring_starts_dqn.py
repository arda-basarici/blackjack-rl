"""Tests for exploring-starts DQN (training/exploring_starts_dqn.py): the forced-start capture
begins from the chosen (state, action), and the training loop runs end-to-end with finite losses."""
from __future__ import annotations

import math
import random

import torch

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.env import problem_a_config
from blackjack_rl.dqn.exploring_starts_dqn import es_capture, train_dqn_es


def _state(value: int = 16, soft: bool = False, upcard: int = 10):
    from simulator.game_state import GameState
    return GameState(
        player_value=value, player_is_soft=soft, player_card_count=2, dealer_upcard=upcard,
        can_hit=True, can_stand=True, can_double=True, can_split=False, can_surrender=False,
    )


def _capture_with_retry(agent, spec, action):
    """Retry across seeds to skip the occasional discarded dealer-blackjack hand."""
    for seed in range(20):
        random.seed(seed)
        hand = es_capture(agent, spec, action, problem_a_config())
        if hand is not None:
            return hand
    raise AssertionError("no non-discarded hand in 20 seeds")


def test_es_capture_starts_from_forced_state_and_action() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(epsilon=0.0)
    hand = _capture_with_retry(agent, (16, False, False, 10), "hit")  # hard 16 v 10, forced hit
    first = hand.steps[0]
    assert first.player_value == 16 and first.player_is_soft is False
    assert first.action == "hit"  # the forced first action


def test_es_capture_forced_double_is_a_single_terminal_step() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(epsilon=0.0)
    hand = _capture_with_retry(agent, (20, True, False, 8), "double")  # soft 20 v 8, forced double
    assert hand.steps[0].action == "double"
    assert hand.steps[0].player_value == 20 and hand.steps[0].player_is_soft
    assert len(hand.steps) == 1            # doubling ends the hand
    assert hand.reward in (-2.0, 0.0, 2.0)  # doubled payout


def test_train_dqn_es_runs_with_finite_losses() -> None:
    cfg = DQNConfig(num_episodes=1500, warmup=200, batch_size=64, target_sync_every=100, seed=0)
    losses: list = []
    agent = train_dqn_es(cfg, progress_every=500, on_checkpoint=lambda d: losses.append(d["recent_loss"]))
    assert isinstance(agent, DQNAgent)
    assert agent.greedy_action(_state()) in ("hit", "stand", "double")
    assert losses and all(x is not None and math.isfinite(x) for x in losses)
