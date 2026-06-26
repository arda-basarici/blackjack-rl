"""Capturing a played hand and reconstructing TD transitions (no-split).

Covers ``env.capture_hand`` and ``training.deep_q.hand_to_transitions``. The reconstruction tests
are deterministic (synthetic hands); the capture test only asserts structure, not exact cards.
"""
from __future__ import annotations

import random

import torch

from simulator.game_state import Action, GameState
from strategies.base import Strategy

from blackjack_rl.dqn.agent import encode_features
from blackjack_rl.env import CapturedHand, Step, capture_hand
from blackjack_rl.dqn.deep_q import hand_to_transitions

ACTIONS: tuple[Action, ...] = ("hit", "stand", "double")


def _step(value: int, action: Action, can_double: bool = True, soft: bool = False) -> Step:
    return Step(
        player_value=value,
        player_is_soft=soft,
        dealer_upcard=10,
        can_split=False,
        can_double=can_double,
        action=action,
    )


def test_single_decision_hand_is_one_terminal_transition() -> None:
    ts = hand_to_transitions(CapturedHand(steps=[_step(20, "stand")], reward=1.0), ACTIONS)
    assert len(ts) == 1
    t = ts[0]
    assert t.done is True
    assert t.reward == 1.0
    assert t.action == ACTIONS.index("stand")


def test_chain_rewards_and_done_flags() -> None:
    # hit (2-card, double legal) -> hit (3-card, double illegal) -> stand (terminal); a loss
    hand = CapturedHand(
        steps=[
            _step(12, "hit", can_double=True),
            _step(15, "hit", can_double=False),
            _step(18, "stand", can_double=False),
        ],
        reward=-1.0,
    )
    ts = hand_to_transitions(hand, ACTIONS)
    assert len(ts) == 3
    assert ts[0].reward == 0.0 and ts[0].done is False
    assert ts[1].reward == 0.0 and ts[1].done is False
    assert ts[2].reward == -1.0 and ts[2].done is True  # payout lands only on the terminal step


def test_next_state_and_mask_come_from_following_step() -> None:
    hand = CapturedHand(
        steps=[_step(12, "hit", can_double=True), _step(15, "stand", can_double=False)],
        reward=1.0,
    )
    ts = hand_to_transitions(hand, ACTIONS)
    expected_next = torch.tensor(encode_features(hand.steps[1]), dtype=torch.float32)
    assert torch.equal(ts[0].next_state, expected_next)
    # step 1 has can_double False -> double is masked illegal in the next-legal-mask
    assert ts[0].next_legal_mask.tolist() == [True, True, False]


def test_terminal_next_fields_are_unused_placeholders() -> None:
    t = hand_to_transitions(CapturedHand(steps=[_step(20, "stand")], reward=1.0), ACTIONS)[0]
    assert bool(t.next_legal_mask.any()) is False  # all-False placeholder
    assert torch.equal(t.next_state, torch.zeros_like(t.next_state))


def test_empty_hand_yields_no_transitions() -> None:
    assert hand_to_transitions(CapturedHand(steps=[], reward=1.5), ACTIONS) == []


class _AlwaysStand(Strategy):
    def decide(self, state: GameState) -> Action:
        return "stand"


def test_capture_hand_structure_and_reconstruction() -> None:
    random.seed(0)
    hand = capture_hand(_AlwaysStand())
    assert isinstance(hand.reward, float)
    assert all(isinstance(s, Step) for s in hand.steps)
    ts = hand_to_transitions(hand, ACTIONS)
    assert len(ts) == len(hand.steps)
    if ts:  # a dealt blackjack would have no decisions
        assert ts[-1].done is True
        assert ts[-1].reward == hand.reward
