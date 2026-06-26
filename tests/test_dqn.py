"""Tests for the DQN agent skeleton (agents/dqn.py).

These check *mechanics* only — feature shape/range, output shape, legal-action masking, and
determinism — not playing skill: with random initial weights the agent plays arbitrarily by
design (training comes later). Same honest-boundary discipline as A5: assert what is
deterministically true now, not what can't yet be measured.
"""
from __future__ import annotations

import random

import torch

from simulator.game_state import GameState

from blackjack_rl.dqn.agent import DQNAgent, QNetwork, encode_features


def _state(**kw) -> GameState:
    """A hard-16-vs-10 decision state by default; override fields via kwargs."""
    base = dict(
        player_value=16,
        player_is_soft=False,
        player_card_count=2,
        dealer_upcard=10,
        can_hit=True,
        can_stand=True,
        can_double=True,
        can_split=False,
        can_surrender=False,
    )
    base.update(kw)
    return GameState(**base)


def test_encode_features_length_and_range() -> None:
    f = encode_features(_state())
    assert len(f) == 3
    assert all(0.0 <= v <= 1.0 for v in f)
    # boundaries land on [0, 1] exactly
    assert encode_features(_state(player_value=4, dealer_upcard=2)) == [0.0, 0.0, 0.0]
    assert encode_features(_state(player_value=21, dealer_upcard=11)) == [1.0, 0.0, 1.0]


def test_encode_features_appends_can_split() -> None:
    f = encode_features(_state(can_split=True), with_splits=True)
    assert len(f) == 4
    assert f[3] == 1.0
    # without the flag the split feature is absent (no-split mode is 3 features)
    assert len(encode_features(_state(can_split=True), with_splits=False)) == 3


def test_network_output_shape() -> None:
    net = QNetwork(3, 3)
    out = net(torch.tensor(encode_features(_state()), dtype=torch.float32))
    assert out.shape == (3,)


def test_qvalues_length_matches_action_space() -> None:
    torch.manual_seed(0)
    assert DQNAgent().q_values(_state()).shape == (3,)
    torch.manual_seed(0)
    assert DQNAgent(with_splits=True).q_values(_state(can_split=True)).shape == (4,)


def test_greedy_action_is_always_legal() -> None:
    torch.manual_seed(0)
    agent = DQNAgent()
    s = _state(can_double=False)  # only hit/stand legal
    assert agent.greedy_action(s) in ("hit", "stand")


def test_masking_respects_legality_under_a_forced_preference() -> None:
    """Force the network to most-prefer 'double', then confirm masking still excludes it when
    illegal — the strong version of the masking guarantee."""
    torch.manual_seed(0)
    agent = DQNAgent()
    last = agent.q_net.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))  # index 2 == 'double' is highest
    assert agent._actions[2] == "double"
    assert agent.greedy_action(_state(can_double=True)) == "double"
    assert agent.greedy_action(_state(can_double=False)) in ("hit", "stand")


def test_same_seed_gives_identical_weights_and_choice() -> None:
    torch.manual_seed(42)
    a1 = DQNAgent()
    torch.manual_seed(42)
    a2 = DQNAgent()
    s = _state()
    assert torch.equal(a1.q_values(s), a2.q_values(s))
    assert a1.greedy_action(s) == a2.greedy_action(s)


def test_epsilon_zero_decide_equals_greedy() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(epsilon=0.0)
    s = _state()
    random.seed(0)
    assert agent.decide(s) == agent.greedy_action(s)
