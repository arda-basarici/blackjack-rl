"""Tests for the config-driven feature encoding (agents/dqn.py): scalar vs one-hot share the same
information but differ in shape and prior. Scalar is the default, so existing behaviour is intact.
"""
from __future__ import annotations

import torch

from simulator.game_state import Action, GameState

from blackjack_rl.agents.dqn import DQNAgent, encode_features, feature_dim


def _state(value: int = 16, soft: bool = False, upcard: int = 10) -> GameState:
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


def test_scalar_is_default_and_unchanged() -> None:
    assert feature_dim("scalar") == 3
    assert encode_features(_state()) == encode_features(_state(), encoding="scalar")
    assert len(encode_features(_state())) == 3


def test_onehot_dim_and_shape() -> None:
    # 18 total bins (4..21) + 1 soft flag + 10 upcard bins (2..11)
    assert feature_dim("onehot") == 29
    f = encode_features(_state(value=16, upcard=10), encoding="onehot")
    assert len(f) == 29
    assert sum(f) == 2.0 + 0.0  # exactly one total bin + one upcard bin hot, soft flag 0


def test_onehot_puts_the_one_in_the_right_slots() -> None:
    f = encode_features(_state(value=16, soft=True, upcard=10), encoding="onehot")
    assert f[16 - 4] == 1.0          # total bin for 16
    assert f[18] == 1.0              # soft flag (index 18 = after the 18 total bins)
    assert f[19 + (10 - 2)] == 1.0   # upcard bin for 10 (block starts at index 19)


def test_onehot_neighbours_are_orthogonal() -> None:
    # the whole point: 16 and 17 share no active input (a category, not a ramp)
    f16 = torch.tensor(encode_features(_state(value=16), encoding="onehot"))
    f17 = torch.tensor(encode_features(_state(value=17), encoding="onehot"))
    assert float((f16 * f17)[:18].sum()) == 0.0  # disjoint total bins


def test_agent_input_width_matches_encoding() -> None:
    torch.manual_seed(0)
    assert DQNAgent(encoding="scalar").q_values(_state()).shape == (3,)  # 3 actions out, not in
    # the network's first layer in-features must match the encoding width
    assert DQNAgent(encoding="scalar").q_net.net[0].in_features == 3
    assert DQNAgent(encoding="onehot").q_net.net[0].in_features == 29
