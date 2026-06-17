"""Tests for blackjack_rl.state — the state contract."""
from dataclasses import dataclass

from simulator.game_state import GameState
from blackjack_rl.state import encode_state


def _gs(player_value: int = 16, soft: bool = False, upcard: int = 10,
        can_split: bool = False) -> GameState:
    return GameState(
        player_value=player_value, player_is_soft=soft, player_card_count=2,
        dealer_upcard=upcard, can_hit=True, can_stand=True, can_double=False,
        can_split=can_split, can_surrender=False,
    )


def test_encode_returns_expected_tuple() -> None:
    assert encode_state(_gs(16, False, 10)) == (16, False, 10)


def test_soft_and_upcard_are_distinguished() -> None:
    assert encode_state(_gs(18, True, 9)) != encode_state(_gs(18, False, 9))
    assert encode_state(_gs(18, True, 9)) != encode_state(_gs(18, True, 6))


def test_key_is_hashable() -> None:
    d = {encode_state(_gs(20, False, 2)): "stand"}
    assert d[encode_state(_gs(20, False, 2))] == "stand"


def test_no_split_mode_ignores_can_split() -> None:
    # default: a pair and a non-pair of the same value collapse to one key (backward compatible)
    assert encode_state(_gs(16, can_split=True)) == encode_state(_gs(16, can_split=False)) == (16, False, 10)


def test_split_mode_appends_can_split() -> None:
    assert encode_state(_gs(16, can_split=True), with_splits=True) == (16, False, 10, True)
    assert encode_state(_gs(16, can_split=False), with_splits=True) == (16, False, 10, False)


def test_split_mode_distinguishes_pair_from_nonpair() -> None:
    pair = encode_state(_gs(16, can_split=True), with_splits=True)       # 8,8
    nonpair = encode_state(_gs(16, can_split=False), with_splits=True)   # e.g. 10,6
    assert pair != nonpair


def test_encode_state_is_duck_typed() -> None:
    # works on anything with the four fields — e.g. the engine's DecisionRecord (env uses this)
    @dataclass
    class _RecordLike:
        player_value: int
        player_is_soft: bool
        dealer_upcard: int
        can_split: bool

    r = _RecordLike(11, False, 6, True)
    assert encode_state(r, with_splits=True) == (11, False, 6, True)
    assert encode_state(r) == (11, False, 6)
