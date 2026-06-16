"""Tests for blackjack_rl.state — the state contract."""
from simulator.game_state import GameState
from blackjack_rl.state import encode_state


def _gs(player_value: int = 16, soft: bool = False, upcard: int = 10) -> GameState:
    return GameState(
        player_value=player_value, player_is_soft=soft, player_card_count=2,
        dealer_upcard=upcard, can_hit=True, can_stand=True, can_double=False,
        can_split=False, can_surrender=False,
    )


def test_encode_returns_expected_tuple():
    assert encode_state(_gs(16, False, 10)) == (16, False, 10)


def test_soft_and_upcard_are_distinguished():
    assert encode_state(_gs(18, True, 9)) != encode_state(_gs(18, False, 9))
    assert encode_state(_gs(18, True, 9)) != encode_state(_gs(18, True, 6))


def test_key_is_hashable():
    d = {encode_state(_gs(20, False, 2)): "stand"}
    assert d[encode_state(_gs(20, False, 2))] == "stand"
