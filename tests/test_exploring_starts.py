"""Tests for the exploring-starts building blocks: card construction, state->cards, the prepared
deck, the forced-action wrapper, and an end-to-end check that a constructed start state is exactly
what the unmodified engine reports at the first decision."""
import pytest

from simulator.hand_simulator import HandSimulator
from simulator.game_state import Action, GameState
from strategies.base import Strategy

from blackjack_rl.env import problem_a_config
from blackjack_rl.training.exploring_starts import (
    ForcedFirstAction,
    PreparedDeck,
    card_of_value,
    player_cards_for,
    start_cards_for,
)


# --- card_of_value -------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_card_of_value_roundtrips(value: int) -> None:
    assert card_of_value(value).value() == value

def test_card_of_value_eleven_is_ace() -> None:
    assert card_of_value(11).is_ace()

@pytest.mark.parametrize("bad", [0, 1, 12, 21])
def test_card_of_value_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        card_of_value(bad)


# --- player_cards_for ----------------------------------------------------------------------------

def _value_soft(cards: list) -> tuple[int, bool]:
    total = sum(c.value() for c in cards)
    aces = sum(1 for c in cards if c.is_ace())
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0

def test_hard_total_is_two_distinct_non_ace_cards() -> None:
    cards = player_cards_for(16, is_soft=False, can_split=False)
    assert cards is not None
    assert _value_soft(cards) == (16, False)
    assert cards[0].value() != cards[1].value()      # not a (splittable) pair
    assert not any(c.is_ace() for c in cards)

def test_soft_total_is_ace_plus_one() -> None:
    cards = player_cards_for(18, is_soft=True, can_split=False)
    assert cards is not None
    assert _value_soft(cards) == (18, True)
    assert any(c.is_ace() for c in cards)

def test_pair_is_equal_value() -> None:
    cards = player_cards_for(16, is_soft=False, can_split=True)   # 8,8
    assert cards is not None
    assert cards[0].value() == cards[1].value() == 8

def test_ace_pair_is_soft_twelve() -> None:
    cards = player_cards_for(12, is_soft=True, can_split=True)
    assert cards is not None
    assert all(c.is_ace() for c in cards)
    assert _value_soft(cards) == (12, True)

@pytest.mark.parametrize("pv", [4, 20])
def test_hard_extremes_not_two_card_constructible(pv: int) -> None:
    assert player_cards_for(pv, is_soft=False, can_split=False) is None

def test_soft_twelve_nonpair_unconstructible() -> None:
    # soft 12 can only be A,A (a pair), so as a non-pair it has no realisation
    assert player_cards_for(12, is_soft=True, can_split=False) is None


# --- start_cards_for -----------------------------------------------------------------------------

def test_start_cards_orders_upcard_in_the_middle() -> None:
    cards = start_cards_for(16, is_soft=False, can_split=False, dealer_upcard=10)
    assert cards is not None and len(cards) == 3
    assert cards[1].value() == 10                      # deal order: player1, UPCARD, player2

def test_start_cards_none_when_unconstructible() -> None:
    assert start_cards_for(20, is_soft=False, can_split=False, dealer_upcard=6) is None


# --- PreparedDeck --------------------------------------------------------------------------------

def test_prepared_deck_serves_forced_prefix_then_shoe() -> None:
    forced = [card_of_value(10), card_of_value(6), card_of_value(7)]
    deck = PreparedDeck(forced)
    dealt = [deck.deal().value() for _ in range(3)]
    assert dealt == [10, 6, 7]                          # forced, in order
    assert deck.deal() is not None                      # falls through to the shoe

def test_prepared_deck_hole_is_random() -> None:
    # 4th card (the hole) comes from the shuffled shoe -> not fixed across decks
    forced = [card_of_value(10), card_of_value(6), card_of_value(7)]
    holes = set()
    for _ in range(50):
        d = PreparedDeck(forced)
        for _ in range(3):
            d.deal()
        holes.add(d.deal().value())
    assert len(holes) > 1


# --- ForcedFirstAction ---------------------------------------------------------------------------

class _Always(Strategy):
    def __init__(self, action: Action) -> None:
        self._action = action
    def decide(self, state: GameState) -> Action:
        return self._action
    def name(self) -> str:
        return "always"

def test_forced_first_action_then_delegates() -> None:
    wrapped = ForcedFirstAction(_Always("hit"), "double")
    dummy = object()                                    # decide ignores state for _Always
    assert wrapped.decide(dummy) == "double"            # first: forced
    assert wrapped.decide(dummy) == "hit"               # then: delegate
    assert wrapped.decide(dummy) == "hit"


# --- end to end: the engine reports the state we constructed --------------------------------------

class _Recorder(Strategy):
    def __init__(self) -> None:
        self.first: GameState | None = None
    def decide(self, state: GameState) -> Action:
        if self.first is None:
            self.first = state
        return "stand"
    def name(self) -> str:
        return "recorder"

def _first_state(pv: int, soft: bool, split: bool, up: int, hole: int = 7) -> GameState:
    forced = start_cards_for(pv, soft, split, up)
    assert forced is not None
    deck = PreparedDeck(forced + [card_of_value(hole)])  # benign hole -> no dealer blackjack
    rec = _Recorder()
    HandSimulator(problem_a_config(), deck, rec).play_hand("es", 0.0, 1.0, 0)
    assert rec.first is not None
    return rec.first

@pytest.mark.parametrize("pv,soft,split,up", [
    (16, False, False, 10),
    (18, True, False, 6),
    (16, False, True, 10),    # pair of 8s
    (12, True, True, 11),     # pair of aces vs ace upcard
])
def test_constructed_start_state_matches_engine(pv: int, soft: bool, split: bool, up: int) -> None:
    s = _first_state(pv, soft, split, up)
    assert (s.player_value, s.player_is_soft, s.can_split, s.dealer_upcard) == (pv, soft, split, up)
