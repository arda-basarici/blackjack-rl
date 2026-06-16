"""Tests for blackjack_rl.evaluation.metrics — mechanics, not statistical precision.

The precise 'agent matches basic strategy' result is the project's headline finding and is
produced by a full evaluation run, not asserted here. These tests check the harness maths.
"""
from simulator.game_state import Action, GameState
from strategies.base import Strategy
from blackjack_rl.evaluation.metrics import EdgeResult, GreedyPolicy, evaluate_policy


class _AlwaysStand(Strategy):
    def decide(self, state: GameState) -> Action:
        return "stand"

    def name(self) -> str:
        return "AlwaysStand"


class _FakeAgent:
    def greedy_action(self, state: GameState) -> Action:
        return "stand"

    def name(self) -> str:
        return "Fake"


def _gs() -> GameState:
    return GameState(
        player_value=16, player_is_soft=False, player_card_count=2, dealer_upcard=10,
        can_hit=True, can_stand=True, can_double=False, can_split=False, can_surrender=False,
    )


def test_greedy_policy_delegates_to_agent() -> None:
    gp = GreedyPolicy(_FakeAgent())
    assert gp.decide(_gs()) == "stand"
    assert "greedy" in gp.name()


def test_edge_is_negative_mean_reward() -> None:
    result = evaluate_policy(_AlwaysStand(), n_hands=2000, seed=1)
    assert isinstance(result, EdgeResult)
    assert result.n == 2000
    assert result.edge == -result.mean_reward
    assert result.std_error >= 0.0


def test_always_stand_has_positive_edge() -> None:
    # Standing on every hand is clearly losing; the true edge is several %, far above the
    # standard error at 20k hands, so this one-sided check is robust (not flaky).
    result = evaluate_policy(_AlwaysStand(), n_hands=20_000, seed=2)
    assert result.edge > 0.02
