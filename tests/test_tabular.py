"""Tests for blackjack_rl.agents.tabular — the Q-table policy."""
import random

from simulator.game_state import GameState
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.state import encode_state

_STAGE2 = {"hit", "stand", "double"}


def _gs(pv=16, soft=False, up=10, can_double=False, can_split=False) -> GameState:
    return GameState(
        player_value=pv, player_is_soft=soft, player_card_count=2, dealer_upcard=up,
        can_hit=True, can_stand=True, can_double=can_double, can_split=can_split,
        can_surrender=False,
    )


def test_update_is_a_running_mean():
    a = TabularAgent()
    key = (16, False, 10)
    for g in (1.0, 0.0, -1.0):
        a.update(key, "hit", g)
    assert a.n[(key, "hit")] == 3
    assert abs(a.q[(key, "hit")] - 0.0) < 1e-12  # mean of {1, 0, -1}


def test_constant_step_size_is_recency_weighted():
    a = TabularAgent(step_size=0.5)
    key = (16, False, 10)
    a.update(key, "hit", 1.0)   # 0 + 0.5*(1 - 0)   = 0.5
    a.update(key, "hit", 1.0)   # 0.5 + 0.5*(1-0.5) = 0.75
    assert a.n[(key, "hit")] == 2
    assert abs(a.q[(key, "hit")] - 0.75) < 1e-12   # != sample-average (1.0)


def test_greedy_picks_highest_q():
    a = TabularAgent()
    s = _gs(16, False, 10)              # legal: hit, stand
    key = encode_state(s)
    a.q[(key, "stand")] = 0.5
    a.q[(key, "hit")] = -0.5
    assert a.greedy_action(s) == "stand"


def test_epsilon_zero_decide_equals_greedy():
    a = TabularAgent(epsilon=0.0)
    s = _gs(11, False, 6, can_double=True)
    key = encode_state(s)
    a.q[(key, "double")] = 1.0
    a.q[(key, "hit")] = 0.2
    a.q[(key, "stand")] = -0.3
    assert a.decide(s) == "double" == a.greedy_action(s)


def test_only_legal_stage2_actions_and_never_splits():
    random.seed(0)
    a = TabularAgent(epsilon=1.0)       # always explore
    s = _gs(16, False, 10, can_double=False, can_split=True)  # a pair: split is legal in engine
    chosen = {a.decide(s) for _ in range(200)}
    assert chosen <= _STAGE2
    assert "split" not in chosen and "surrender" not in chosen
    assert a.greedy_action(s) != "split"


def test_double_only_offered_when_legal():
    random.seed(1)
    a = TabularAgent(epsilon=1.0)
    s = _gs(11, False, 6, can_double=False)   # double NOT allowed here
    chosen = {a.decide(s) for _ in range(200)}
    assert "double" not in chosen
    assert chosen <= {"hit", "stand"}
