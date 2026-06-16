"""Tests for blackjack_rl.evaluation.policy_diff."""
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.evaluation.policy_diff import (
    _canonical_state,
    classify,
    diff_policy,
)
from strategies.basic_strategy import BasicStrategy

_MIN, _TOL = 1000, 0.02


def test_classify_agree() -> None:
    assert classify(True, 5, 1.0, _MIN, _TOL) == "agree"


def test_classify_under_visited() -> None:
    assert classify(False, 10, 1.0, _MIN, _TOL) == "under_visited"


def test_classify_near_equal_ev() -> None:
    assert classify(False, 5000, 0.005, _MIN, _TOL) == "near_equal_ev"


def test_classify_genuine_disagreement() -> None:
    assert classify(False, 5000, 0.5, _MIN, _TOL) == "genuine_disagreement"


def test_diff_policy_applies_classification_consistently() -> None:
    agent = TabularAgent()
    # a few cells with controlled tables
    agent.q[((20, False, 10), "stand")] = 1.0
    agent.n[((20, False, 10), "stand")] = 5000
    agent.q[((16, False, 10), "stand")] = 0.5
    agent.n[((16, False, 10), "stand")] = 20
    report = diff_policy(agent, min_visits=_MIN, ev_tol=_TOL)
    assert report.cells
    for cell in report.cells:
        agree = cell.agent_action == cell.basic_action
        expected = classify(agree, cell.visits, abs(cell.agent_q - cell.basic_q), _MIN, _TOL)
        assert cell.category == expected
    assert 0.0 <= report.agreement_unweighted <= 1.0
    assert 0.0 <= report.agreement_weighted <= 1.0
    assert sum(report.category_counts.values()) == len(report.cells)


def test_agreeing_cell_is_agree() -> None:
    agent = TabularAgent()
    key = (20, False, 10)
    basic_action = BasicStrategy().decide(_canonical_state(*key))
    agent.q[(key, basic_action)] = 1.0   # make greedy pick basic's action
    agent.n[(key, basic_action)] = 5000
    report = diff_policy(agent)
    cell = next(c for c in report.cells if (c.player_value, c.is_soft, c.dealer_upcard) == key)
    assert cell.agent_action == basic_action == cell.basic_action
    assert cell.category == "agree"


def test_undervisited_disagreement_is_flagged() -> None:
    agent = TabularAgent()
    key = (16, False, 10)
    basic_action = BasicStrategy().decide(_canonical_state(*key))
    other = "stand" if basic_action != "stand" else "hit"
    agent.q[(key, other)] = 1.0          # greedy picks a non-basic action
    agent.n[(key, other)] = 10           # below min_visits
    report = diff_policy(agent, min_visits=_MIN)
    cell = next(c for c in report.cells if (c.player_value, c.is_soft, c.dealer_upcard) == key)
    assert cell.agent_action == other != cell.basic_action
    assert cell.category == "under_visited"
