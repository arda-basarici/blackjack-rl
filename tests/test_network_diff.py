"""Tests for the network fidelity diff (evaluation/network_diff.py): the cell enumeration, the
materialized tabular-shaped view, and that diff_network reuses the auditor with no under_visited
category. The policy is forced (not trained) so the comparison is deterministic."""
from __future__ import annotations

import torch

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.evaluation.network_diff import diff_network, enumerate_cells, materialize


def _force_action(agent: DQNAgent, action: str) -> None:
    """Make the network always most-prefer ``action`` (regardless of input) by overwriting the
    last layer: zero weights, bias high on that action's index."""
    idx = agent.actions.index(action)
    last = agent.q_net.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        bias = torch.full((len(agent.actions),), -1.0)
        bias[idx] = 1.0
        last.bias.copy_(bias)


def test_enumerate_cells_count() -> None:
    cells = enumerate_cells()
    # 16 hard totals + 8 soft totals = 24 per upcard, x 10 dealer upcards
    assert len(cells) == 240
    assert len(set(cells)) == 240  # all distinct


def test_materialize_fills_q_for_every_action_each_cell() -> None:
    torch.manual_seed(0)
    table = materialize(DQNAgent())
    assert len(table.n) == 240  # one n entry per cell
    # q has an entry for every (cell, action)
    assert len(table.q) == 240 * 3


def test_diff_network_has_no_under_visited_and_full_grid() -> None:
    torch.manual_seed(0)
    report = diff_network(DQNAgent())
    assert len(report.cells) == 240
    assert report.category_counts.get("under_visited", 0) == 0  # N/A for a generalizing net
    assert 0.0 <= report.agreement_unweighted <= 1.0


def test_forced_stand_agrees_on_20_disagrees_on_5() -> None:
    torch.manual_seed(0)
    agent = DQNAgent()
    _force_action(agent, "stand")
    cells = {(c.player_value, c.is_soft, c.dealer_upcard): c for c in diff_network(agent).cells}

    hard20 = cells[(20, False, 7)]  # basic strategy stands on 20 -> agree
    assert hard20.agent_action == "stand"
    assert hard20.category == "agree"

    hard5 = cells[(5, False, 7)]  # basic strategy hits a 5 -> disagreement (net wrongly stands)
    assert hard5.agent_action == "stand"
    assert hard5.basic_action == "hit"
    assert hard5.category in ("near_equal_ev", "genuine_disagreement")
