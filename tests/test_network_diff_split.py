"""Tests for the split column in the network diff: pair cells are enumerated and scored against
basic strategy when the agent plays splits, and a no-split agent is unaffected (A11)."""
from __future__ import annotations

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.dqn.network_diff import diff_network, enumerate_cells


def test_enumerate_cells_adds_pair_column() -> None:
    base = enumerate_cells(with_splits=False)
    full = enumerate_cells(with_splits=True)
    assert all(len(c) == 4 for c in base + full)        # uniform (value, soft, upcard, can_split)
    assert all(c[3] is False for c in base)             # no-split grid: never can_split
    pairs = [c for c in full if c[3]]
    assert len(pairs) == 100                            # 10 pairs x 10 upcards
    assert len(full) == len(base) + 100
    assert (12, True, 6, True) in full                  # A,A vs dealer 6 (soft 12, splittable)
    assert (16, False, 10, True) in full                # 8,8 vs dealer 10
    assert (16, False, 10, False) in base               # hard 16 stays a separate, non-pair cell


def test_diff_network_scores_split_cells() -> None:
    agent = DQNAgent(epsilon=0.0, with_splits=True, encoding="onehot")
    report = diff_network(agent)
    keys = {(c.player_value, c.is_soft, c.dealer_upcard, c.can_split) for c in report.cells}
    assert (16, False, 10, True) in keys                # the 8,8 pair cell is scored
    assert (16, False, 10, False) in keys               # hard 16 is still there, distinct
    assert any(c.basic_action == "split" for c in report.cells)  # basic strategy splits some pair
    assert any(c.can_split for c in report.cells)


def test_no_split_agent_unaffected() -> None:
    report = diff_network(DQNAgent(epsilon=0.0, with_splits=False, encoding="onehot"))
    assert all(not c.can_split for c in report.cells)   # no pair cells leak in
    assert len(report.cells) == 240                     # the original no-split grid
