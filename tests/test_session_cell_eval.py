"""Tests for the parallel cell-evaluation engine (DESIGN D17, B2c) — the shared runner behind the
bet-ladder, Wonging, and bankroll-sweep scripts.

Confirms the engine wires up (labels -> metric dicts, correct session counts) and is deterministic:
the per-worker streams are fixed by ``base_seed``, and the aggregates are order-independent, so two
runs match even though ``imap_unordered`` returns chunks in arbitrary order.
"""
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.session.bet_agent import FlatBet, KellyBet
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import SessionConfig


def test_evaluate_cells_structure_and_determinism():
    cfg = SessionConfig(starting_bankroll=400.0, max_hands=20)
    cells = [
        Cell("a/flat", cfg, BasicStrategy(), FlatBet(1.0)),
        Cell("b/kelly", cfg, BasicStrategy(), KellyBet({0: 0.0, 4: 0.01})),
    ]
    metrics, n = evaluate_cells(cells, n_sessions=4, n_workers=2, base_seed=0)

    assert set(metrics) == {"a/flat", "b/kelly"}
    assert n == 4  # ceil(4/2) * 2 workers
    for cell_m in metrics.values():
        assert {"growth_rate", "ruin", "drawdown", "bankroll", "n_sessions"} <= set(cell_m)
        assert cell_m["n_sessions"] == 4

    again, _ = evaluate_cells(cells, n_sessions=4, n_workers=2, base_seed=0)
    assert metrics == again  # order-independent aggregates -> reproducible


def test_evaluate_cells_progress_callback_fires_per_task():
    cfg = SessionConfig(starting_bankroll=400.0, max_hands=10)
    cells = [Cell("only", cfg, BasicStrategy(), FlatBet(1.0))]
    seen: list[tuple[int, int]] = []
    evaluate_cells(cells, n_sessions=2, n_workers=2, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (2, 2)  # one task per worker, all reported, final = (total, total)
