"""Generic parallel cell-evaluation for the Problem-B baseline experiments (DESIGN D17, B2c+).

A **cell** = a labelled ``(SessionConfig, play Strategy, BetPolicy)`` to evaluate. ``evaluate_cells``
fans every cell across CPU cores — each (cell, worker) chunk seeded distinctly so it runs as its own
stream (the multi-stream model: regenerable from base_seed/n_workers/n_sessions/configs) — reduces
each session to per-session scalars *in-worker* (so only tiny arrays cross the process boundary, never
full captures), and aggregates per cell via the ``metrics`` scalar cores.

Shared by the bet-ladder, the Wonging experiment, and the bankroll sweep, so all three run on one
tested engine rather than three copies of the parallel plumbing. The scripts own only their cell
definitions, table rendering, and artifact provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from multiprocessing import Pool
from typing import Callable

from strategies.base import Strategy

from blackjack_rl.session.env import BetPolicy, SessionConfig, run_sessions
from blackjack_rl.session.metrics import (
    bankroll_distribution_of,
    drawdown_distribution_of,
    growth_rate_of,
    session_growth_rate,
    session_max_drawdown,
    wilson_interval,
)


@dataclass(frozen=True)
class Cell:
    """One labelled evaluation condition. ``label`` keys the result and must be unique within a run."""

    label: str
    config: SessionConfig
    play: Strategy
    bet: BetPolicy


def _worker(task: tuple[str, SessionConfig, Strategy, BetPolicy, int]) -> tuple:
    """Run one (cell, worker) chunk and reduce each session to scalars (top-level for Windows spawn)."""
    label, config, play, bet, n_chunk = task
    g: list[float] = []
    dd: list[float] = []
    final: list[float] = []
    ruined = 0
    for cap in run_sessions(config, play, bet, n_chunk):
        g.append(session_growth_rate(cap))
        dd.append(session_max_drawdown(cap))
        final.append(cap.final_bankroll)
        ruined += int(cap.ruined)
    return label, g, dd, final, ruined, len(g)


def _cell_metrics(part: dict, drawdown_level: float) -> dict:
    """Aggregate one cell's pooled per-session scalars into the outcome+risk metrics (JSON-able)."""
    growth = growth_rate_of(part["g"])
    return {
        "n_sessions": part["n"],
        "n_wiped": part["n"] - growth.n,  # -inf wipeouts dropped from growth (D2a), surfaced not hidden
        "growth_rate": asdict(growth),
        "ruin": asdict(wilson_interval(part["ruined"], part["n"])),
        "drawdown": asdict(drawdown_distribution_of(part["dd"])),
        f"drawdown_breach_{drawdown_level}": asdict(
            wilson_interval(sum(1 for d in part["dd"] if d >= drawdown_level), part["n"])
        ),
        "bankroll": asdict(bankroll_distribution_of(part["final"])),
    }


def evaluate_cells(
    cells: list[Cell],
    *,
    n_sessions: int,
    n_workers: int,
    base_seed: int = 0,
    drawdown_level: float = 0.5,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict], int]:
    """Evaluate every cell over ~``n_sessions`` sessions, parallelised across ``n_workers``.

    Returns ``(metrics_by_label, n_sessions_per_cell)``. ``on_progress(done, total)`` is called as each
    worker task completes (the caller renders elapsed/ETA). Each cell's config seed is overridden per
    worker so the chunks are independent streams.
    """
    per_worker = ceil(n_sessions / n_workers)
    tasks: list[tuple] = []
    seed = base_seed
    for cell in cells:
        for _ in range(n_workers):
            tasks.append((cell.label, replace(cell.config, seed=seed), cell.play, cell.bet, per_worker))
            seed += 1

    acc = {cell.label: {"g": [], "dd": [], "final": [], "ruined": 0, "n": 0} for cell in cells}
    with Pool(n_workers) as pool:
        for i, (label, g, dd, final, ruined, n) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            a = acc[label]
            a["g"] += g
            a["dd"] += dd
            a["final"] += final
            a["ruined"] += ruined
            a["n"] += n
            if on_progress is not None:
                on_progress(i, len(tasks))

    metrics = {label: _cell_metrics(part, drawdown_level) for label, part in acc.items()}
    return metrics, per_worker * n_workers
