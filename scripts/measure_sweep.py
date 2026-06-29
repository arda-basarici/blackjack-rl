"""Bankroll sweep — growth vs starting bankroll (DESIGN D17, build stage B2c sidebar).

Full-Kelly's net loss is the table-minimum tax, which fades as ~1/bankroll (the Wonging sidebar
isolated it). This sweeps the bankroll to *show that curve*: continuous full-Kelly across a range of
starting bankrolls, under both table rules —

    {100, 200, 400, 800, 1600}u  x  {forced (min_wager=1), wong (min_wager=0)}

Expectation: **forced** growth climbs toward 0 and beyond as the bankroll grows (the fixed 1u tax
shrinks relative to the roll), while **wong** is ~flat-positive (the scale-free Kelly growth, no tax).
The two converge at large bankroll — the tax made visible as a curve. Reuses ``session.cell_eval``.

    .venv\\Scripts\\python.exe scripts/measure_sweep.py [n_sessions]
"""
from __future__ import annotations

import sys
from datetime import datetime

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR, RUNS_DIR
from blackjack_rl.core.persistence import save_run
from blackjack_rl.session.bet_agent import KellyBet
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import SessionConfig
from blackjack_rl.session.references import load_edge_reference

SANITY_SESSIONS = 2_000
N_WORKERS = 20
BASE_SEED = 0
BANKROLLS = (100, 200, 400, 800, 1600)
MODES = {"forced": 1.0, "wong": 0.0}  # min_wager

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass


def _log(line: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[{stamp}] {line}"
    print(msg, flush=True)
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


def _build_cells(curve: dict[int, float]) -> list[Cell]:
    kelly = KellyBet(curve, discretize=False)
    return [
        Cell(f"{bk}/{m}", SessionConfig(starting_bankroll=float(bk), min_wager=mw), BasicStrategy(), kelly)
        for bk in BANKROLLS
        for m, mw in MODES.items()
    ]


def _log_table(metrics: dict) -> None:
    def growth(bk: int, mode: str) -> dict:
        return metrics[f"{bk}/{mode}"]["growth_rate"]

    _log(f"{'bankroll':>10}{'forced growth(x1e-4, CI)':>30}{'wong growth(x1e-4, CI)':>30}")
    for bk in BANKROLLS:
        cells = []
        for m in MODES:
            g = growth(bk, m)
            cells.append(f"{g['value'] * 1e4:+.3f} [{g['low'] * 1e4:+.2f},{g['high'] * 1e4:+.2f}]")
        _log(f"{bk:>10}{cells[0]:>30}{cells[1]:>30}")


def main() -> None:
    n_sessions = int(sys.argv[1]) if len(sys.argv) > 1 else SANITY_SESSIONS
    ref = load_edge_reference()
    cells = _build_cells(ref.kelly_curve)

    _log(
        f"bankroll sweep: {len(BANKROLLS)}x{len(MODES)} cells (continuous Kelly, forced vs wong), "
        f"bankrolls={BANKROLLS}, ~{n_sessions:,} sessions/cell ({N_WORKERS} workers), "
        f"curve={ref.provenance.get('git_hash')}"
    )
    start = datetime.now()

    def on_progress(done: int, total: int) -> None:
        elapsed = (datetime.now() - start).total_seconds()
        eta = elapsed * (total - done) / done
        _log(f"  {done}/{total} tasks done | elapsed {_fmt_dur(elapsed)} | ETA ~{_fmt_dur(eta)}")

    metrics, n_actual = evaluate_cells(
        cells, n_sessions=n_sessions, n_workers=N_WORKERS, base_seed=BASE_SEED, on_progress=on_progress
    )
    elapsed = (datetime.now() - start).total_seconds()

    _log(f"--- results ({_fmt_dur(elapsed)}) ---")
    _log_table(metrics)

    record = {
        "kind": "bankroll_sweep",
        "config": {
            "n_sessions_per_cell": n_actual,
            "n_workers": N_WORKERS,
            "base_seed": BASE_SEED,
            "seed": BASE_SEED,
            "horizon_max_hands": SessionConfig().max_hands,
            "bankrolls": list(BANKROLLS),
            "modes": MODES,
            "edge_reference": ref.provenance,
        },
        "elapsed_s": round(elapsed, 1),
        "cells": metrics,
    }
    run_id = f"{start.strftime('%Y%m%d-%H%M%S')}_bankroll-sweep_{n_actual}sess"
    run_dir = save_run(RUNS_DIR, record, run_id=run_id)
    _log(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
