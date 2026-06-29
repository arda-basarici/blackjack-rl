"""Wonging experiment — the cost of the table minimum (DESIGN D17, build stage B2c sidebar).

Full-Kelly's net loss at modest bankrolls (the bet-ladder result) is driven by the **mandatory table
minimum**: you must bet 1u even on the frequent non-positive-count hands (−EV), and that tax drags
growth below zero. This isolates it: run continuous full-Kelly (which wants to bet 0 at a non-positive
edge) under two table rules —

    {growth, ruin} config  x  {forced (min_wager=1), wong (min_wager=0)}

- **forced** — must bet >=1u every hand (the realistic casino; what the bet-ladder measured).
- **wong** — may bet 0 on non-positive counts = back-counting / "Wonging" (the hand is still dealt, so
  the shoe + count advance, but no −EV stake is risked).

Same cards either way (the bet doesn't change the deal) — so the forced-vs-wong gap is purely the
table-minimum tax / the value of being able to sit out. Reuses ``session.cell_eval``.

Run from the repo root (default = 2000-session sanity pass; pass a count for the committed run):

    .venv\\Scripts\\python.exe scripts/measure_wonging.py [n_sessions]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR, RUNS_DIR
from blackjack_rl.core.persistence import save_run
from blackjack_rl.session.bet_agent import KellyBet
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import (
    GROWTH_BANKROLL,
    RUIN_BANKROLL,
    SessionConfig,
    growth_config,
    ruin_config,
)
from blackjack_rl.session.references import load_edge_reference

SANITY_SESSIONS = 2_000
N_WORKERS = 20
BASE_SEED = 0
DRAWDOWN_LEVEL = 0.5
CONFIGS = ("growth", "ruin")
MODES = {"forced": 1.0, "wong": 0.0}  # min_wager: forced table minimum vs sit-out allowed

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


def _row(label: str, values: list[str]) -> None:
    _log(f"{label:<26}" + "".join(f"{v:>16}" for v in values))


def _build_cells(curve: dict[int, float]) -> list[Cell]:
    configs = {"growth": growth_config(), "ruin": ruin_config()}
    kelly = KellyBet(curve, discretize=False)  # continuous full-Kelly wants 0 at a non-positive edge
    return [
        Cell(f"{c}/{m}", replace(configs[c], min_wager=mw), BasicStrategy(), kelly)
        for c in CONFIGS
        for m, mw in MODES.items()
    ]


def _log_table(config_name: str, bankroll0: float, metrics: dict) -> None:
    """forced vs wong for one config — the growth row is the table-minimum tax made visible."""

    def cell(mode: str, key: str) -> dict:
        return metrics[f"{config_name}/{mode}"][key]

    modes = list(MODES)
    n = metrics[f"{config_name}/forced"]["n_sessions"]
    _log(f"=== {config_name.upper()} config  (start {bankroll0:.0f}u, {n:,} sessions) ===")
    _row("metric", modes)
    _row("growth/hand (x1e-4)", [f"{cell(m, 'growth_rate')['value'] * 1e4:+.3f}" for m in modes])
    _row(
        "  95% CI",
        [
            f"[{cell(m, 'growth_rate')['low'] * 1e4:+.2f},{cell(m, 'growth_rate')['high'] * 1e4:+.2f}]"
            for m in modes
        ],
    )
    _row("ruin %", [f"{cell(m, 'ruin')['estimate'] * 100:.2f}" for m in modes])
    _row(
        "drawdown p50/p90 %",
        [
            f"{cell(m, 'drawdown')['quantiles'][0.5] * 100:.0f}/{cell(m, 'drawdown')['quantiles'][0.9] * 100:.0f}"
            for m in modes
        ],
    )
    _row(
        "final bankroll p10/50/90",
        [
            f"{cell(m, 'bankroll')['quantiles'][0.1]:.0f}/{cell(m, 'bankroll')['quantiles'][0.5]:.0f}/{cell(m, 'bankroll')['quantiles'][0.9]:.0f}"
            for m in modes
        ],
    )


def main() -> None:
    n_sessions = int(sys.argv[1]) if len(sys.argv) > 1 else SANITY_SESSIONS
    ref = load_edge_reference()
    cells = _build_cells(ref.kelly_curve)

    _log(
        f"wonging: {len(CONFIGS)}x{len(MODES)} cells (continuous Kelly, forced vs sit-out), "
        f"~{n_sessions:,} sessions/cell ({N_WORKERS} workers), curve={ref.provenance.get('git_hash')}"
    )
    start = datetime.now()

    def on_progress(done: int, total: int) -> None:
        elapsed = (datetime.now() - start).total_seconds()
        eta = elapsed * (total - done) / done
        _log(f"  {done}/{total} tasks done | elapsed {_fmt_dur(elapsed)} | ETA ~{_fmt_dur(eta)}")

    metrics, n_actual = evaluate_cells(
        cells,
        n_sessions=n_sessions,
        n_workers=N_WORKERS,
        base_seed=BASE_SEED,
        drawdown_level=DRAWDOWN_LEVEL,
        on_progress=on_progress,
    )
    elapsed = (datetime.now() - start).total_seconds()

    _log(f"--- results ({_fmt_dur(elapsed)}) ---")
    _log_table("growth", GROWTH_BANKROLL, metrics)
    _log_table("ruin", RUIN_BANKROLL, metrics)

    record = {
        "kind": "wonging",
        "config": {
            "n_sessions_per_cell": n_actual,
            "n_workers": N_WORKERS,
            "base_seed": BASE_SEED,
            "seed": BASE_SEED,
            "horizon_max_hands": SessionConfig().max_hands,
            "growth_bankroll": GROWTH_BANKROLL,
            "ruin_bankroll": RUIN_BANKROLL,
            "modes": MODES,
            "edge_reference": ref.provenance,
        },
        "elapsed_s": round(elapsed, 1),
        "cells": metrics,
    }
    run_id = f"{start.strftime('%Y%m%d-%H%M%S')}_wonging_{n_actual}sess"
    run_dir = save_run(RUNS_DIR, record, run_id=run_id)
    _log(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
