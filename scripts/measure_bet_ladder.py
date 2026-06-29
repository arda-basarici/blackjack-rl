"""Bet-ladder evaluation — flat vs Kelly across the growth/ruin regimes (DESIGN D17, build stage B2c).

Rungs 1-2 of the baseline ladder (+ an over-bet foil), measured on identical terms: basic-strategy
play throughout, each bettor scored on the two axes (metrics.py) — outcome (log-growth rate,
final-bankroll shape) and risk (ruin probability, drawdown). Eight cells:

    {growth, ruin} config  x  {flat, Kelly-discrete, Kelly-continuous, flat-max-8} bettor

- **flat** (1u) is the under-bet floor (rung 1); **Kelly-discrete** snaps to the spread = the
  comparison baseline the learned DQN bettor (B2d) is audited against (same action set; decision A);
  **Kelly-continuous** is the analytic ceiling (unrounded f*.W) — the discrete->continuous gap is the
  cost of the finite menu.
- **flat-max-8** bets the spread top every hand = the naive over-bet foil. It populates the *ruin*
  axis (Kelly correctly declines the over-bet, so without a foil ruin is ~0 everywhere): ruin comes
  from over-betting, not counting — the restraint the DQN must learn (D14).

The parallel/reduce/aggregate plumbing lives in ``session.cell_eval``; this script only defines the
cells, renders the comparison tables, and stamps the artifact. Progress (elapsed + ETA) streams to
logs/live.log; the artifact lands under runs/.

Run from the repo root (default = 2000-session sanity pass; pass a count for the committed run):

    .venv\\Scripts\\python.exe scripts/measure_bet_ladder.py [n_sessions]
"""
from __future__ import annotations

import sys
from datetime import datetime

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR, RUNS_DIR
from blackjack_rl.core.persistence import save_run
from blackjack_rl.session.bet_agent import FlatBet, KellyBet
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import (
    BET_SPREAD,
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
DRAWDOWN_LEVEL = 0.5  # headline drawdown-breach threshold ("lost half the roll at some point")
CONFIGS = ("growth", "ruin")
BETTORS = ("flat", "kelly-disc", "kelly-cont", "flat-8")

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
    bettors = {
        "flat": FlatBet(1.0),
        "kelly-disc": KellyBet(curve, discretize=True),
        "kelly-cont": KellyBet(curve, discretize=False),
        "flat-8": FlatBet(float(max(BET_SPREAD))),  # over-bet foil: bet the spread top every hand
    }
    return [Cell(f"{c}/{b}", configs[c], BasicStrategy(), bettors[b]) for c in CONFIGS for b in BETTORS]


def _log_table(config_name: str, bankroll0: float, metrics: dict) -> None:
    """Stream a readable per-config comparison table (flat | kelly-disc | kelly-cont | flat-8)."""

    def cell(bettor: str, key: str) -> dict:
        return metrics[f"{config_name}/{bettor}"][key]

    n = metrics[f"{config_name}/flat"]["n_sessions"]
    _log(f"=== {config_name.upper()} config  (start {bankroll0:.0f}u, {n:,} sessions) ===")
    _row("metric", list(BETTORS))
    _row("growth/hand (x1e-4)", [f"{cell(b, 'growth_rate')['value'] * 1e4:+.3f}" for b in BETTORS])
    _row(
        "  95% CI",
        [
            f"[{cell(b, 'growth_rate')['low'] * 1e4:+.2f},{cell(b, 'growth_rate')['high'] * 1e4:+.2f}]"
            for b in BETTORS
        ],
    )
    _row("ruin %", [f"{cell(b, 'ruin')['estimate'] * 100:.2f}" for b in BETTORS])
    _row(
        f"P(dd>={DRAWDOWN_LEVEL:.0%}) %",
        [f"{cell(b, f'drawdown_breach_{DRAWDOWN_LEVEL}')['estimate'] * 100:.2f}" for b in BETTORS],
    )
    _row(
        "drawdown p50/p90 %",
        [
            f"{cell(b, 'drawdown')['quantiles'][0.5] * 100:.0f}/{cell(b, 'drawdown')['quantiles'][0.9] * 100:.0f}"
            for b in BETTORS
        ],
    )
    _row(
        "final bankroll p10/50/90",
        [
            f"{cell(b, 'bankroll')['quantiles'][0.1]:.0f}/{cell(b, 'bankroll')['quantiles'][0.5]:.0f}/{cell(b, 'bankroll')['quantiles'][0.9]:.0f}"
            for b in BETTORS
        ],
    )


def main() -> None:
    n_sessions = int(sys.argv[1]) if len(sys.argv) > 1 else SANITY_SESSIONS
    ref = load_edge_reference()
    cells = _build_cells(ref.kelly_curve)

    _log(
        f"bet ladder: {len(CONFIGS)}x{len(BETTORS)} cells, ~{n_sessions:,} sessions/cell "
        f"({N_WORKERS} workers), horizon {SessionConfig().max_hands} hands, "
        f"base_seed={BASE_SEED}, curve={ref.provenance.get('git_hash')}"
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
        "kind": "bet_ladder",
        "config": {
            "n_sessions_per_cell": n_actual,
            "n_workers": N_WORKERS,
            "base_seed": BASE_SEED,
            "seed": BASE_SEED,
            "horizon_max_hands": SessionConfig().max_hands,
            "growth_bankroll": GROWTH_BANKROLL,
            "ruin_bankroll": RUIN_BANKROLL,
            "drawdown_level": DRAWDOWN_LEVEL,
            "edge_reference": ref.provenance,
        },
        "elapsed_s": round(elapsed, 1),
        "cells": metrics,
    }
    run_id = f"{start.strftime('%Y%m%d-%H%M%S')}_bet-ladder_{n_actual}sess"
    run_dir = save_run(RUNS_DIR, record, run_id=run_id)
    _log(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
