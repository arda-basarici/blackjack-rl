"""Four-axis evaluation of the learned ``BetAgent`` vs the analytic baselines (B2d-3 deliverable).

Loads a trained agent (``load_bet_agent``) and scores it on the full ``metrics`` suite — never collapsed
— against discrete-``KellyBet`` and ``FlatBet``, in **both** regimes, on identical terms via
``cell_eval``:

    {growth, ruin} config  x  {agent, kelly-disc, flat} bettor   (6 cells)

Outcome axis (growth-rate ± CI, final-bankroll quantiles) and risk axis (ruin Wilson, drawdown) are
reported side by side, never merged. Plus the bet-vs-count curve overlaid on Kelly (a diagnostic). The
agent is the same loaded policy in both regimes (trained on its own regime; evaluating it in the other
is an honest cross-check, not a claim of optimality there).

Thread note: this is a torch policy fanned across CPU workers, so OMP/MKL threads are pinned to 1 here
(workers inherit it) to avoid n_workers x threads oversubscription — inference on this tiny net is
single-thread-fine.

    .venv\\Scripts\\python.exe scripts/eval_bet_agent.py <run_dir> [n_sessions]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # pin BEFORE torch import; spawned eval workers inherit it
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from datetime import datetime

import torch
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.session.bet_agent import FlatBet, KellyBet, greedy_bet_curve
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import GROWTH_BANKROLL, RUIN_BANKROLL, growth_config, ruin_config
from blackjack_rl.session.persistence import load_bet_agent
from blackjack_rl.session.references import load_edge_reference

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
N_WORKERS = 12
DRAWDOWN_LEVEL = 0.5
CONFIGS = ("growth", "ruin")
BETTORS = ("agent", "kelly", "flat")


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    msg = f"[{datetime.now():%H:%M:%S}] {line}"
    print(msg, flush=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _row(label: str, values: list[str]) -> None:
    _log(f"{label:<26}" + "".join(f"{v:>16}" for v in values))


def _build_cells(agent, curve: dict[int, float]) -> list[Cell]:
    configs = {"growth": growth_config(), "ruin": ruin_config()}
    bettors = {"agent": agent, "kelly": KellyBet(curve, discretize=True), "flat": FlatBet(1.0)}
    return [Cell(f"{c}/{b}", configs[c], BasicStrategy(), bettors[b]) for c in CONFIGS for b in BETTORS]


def _log_table(config_name: str, bankroll0: float, m: dict) -> None:
    def cell(b: str, key: str) -> dict:
        return m[f"{config_name}/{b}"][key]

    n = m[f"{config_name}/agent"]["n_sessions"]
    _log(f"=== {config_name.upper()} (start {bankroll0:.0f}u, {n:,} sessions) ===")
    _row("metric", list(BETTORS))
    _row("growth/hand (x1e-4)", [f"{cell(b, 'growth_rate')['value'] * 1e4:+.3f}" for b in BETTORS])
    _row("  95% CI", [
        f"[{cell(b, 'growth_rate')['low'] * 1e4:+.2f},{cell(b, 'growth_rate')['high'] * 1e4:+.2f}]"
        for b in BETTORS
    ])
    _row("ruin %", [f"{cell(b, 'ruin')['estimate'] * 100:.2f}" for b in BETTORS])
    _row(f"P(dd>={DRAWDOWN_LEVEL:.0%}) %",
         [f"{cell(b, f'drawdown_breach_{DRAWDOWN_LEVEL}')['estimate'] * 100:.2f}" for b in BETTORS])
    _row("final bankroll p10/50/90", [
        f"{cell(b, 'bankroll')['quantiles'][0.1]:.0f}/{cell(b, 'bankroll')['quantiles'][0.5]:.0f}/"
        f"{cell(b, 'bankroll')['quantiles'][0.9]:.0f}" for b in BETTORS
    ])


def _log_bet_curve(agent, curve: dict[int, float]) -> None:
    """Diagnostic: the learned greedy bet vs discrete-Kelly across the probe counts (growth bankroll)."""
    agent_bets = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=GROWTH_BANKROLL, decks_remaining=3.0)
    kelly = KellyBet(curve, discretize=True)
    kelly_bets = {c: kelly.bet(true_count=float(c), decks_remaining=3.0, bankroll=GROWTH_BANKROLL)
                  for c in PROBE_COUNTS}
    _log("=== bet-vs-count (greedy, @400u) ===")
    _row("true count", [f"{c:+d}" for c in PROBE_COUNTS])
    _row("agent", [f"{agent_bets[c]:g}" for c in PROBE_COUNTS])
    _row("kelly", [f"{kelly_bets[c]:g}" for c in PROBE_COUNTS])


def main() -> None:
    run_dir = sys.argv[1]
    n_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    torch.set_num_threads(1)

    agent = load_bet_agent(run_dir)
    ref = load_edge_reference()
    cells = _build_cells(agent, ref.kelly_curve)

    _log(f"eval_bet_agent: {len(cells)} cells, ~{n_sessions:,} sessions/cell, {N_WORKERS} workers, "
         f"agent={run_dir}")
    start = datetime.now()

    def on_progress(done: int, total: int) -> None:
        if done == total or done % N_WORKERS == 0:
            _log(f"  {done}/{total} tasks")

    metrics, n_actual = evaluate_cells(
        cells, n_sessions=n_sessions, n_workers=N_WORKERS, base_seed=0,
        drawdown_level=DRAWDOWN_LEVEL, on_progress=on_progress,
    )
    elapsed = (datetime.now() - start).total_seconds()
    _log(f"--- results ({elapsed:.0f}s, {n_actual:,} sessions/cell) ---")
    _log_bet_curve(agent, ref.kelly_curve)
    _log_table("growth", GROWTH_BANKROLL, metrics)
    _log_table("ruin", RUIN_BANKROLL, metrics)


if __name__ == "__main__":
    main()
