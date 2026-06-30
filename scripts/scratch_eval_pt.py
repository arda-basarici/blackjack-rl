"""Quick four-axis eval of a *scratch* `.pt` agent bundle (loads it directly, not via a run_dir).

Answers "does the wandering curve actually MATTER?" — scores the trained agent vs discrete-Kelly + Flat
on full sessions (growth / ruin / drawdown / bankroll), both regimes, via cell_eval. Throwaway probe.

    .venv\\Scripts\\python.exe scripts/scratch_eval_pt.py <path.pt> [n_sessions]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

import torch
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.session.bet_agent import BetAgent, FlatBet, KellyBet, greedy_bet_curve
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import GROWTH_BANKROLL, RUIN_BANKROLL, growth_config, ruin_config
from blackjack_rl.session.references import load_edge_reference

PROBE = (-4, -2, 0, 2, 4, 6, 8)
N_WORKERS, DD = 12, 0.5
CONFIGS, BETTORS = ("growth", "ruin"), ("agent", "kelly", "flat")


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    print(line, flush=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_pt(path: str) -> BetAgent:
    d = torch.load(path, weights_only=False)
    c = d["construction"]
    agent = BetAgent(
        levels=c["levels"], hidden=tuple(c["hidden"]), epsilon=0.0,
        num_decks=c["num_decks"], bankroll_scale=c["bankroll_scale"],
    )
    agent.q_net.load_state_dict(d["state_dict"])
    agent.q_net.eval()
    return agent


def _row(label: str, vals: list[str]) -> None:
    _log(f"{label:<22}" + "".join(f"{v:>15}" for v in vals))


def main() -> None:
    pt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    torch.set_num_threads(1)

    agent = _load_pt(pt)
    ref = load_edge_reference()
    configs = {"growth": growth_config(), "ruin": ruin_config()}
    bettors = {"agent": agent, "kelly": KellyBet(ref.kelly_curve, discretize=True), "flat": FlatBet(1.0)}
    cells = [Cell(f"{c}/{b}", configs[c], BasicStrategy(), bettors[b]) for c in CONFIGS for b in BETTORS]

    _log(f"eval {pt}  ({len(cells)} cells x ~{n} sessions)")
    metrics, na = evaluate_cells(cells, n_sessions=n, n_workers=N_WORKERS, base_seed=0, drawdown_level=DD)

    ab = greedy_bet_curve(agent, PROBE, bankroll=GROWTH_BANKROLL, decks_remaining=3.0)
    _log("agent bet-vs-count: " + " ".join(f"{c:+d}:{ab[c]:g}" for c in PROBE) + "   (Kelly 1 1 1 2 5 8 8)")
    for cfg, b0 in (("growth", GROWTH_BANKROLL), ("ruin", RUIN_BANKROLL)):
        def cell(b: str, k: str, _cfg: str = cfg) -> dict:
            return metrics[f"{_cfg}/{b}"][k]
        _log(f"=== {cfg.upper()} (start {b0:.0f}u, {na:,} sessions) ===")
        _row("metric", list(BETTORS))
        _row("growth/hand x1e-4", [f"{cell(b, 'growth_rate')['value'] * 1e4:+.3f}" for b in BETTORS])
        _row("ruin %", [f"{cell(b, 'ruin')['estimate'] * 100:.2f}" for b in BETTORS])
        _row("P(dd>=50%) %", [f"{cell(b, f'drawdown_breach_{DD}')['estimate'] * 100:.2f}" for b in BETTORS])
        _row("bankroll p10/50/90", [
            f"{cell(b, 'bankroll')['quantiles'][0.1]:.0f}/{cell(b, 'bankroll')['quantiles'][0.5]:.0f}/"
            f"{cell(b, 'bankroll')['quantiles'][0.9]:.0f}" for b in BETTORS
        ])


if __name__ == "__main__":
    main()
