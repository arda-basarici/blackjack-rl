"""Figure for the bankroll sweep (DESIGN D17, build stage B2c sidebar).

Growth/hand vs starting bankroll for continuous full-Kelly under both table rules — the forced curve
climbing toward the flat wong line as the table-minimum tax (~1/bankroll) fades. Derived view, written
beside the artifact under ``runs/`` (git-ignored), never committed.

    .venv\\Scripts\\python.exe scripts/plot_sweep.py [runs/<run_id>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from blackjack_rl.core.paths import RUNS_DIR

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

MODES = ("forced", "wong")
COLORS = {"forced": "#c0392b", "wong": "#2e8b57"}


def latest_sweep_run(runs_dir: Path) -> Path:
    candidates = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir()
        and (d / "record.json").exists()
        and json.loads((d / "record.json").read_text(encoding="utf-8")).get("kind") == "bankroll_sweep"
    ]
    if not candidates:
        raise FileNotFoundError(f"no bankroll_sweep run under {runs_dir} — run measure_sweep.py first")
    return max(candidates, key=lambda d: d.name)


def render(record: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bankrolls = record["config"]["bankrolls"]
    for mode in MODES:
        ys, lo, hi = [], [], []
        for bk in bankrolls:
            g = record["cells"][f"{bk}/{mode}"]["growth_rate"]
            ys.append(g["value"] * 1e4)
            lo.append((g["value"] - g["low"]) * 1e4)
            hi.append((g["high"] - g["value"]) * 1e4)
        ax.errorbar(bankrolls, ys, yerr=[lo, hi], marker="o", capsize=3, lw=1.6, label=mode,
                    color=COLORS[mode])
    ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.7)  # break-even
    ax.set_xscale("log")
    ax.set_xticks(bankrolls)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("starting bankroll (units, log scale)")
    ax.set_ylabel("growth/hand (x1e-4 log-wealth)")
    ax.set_title("Bankroll sweep — the table-minimum tax fades as the roll grows\n"
                 "(forced climbs toward the scale-free wong line)", fontsize=10, fontweight="bold")
    ax.legend(title="table rule")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_sweep_run(RUNS_DIR)
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    print(f"figure -> {render(record, run_dir / 'bankroll_sweep.png')}")


if __name__ == "__main__":
    main()
