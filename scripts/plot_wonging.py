"""Figure for the Wonging experiment (DESIGN D17, build stage B2c sidebar).

Renders the headline of ``measure_wonging.py``: forced (must bet the table minimum) vs wong (may sit
out non-positive counts) growth per config — the table-minimum tax flipping growth from negative to
positive. Derived view, written beside the artifact under ``runs/`` (git-ignored), never committed.

    .venv\\Scripts\\python.exe scripts/plot_wonging.py [runs/<run_id>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack_rl.core.paths import RUNS_DIR

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

CONFIGS = ("growth", "ruin")
MODES = ("forced", "wong")
COLORS = {"forced": "#c0392b", "wong": "#2e8b57"}


def latest_wonging_run(runs_dir: Path) -> Path:
    candidates = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir()
        and (d / "record.json").exists()
        and json.loads((d / "record.json").read_text(encoding="utf-8")).get("kind") == "wonging"
    ]
    if not candidates:
        raise FileNotFoundError(f"no wonging run under {runs_dir} — run measure_wonging.py first")
    return max(candidates, key=lambda d: d.name)


def render(record: dict, out: Path) -> Path:
    """Grouped bars: growth/hand (x1e-4, +/- CI) for forced vs wong, per config; break-even at 0."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    width = 0.38
    xs = range(len(CONFIGS))
    for j, mode in enumerate(MODES):
        vals, lo, hi = [], [], []
        for config in CONFIGS:
            g = record["cells"][f"{config}/{mode}"]["growth_rate"]
            vals.append(g["value"] * 1e4)
            lo.append((g["value"] - g["low"]) * 1e4)
            hi.append((g["high"] - g["value"]) * 1e4)
        offsets = [x + (j - 0.5) * width for x in xs]
        ax.bar(offsets, vals, width, yerr=[lo, hi], capsize=4, label=mode, color=COLORS[mode],
               alpha=0.85)
    ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.7)  # break-even
    labels = [f"{c}\n({record['config'][f'{c}_bankroll']:.0f}u)" for c in CONFIGS]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("growth/hand (x1e-4 log-wealth)")
    ax.set_title("Wonging — sitting out -EV hands flips growth positive\n"
                 "(forced = must bet table min; wong = back-count, bet 0 on bad counts)",
                 fontsize=10, fontweight="bold")
    ax.legend(title="table rule")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_wonging_run(RUNS_DIR)
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    print(f"figure -> {render(record, run_dir / 'wonging_growth.png')}")


if __name__ == "__main__":
    main()
