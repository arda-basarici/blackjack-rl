"""Figures for the bet-ladder evaluation (DESIGN D17, build stage B2c).

Renders the flat-vs-Kelly-vs-over-bet story from a saved ``bet_ladder`` artifact (``measure_bet_ladder
.py``), three figures each making one point:

- ``bet_ladder_pareto.png``   — growth vs risk (drawdown), the EV-vs-risk tradeoff, realistic bettors.
- ``bet_ladder_bankroll.png`` — final-bankroll box plots (all bettors; the over-bet's bust/moon split).
- ``bet_ladder_ruin.png``     — ruin probability with Wilson CIs (ruin comes from over-betting).

Derived views — regenerated from the artifact, written beside it under ``runs/`` (git-ignored),
never committed. Run from the repo root (defaults to the newest bet-ladder run; pass a run dir to pin):

    .venv\\Scripts\\python.exe scripts/plot_bet_ladder.py [runs/<run_id>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack_rl.core.paths import RUNS_DIR

plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "font.family": "sans-serif"})

CONFIGS = ("growth", "ruin")
REALISTIC = ("flat", "kelly-disc", "kelly-cont")  # the Pareto-comparable rungs (over-bet excluded)
ALL_BETTORS = ("flat", "kelly-disc", "kelly-cont", "flat-8")
COLORS = {"flat": "#7f7f7f", "kelly-disc": "#1f5fbf", "kelly-cont": "#3ca0a0", "flat-8": "#c0392b"}


def latest_ladder_run(runs_dir: Path) -> Path:
    """Newest run dir holding a ``bet_ladder`` artifact."""
    candidates = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir()
        and (d / "record.json").exists()
        and json.loads((d / "record.json").read_text(encoding="utf-8")).get("kind") == "bet_ladder"
    ]
    if not candidates:
        raise FileNotFoundError(f"no bet_ladder run under {runs_dir} — run measure_bet_ladder.py first")
    return max(candidates, key=lambda d: d.name)


def _cell(record: dict, config: str, bettor: str) -> dict:
    return record["cells"][f"{config}/{bettor}"]


def _bankroll0(record: dict, config: str) -> float:
    return record["config"][f"{config}_bankroll"]


def render_pareto(record: dict, out: Path) -> Path:
    """Growth (y, x1e-4 +/- CI) vs risk (x, drawdown p90 %) — the EV-vs-risk layout, realistic rungs."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=False)
    for ax, config in zip(axes, CONFIGS):
        for b in REALISTIC:
            c = _cell(record, config, b)
            x = c["drawdown"]["quantiles"]["0.9"] * 100
            y = c["growth_rate"]["value"] * 1e4
            lo = (c["growth_rate"]["value"] - c["growth_rate"]["low"]) * 1e4
            hi = (c["growth_rate"]["high"] - c["growth_rate"]["value"]) * 1e4
            ax.errorbar(x, y, yerr=[[lo], [hi]], fmt="o", color=COLORS[b], capsize=3, ms=7)
            ax.annotate(b, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)  # break-even
        ax.set_title(f"{config} ({_bankroll0(record, config):.0f}u)")
        ax.set_xlabel("risk — drawdown p90 (%)")
        ax.set_ylabel("growth/hand (x1e-4 log-wealth)")
        ax.grid(alpha=0.25)
    fig.suptitle("Bet ladder — growth vs risk (higher & left is better)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_bankroll(record: dict, out: Path) -> Path:
    """Final-bankroll box plots (p10/p50/p90 box, min/max whiskers) per bettor; line at the start."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, config in zip(axes, CONFIGS):
        stats = []
        for b in ALL_BETTORS:
            bk = _cell(record, config, b)["bankroll"]
            q = bk["quantiles"]
            stats.append(
                {
                    "label": b,
                    "med": q["0.5"],
                    "q1": q["0.1"],
                    "q3": q["0.9"],
                    "whislo": bk["minimum"],
                    "whishi": bk["maximum"],
                }
            )
        boxes = ax.bxp(stats, showfliers=False, patch_artist=True)
        for patch, b in zip(boxes["boxes"], ALL_BETTORS):
            patch.set_facecolor(COLORS[b])
            patch.set_alpha(0.55)
        ax.axhline(_bankroll0(record, config), color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"{config} (start {_bankroll0(record, config):.0f}u)")
        ax.set_ylabel("final bankroll (units)")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Bet ladder — final-bankroll distribution (box = p10/p50/p90, whisker = min/max)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_ruin(record: dict, out: Path) -> Path:
    """Ruin probability (%) with Wilson 95% CIs — flat & Kelly stay ~0; only the over-bet ruins."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, config in zip(axes, CONFIGS):
        for i, b in enumerate(ALL_BETTORS):
            r = _cell(record, config, b)["ruin"]
            est = r["estimate"] * 100
            lo = (r["estimate"] - r["low"]) * 100
            hi = (r["high"] - r["estimate"]) * 100
            ax.bar(i, est, color=COLORS[b], alpha=0.8)
            ax.errorbar(i, est, yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3, lw=0.9)
        ax.set_xticks(range(len(ALL_BETTORS)))
        ax.set_xticklabels(ALL_BETTORS, rotation=15)
        ax.set_title(f"{config} ({_bankroll0(record, config):.0f}u)")
        ax.set_ylabel("ruin probability (%)")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Bet ladder — ruin comes from over-betting, not counting", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_ladder_run(RUNS_DIR)
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    for fn, name in (
        (render_pareto, "bet_ladder_pareto.png"),
        (render_bankroll, "bet_ladder_bankroll.png"),
        (render_ruin, "bet_ladder_ruin.png"),
    ):
        print(f"figure -> {fn(record, run_dir / name)}")


if __name__ == "__main__":
    main()
