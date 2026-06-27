"""Signature figure for the edge-by-count measurement (B2a, commit ii).

Renders the player edge vs Hi-Lo true count from a saved ``edge_by_count`` artifact (the runner,
``measure_edge_by_count.py``): edge% with 95% CIs on the left axis, the implied full-Kelly fraction
on a twin right axis, the break-even count, and a shaded low-n tail where buckets are too sparse to
trust. The figure is a *derived view* — regenerated from the artifact, saved beside it under
``runs/`` (git-ignored), never committed.

Run from the repo root (defaults to the newest edge-by-count run; pass a run dir to pin one):

    .venv\\Scripts\\python.exe scripts/plot_edge_by_count.py [runs/<run_id>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display
import matplotlib.pyplot as plt

from blackjack_rl.core.paths import RUNS_DIR

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

DISPLAY_MIN_N = 2_000      # drop the wild singleton tails (TC |.|>~12), unreadable at any scale
RELIABLE_MIN_N = 50_000    # at/above this a bucket's CI is tight; below it we fade + shade as low-n
EDGE_COLOR = "#1f5fbf"
KELLY_COLOR = "#c0392b"


def latest_edge_run(runs_dir: Path) -> Path:
    """Newest run dir holding an ``edge_by_count`` artifact (timestamp prefix sorts lexically)."""
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "record.json").exists()
        and json.loads((d / "record.json").read_text(encoding="utf-8")).get("kind") == "edge_by_count"
    ]
    if not candidates:
        raise FileNotFoundError(f"no edge_by_count run under {runs_dir} — run measure_edge_by_count.py first")
    return max(candidates, key=lambda d: d.name)


def break_even_tc(points: list[dict]) -> float | None:
    """The true count where edge crosses zero, linearly interpolated between the straddling integer
    buckets (the classic break-even count). None if the curve never crosses in the measured range."""
    ordered = sorted(points, key=lambda p: p["true_count"])
    for lo, hi in zip(ordered, ordered[1:]):
        if lo["mean_return"] < 0 <= hi["mean_return"]:
            span = hi["mean_return"] - lo["mean_return"]
            frac = -lo["mean_return"] / span if span else 0.0
            return lo["true_count"] + frac * (hi["true_count"] - lo["true_count"])
    return None


def render(record: dict, out_path: Path) -> Path:
    """Draw the edge-vs-count signature figure from a record and write it to ``out_path``."""
    pts = sorted(
        (e for e in record["edges"] if e["n"] >= DISPLAY_MIN_N), key=lambda e: e["true_count"]
    )
    kelly = record["kelly_curve"]

    tc = [p["true_count"] for p in pts]
    edge = [p["mean_return"] * 100 for p in pts]
    ci95 = [1.96 * p["std_error"] * 100 for p in pts]  # 95% CI half-width
    reliable = [p["n"] >= RELIABLE_MIN_N for p in pts]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    # --- shaded low-n tails: the count range outside the contiguous reliable band ---
    rel_tc = [t for t, ok in zip(tc, reliable) if ok]
    if rel_tc:
        lo_edge, hi_edge = min(tc) - 0.5, max(tc) + 0.5
        if min(rel_tc) - 0.5 > lo_edge:
            ax.axvspan(lo_edge, min(rel_tc) - 0.5, color="0.85", alpha=0.5, lw=0, label="low n (wide CI)")
        if max(rel_tc) + 0.5 < hi_edge:
            ax.axvspan(max(rel_tc) + 0.5, hi_edge, color="0.85", alpha=0.5, lw=0)

    # --- edge curve with 95% CI error bars (faded where low-n) ---
    for t, e, c, ok in zip(tc, edge, ci95, reliable):
        ax.errorbar(
            t, e, yerr=c, fmt="o", ms=4, color=EDGE_COLOR,
            ecolor=EDGE_COLOR, elinewidth=1.1, capsize=2.5, alpha=1.0 if ok else 0.35,
        )
    ax.plot(tc, edge, "-", color=EDGE_COLOR, lw=1.3, alpha=0.6, zorder=0)

    ax.axhline(0, color="0.4", lw=0.9, ls="-")  # zero-edge reference
    be = break_even_tc(pts)
    if be is not None:
        ax.axvline(be, color="0.3", lw=1.0, ls="--")
        ax.annotate(
            f"break-even ≈ TC {be:+.2f}", xy=(be, 0), xytext=(be + 0.4, min(edge) * 0.45),
            fontsize=9, color="0.25",
        )

    ax.set_xlabel("Hi-Lo true count")
    ax.set_ylabel("player edge  (% per unit bet)", color=EDGE_COLOR)
    ax.tick_params(axis="y", labelcolor=EDGE_COLOR)
    ax.grid(True, axis="both", color="0.92", lw=0.6)

    # --- twin axis: implied full-Kelly fraction (different quantity, own scale) ---
    ax2 = ax.twinx()
    n_by_tc = {p["true_count"]: p["n"] for p in pts}
    k_tc = [t for t in tc if str(t) in kelly]
    k_f = [kelly[str(t)] * 100 for t in k_tc]
    # solid through the reliable band, faded in the low-n tail (matches the edge points)
    k_reliable = [n_by_tc[t] >= RELIABLE_MIN_N for t in k_tc]
    ax2.plot(k_tc, k_f, "-", color=KELLY_COLOR, lw=1.3, alpha=0.35, zorder=0)
    for t, f, ok in zip(k_tc, k_f, k_reliable):
        ax2.plot(t, f, "s", color=KELLY_COLOR, ms=3.5, alpha=1.0 if ok else 0.35)
    ax2.plot([], [], "s-", color=KELLY_COLOR, ms=3.5, lw=1.3, label="full Kelly f*")  # legend proxy
    ax2.set_ylabel("full-Kelly bet  (% of bankroll)", color=KELLY_COLOR)
    ax2.tick_params(axis="y", labelcolor=KELLY_COLOR)
    ax2.set_ylim(bottom=0)

    cfg, anc = record["config"], record["anchor_check"]
    fig.suptitle(
        f"Player edge vs true count — {record['n_total']:,} hands, flat-bet basic strategy",
        fontsize=12, y=0.98,
    )
    ax.set_title(
        f"{cfg['sim_config']['rules']}   ·   pooled edge {anc['pooled_pct']:+.3f}% "
        f"vs {anc['flat_bet_anchor_pct']:+.2f}% anchor   ·   error bars = 95% CI",
        fontsize=8.5, color="0.35",
    )

    # one combined legend (edge markers, Kelly line, low-n band)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_edge_run(RUNS_DIR)
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    out = render(record, run_dir / "edge_by_count.png")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
