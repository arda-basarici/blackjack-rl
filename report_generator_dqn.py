"""From Table to Network — Phase-3 DQN follow-up report (PDF).

Companion to ``report_generator.py`` (the tabular-audit report): same house style, same
reportlab + matplotlib stack, same cover/section/box vocabulary. This one tells the third movement
of the blackjack project — replacing the lookup table with a DQN and watching the failure mode change.

Numbers policy (read this before editing a figure):
  * The headline SCOREBOARD numbers (agreement %, edge %) are the audited values rendered by the
    chapter notebooks (analysis/ch1_result.ipynb, ch6_complete_game.ipynb) and the millions-of-hands
    re-evaluation in reeval_results.json. They are pinned here as sourced constants (see DATA below)
    so the report can never silently drift from the notebooks.
  * The illustrative FIGURES (Q-trajectories, instability map, encoding grids, peak-vs-back-half,
    capacity, game comparison) are computed LIVE from the saved run records in runs/ — single-run
    mechanism plots, honest to the data on disk.

Run from the repo root:  python report_generator_dqn.py
Needs: reportlab, seaborn, matplotlib, pandas, numpy (no torch — nothing is re-trained or re-evaluated).
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, HRFlowable, 
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir())
RUNS_DIR = ROOT / "runs"
OUTPUT_PDF = str(ROOT / "from-table-to-network.pdf")
CHARTS_DIR = ROOT / "report_charts_dqn"
CHARTS_DIR.mkdir(exist_ok=True)

# -- house palette (identical to report_generator.py) --------------------------
C_DARK   = HexColor("#1A1A2E")
C_ACCENT = HexColor("#0D47A1")
C_INK    = HexColor("#212121")
C_RED    = HexColor("#B71C1C")
C_PANEL  = HexColor("#EEF2F7")
GREEN, ORANGE, RED, GRAY, BLUE = "#2E7D32", "#FB8C00", "#B71C1C", "#BDBDBD", "#0D47A1"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

# -- canonical runs (verified against the run inventory + the chapter scoreboards) -------------------
RUN = {
    "tab_trim":   "20260617-152831_seed42_c298e9c",  # tabular, trimmed (no split): scoreboard 92.8% / 0.99
    "naive":      "20260619-224926_seed42_bf0dd17",  # naive DQN onehot[64,64] const 2M: soft20 sigma~1.14
    "best_trim":  "20260622-015812_seed42_afbf4c5",  # onehot[64,64] harmonic 5M dd stand: diff 92.9, back-half 91.1, edge 1.082
    "scalar_match":"20260621-204934_seed42_afbf4c5", # scalar[64,64] harmonic 5M dd stand (matched): 87.5, peak 93.3, back-half 88.5
    "tab_split":  "20260618-001215_seed42_a642d60",  # tabular, split game: 94.0 / 0.58
    "dqn_full16": "20260622-144112_seed42_050857d",  # onehot[16,16] split+surrender 5M: 83.0 / 1.14
    "dqn_full64": "20260623-173931_seed42_050857d",  # onehot[64,64] split+surrender 5M: edge 1.165 (capacity doesn't help)
    "cap8":       "20260622-024924_seed42_afbf4c5",  # onehot[8,8]  harmonic 1.5M dd stand
    "cap16":      "20260622-031739_seed42_afbf4c5",  # onehot[16,16] harmonic 1.5M dd stand
    "cap64":      "20260621-215815_seed42_afbf4c5",  # onehot[64,64] harmonic 1.5M dd stand
    # matched LR pair — same scalar[64,64] 1.5M Double-DQN stand seed42, differ ONLY in lr schedule
    "lr_const":   "20260620-223437_seed42_48db1fa",  # constant lr  (soft20 double std 0.32)
    "lr_decay":   "20260621-172210_seed42_5f41c7a",  # harmonic decay (soft20 double std 0.16)
}

# Pinned, audited numbers (source noted). Agreements are the notebook scoreboard values; edges are the
# millions-of-hands re-evaluation (reeval_results.json, 5M hands, seed 0, SE ~0.05%/hand).
DATA = dict(
    # trimmed (no-split) scoreboard — ch1_result.ipynb
    trim_tab_agree=92.8, trim_tab_edge=0.99,
    trim_naive_agree=82.1, trim_naive_edge=1.94,
    trim_best_agree=91.1, trim_best_edge=1.08,
    trim_optimum=1.11,                       # no-split basic, literature row 6 (OPTIMUM_PCT['trimmed'])
    # complete (split+surrender) scoreboard — ch6_complete_game.ipynb
    full_tab_agree=94.0, full_tab_edge=0.58,
    full_dqn_agree=83.0, full_dqn_edge=1.14,
    full_dqn64_edge=1.165,                   # [64,64], capacity-doesn't-help
    full_optimum=0.54,                       # full basic, literature row 2 (OPTIMUM_PCT['complete'])
    # re-evaluated basic (in-harness, seed-0, 5M) — shows the seed-0 low bias vs the literature optimum
    basic_full_meas=0.331, basic_full_surr_meas=0.260,
    grid_trim=240, grid_full=340,            # +100 pair cells; surrender adds an action, not cells
)


# ============================ data layer (records on disk) ============================
def record(run: str) -> dict:
    return json.loads((RUNS_DIR / run / "record.json").read_text(encoding="utf-8"))

def curve(run: str) -> list[dict]:
    return record(run).get("learning_curve", [])

def cells(run: str) -> list[dict]:
    return record(run)["diff"]["cells"]

def diff_agreement(run: str) -> float:
    return record(run)["diff"]["agreement_unweighted"] * 100

def peak_backhalf(run: str) -> tuple[float, float]:
    ag = [cp["agreement"] for cp in curve(run) if "agreement" in cp]
    return (max(ag) * 100, st.mean(ag[len(ag) // 2:]) * 100) if ag else (float("nan"), float("nan"))

def probe_series(run: str, cell: str, action: str):
    lc = curve(run)
    eps = [cp["episode"] for cp in lc if cp.get("probe_q") and cell in cp["probe_q"]]
    val = [cp["probe_q"][cell][action] for cp in lc if cp.get("probe_q") and cell in cp["probe_q"]]
    return eps, val

def probe_backhalf_std(run: str, cell: str, action: str):
    _, v = probe_series(run, cell, action)
    return st.pstdev(v[len(v) // 2:]) if len(v) >= 2 else float("nan")

def double_class(c: dict) -> str:
    """Classify a non-pair cell by how its action relates to the basic double decision."""
    if c["agent_action"] == c["basic_action"]:
        return "agree"
    if c["agent_action"] == "double":
        return "over"     # doubles where basic would not
    if c["basic_action"] == "double":
        return "under"    # fails to double where basic would
    return "other"

def double_counts(run: str) -> dict:
    out = {"agree": 0, "over": 0, "under": 0, "other": 0}
    for c in cells(run):
        if c.get("can_split"):
            continue
        out[double_class(c)] += 1
    return out


# ============================ figures (live, from records) ============================
def save(fig, name):
    p = CHARTS_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(p)

def chart_timeline():
    fig, ax = plt.subplots(figsize=(9.2, 1.9))
    ax.axis("off")
    stages = [("Monte Carlo\nsimulator", "#90A4AE"), ("Tabular RL\npolicy audit", "#5C6BC0"),
              ("DQN / function\napproximation", "#0D47A1"),
              ("Next: counting,\nbankroll, risk", "#B0BEC5")]
    x = 0.04
    for i, (label, col) in enumerate(stages):
        cur = (i == 2)
        ax.add_patch(plt.Rectangle((x, 0.30), 0.19, 0.42, color=col, ec="white", lw=2,
                                   alpha=1.0 if cur else 0.85))
        ax.text(x + 0.095, 0.51, label, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold" if cur else "normal")
        if cur:
            ax.text(x + 0.095, 0.16, "this report", ha="center", va="center",
                    fontsize=8.5, color="#0D47A1", style="italic")
        if i < 3:
            ax.annotate("", xy=(x + 0.235, 0.51), xytext=(x + 0.19, 0.51),
                        arrowprops=dict(arrowstyle="-|>", color="#607D8B", lw=1.6))
        x += 0.235
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return save(fig, "timeline")

def chart_qtraj():
    """Naive run: Q(double) swings while Q(stand) stays calm — soft-20 vs dealer-8."""
    eps, qd = probe_series(RUN["naive"], "soft20_v8", "double")
    _, qs = probe_series(RUN["naive"], "soft20_v8", "stand")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(eps, qd, color=RED, lw=1.4, label="Q(double)")
    ax.plot(eps, qs, color=BLUE, lw=1.8, label="Q(stand)  — basic strategy's choice")
    ax.set_xlabel("training episodes"); ax.set_ylabel("estimated value Q")
    ax.set_title("The value behind a correct action can still be unstable\n"
                 "soft 20 vs dealer 8 — Q(double) keeps swinging; Q(stand) is settled",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    sd = probe_backhalf_std(RUN["naive"], "soft20_v8", "double")
    ss = probe_backhalf_std(RUN["naive"], "soft20_v8", "stand")
    ax.text(0.02, 0.04, f"back-half std:  double {sd:.2f}   vs   stand {ss:.2f}",
            transform=ax.transAxes, fontsize=9, color="#455A64", style="italic")
    return save(fig, "qtraj")

def chart_instability():
    """2x2 back-half std of Q: rows hard/soft, cols double/stand — faithful to the notebook's grid.
    The point lives in the contrast: Q(double) wobbles where doubling is marginal; Q(stand) is calm everywhere."""
    lc = curve(RUN["naive"])

    def matrix(prefix, action):
        pat = re.compile(prefix + r"(\d+)_v(\d+)")
        series: dict[tuple, list] = {}
        for cp in lc:
            for k, v in (cp.get("q_grid") or {}).items():
                m = pat.fullmatch(k)
                if m and action in v:
                    series.setdefault((int(m.group(1)), int(m.group(2))), []).append(v[action])
        totals = sorted({t for t, _ in series}); ups = sorted({u for _, u in series})
        M = np.full((len(totals), len(ups)), np.nan)
        for (t, u), vals in series.items():
            if len(vals) >= 2:
                M[totals.index(t), ups.index(u)] = st.pstdev(vals[len(vals) // 2:])
        return totals, ups, M

    panels = {(p, a): matrix(p, a) for p in ("hard", "soft") for a in ("double", "stand")}
    if panels[("hard", "double")][2].size == 0:
        return None
    vmax = float(np.nanmax([np.nanmax(M) for _, _, M in panels.values() if M.size]))
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.4))
    im = None
    for (prefix, row) in (("hard", 0), ("soft", 1)):
        for (action, col) in (("double", 0), ("stand", 1)):
            ax = axes[row, col]
            totals, ups, M = panels[(prefix, action)]
            im = ax.imshow(M, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(ups))); ax.set_xticklabels([("A" if u == 11 else u) for u in ups], fontsize=8)
            ax.set_yticks(range(len(totals))); ax.set_yticklabels(totals, fontsize=7)
            ax.grid(False)
            if row == 0:
                ax.set_title(f"Q({action}) — back-half std", fontsize=10.5, fontweight="bold")
            if row == 1:
                ax.set_xlabel("dealer upcard", fontsize=9)
        axes[row, 0].set_ylabel(f"{prefix} total", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.045, pad=0.03,
                 label="back-half std of Q (training-end wobble)")
    fig.suptitle("Instability is concentrated on the double, not on stand", fontweight="bold", fontsize=12, y=0.97)
    return save(fig, "instability")

def chart_lr():
    """Left: a matched pair (same config, only the schedule differs) — Q(double) over training, the mechanism.
    Right: the population — back-half std across every logged run, constant vs decaying."""
    const, decay = [], []
    for d in sorted(os.listdir(RUNS_DIR)):
        rec_p = RUNS_DIR / d / "record.json"
        if not rec_p.exists():
            continue
        try:
            r = json.loads(rec_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        c = r.get("config") or {}
        if "encoding" not in c:
            continue
        s = probe_backhalf_std(d, "soft20_v8", "double")
        if not np.isfinite(s):
            continue
        (const if (c.get("lr_schedule") or "constant") == "constant" else decay).append(s)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw={"width_ratios": [1.45, 1]})
    # left — matched trajectory (same scalar[64,64] 1.5M Double-DQN stand seed42; only the schedule differs)
    for run, lab, col in [(RUN["lr_const"], "constant lr", "#B71C1C"),
                          (RUN["lr_decay"], "decaying lr (harmonic)", "#2E7D32")]:
        eps, val = probe_series(run, "soft20_v8", "double")
        a0.plot(eps, val, color=col, lw=1.1, label=lab)
    a0.set_xlabel("training episode"); a0.set_ylabel("Q(double | soft 20 v 8)")
    a0.set_title("matched pair — constant keeps swinging, decay settles", fontsize=9.5, fontweight="bold")
    a0.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    # right — population over every logged run
    data = [const, decay]
    labels = [f"constant\n(n={len(const)})", f"decaying\n(n={len(decay)})"]
    try:                                  # 'labels' renamed to 'tick_labels' in matplotlib 3.9
        parts = a1.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    except TypeError:
        parts = a1.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showfliers=False)
    for patch, col in zip(parts["boxes"], [RED, GREEN]):
        patch.set_facecolor(col); patch.set_alpha(0.35)
    np.random.seed(0)
    for i, arr in enumerate(data, 1):
        a1.scatter(np.random.normal(i, 0.05, len(arr)), arr, s=14, color=["#B71C1C", "#2E7D32"][i - 1],
                   alpha=0.6, zorder=3)
        a1.text(i, max(arr) + 0.04, f"median {np.median(arr):.2f}", ha="center", fontsize=9,
                fontweight="bold", color="#455A64")
    a1.set_ylabel("back-half std of Q(double | soft 20 v 8)", fontsize=9)
    a1.set_title("every logged run", fontsize=9.5, fontweight="bold")
    fig.suptitle("A decaying step size settles the high-variance double", fontweight="bold", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, "lr")

# double-focused strategy grid — faithful to ch4_representation.ipynb (dcat/dpanel/dgrid):
# only correct DOUBLES are green; over-doubles red, under-doubles orange, everything else (hit/stand) gray.
ACT = {"hit": "H", "stand": "S", "double": "D", "split": "P", "surrender": "R"}
DCOL = ["#bfe6b6", "#e24b4a", "#f4a259", "#e7e7e7"]   # 0 correct-double, 1 over, 2 under, 3 other

def _dcat(c):
    a, b = c["agent_action"], c["basic_action"]
    if b == "double" and a == "double":
        return 0
    if b != "double" and a == "double":
        return 1   # over-double
    if b == "double" and a != "double":
        return 2   # under-double
    return 3       # hit/stand region

def genuine_over_under(run):
    """Over/under-doubles counted only over GENUINE disagreements — the notebook's figure (scalar 21/0, one-hot 9/3)."""
    dis = [c for c in cells(run) if c["category"] == "genuine_disagreement" and not c.get("can_split")]
    o = sum(1 for c in dis if c["agent_action"] == "double" and c["basic_action"] != "double")
    u = sum(1 for c in dis if c["basic_action"] == "double" and c["agent_action"] != "double")
    return o, u

def _dpanel(ax, clist, is_soft, title):
    from matplotlib.colors import ListedColormap
    up = list(range(2, 12))
    vals = sorted({c["player_value"] for c in clist
                   if bool(c["is_soft"]) == is_soft and not c.get("can_split")}, reverse=True)
    G = np.full((len(vals), len(up)), np.nan)
    for c in clist:
        if bool(c["is_soft"]) != is_soft or c.get("can_split"):
            continue
        if c["player_value"] not in vals or c["dealer_upcard"] not in up:
            continue
        i, j = vals.index(c["player_value"]), up.index(c["dealer_upcard"])
        G[i, j] = _dcat(c)
        lab = ACT.get(c["agent_action"], "?")
        if c["agent_action"] != c["basic_action"]:
            lab = "%s→%s" % (ACT.get(c["agent_action"], "?"), ACT.get(c["basic_action"], "?"))
        ax.text(j, i, lab, ha="center", va="center", fontsize=5.5)
    ax.imshow(G, cmap=ListedColormap(DCOL), vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(up))); ax.set_xticklabels([str(u) if u < 11 else "A" for u in up], fontsize=7)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(vals, fontsize=6.5)
    ax.set_xlabel("dealer upcard", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(False)   # drop the seaborn whitegrid overlay — it doesn't align with the imshow cells

def chart_encoding():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    for row, (run, enc) in enumerate([(RUN["scalar_match"], "scalar (smooth)"),
                                      (RUN["best_trim"], "one-hot (sharp)")]):
        cl = cells(run); o, u = genuine_over_under(run); ag = diff_agreement(run)
        _dpanel(axes[row, 0], cl, False, f"{enc} · hard totals   ({ag:.1f}% agree · {o} over / {u} under)")
        _dpanel(axes[row, 1], cl, True, f"{enc} · soft totals")
        axes[row, 0].set_ylabel("player total", fontsize=8)
    handles = [Patch(facecolor=DCOL[0], label="correct double"),
               Patch(facecolor=DCOL[1], label="over-double (boundary smears out)"),
               Patch(facecolor=DCOL[2], label="under-double"),
               Patch(facecolor=DCOL[3], label="other (hit / stand)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Encoding changes the geometry the network can draw\n"
                 "cells carry the agent's action (agent→basic where they differ)",
                 fontweight="bold", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return save(fig, "encoding")

def chart_crossover():
    """Agreement over training, matched scalar vs one-hot. Scalar leads while only hit/stand exists
    (means: 79.3 vs 76.6 pre-double); one-hot overtakes once the double is introduced (91.0 vs 88.5
    post-2M) — the double is the only place encoding bites. Faint raw + a moving-average trend."""
    def smooth(e, y, w=21):
        if len(y) < w:
            return np.array(e), np.array(y)
        ys = np.convolve(np.array(y), np.ones(w) / w, mode="valid")
        return np.array(e)[w // 2: w // 2 + len(ys)], ys
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for run, lab, col in [(RUN["best_trim"], "one-hot (sharp boundary)", "#D85A30"),
                          (RUN["scalar_match"], "scalar (ordinal / smooth)", "#378ADD")]:
        pts = [(cp["episode"], cp["agreement"] * 100) for cp in curve(run) if cp.get("agreement") is not None]
        e = [p[0] for p in pts]; a = [p[1] for p in pts]
        ax.plot(e, a, color=col, lw=0.6, alpha=0.20)              # raw
        es, ys = smooth(e, a); ax.plot(es, ys, color=col, lw=2.2, label=lab)   # trend
    ax.axvline(500_000, ls="--", color="#999", lw=1)
    ax.text(560_000, ax.get_ylim()[0] + 1.5, "double introduced", fontsize=8, color="#666")
    ax.set_xlabel("training episode"); ax.set_ylabel("agreement (%)")
    ax.set_title("The gap opens at the double: scalar leads on hit/stand, one-hot pulls ahead once the double enters",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return save(fig, "crossover")

def chart_embedding():
    """The geometry of what the net learned — mirrors ch4 §4.2. Reload each net, take the penultimate-layer
    activation for every cell, and project. Rows: ACTION via PCA (linear axes preserve the global
    hit->double->stand ordering); SOFT/HARD via t-SNE; DEALER UPCARD via t-SNE (smooth strength gradient).
    Columns: one-hot | scalar. Needs torch + sklearn; returns None (figure skipped) if absent, so the build
    never breaks."""
    try:
        from blackjack_rl.dqn.embedding import load_agent, cell_embeddings
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except Exception as e:
        print("  [embedding figure skipped — needs torch + sklearn:", type(e).__name__, str(e)[:60], "]")
        return None
    def embed(run):
        ce = cell_embeddings(load_agent(str(RUNS_DIR / run)))
        X = np.array(ce.embeddings)
        return (ce.cells,
                PCA(n_components=2).fit_transform(X),
                TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X))
    cols = [(RUN["best_trim"], "one-hot"), (RUN["scalar_match"], "scalar")]
    data = {lab: embed(run) for run, lab in cols}
    ACOL = {"hit": "#378ADD", "stand": "#1D9E75", "double": "#D85A30"}
    SOFTC = {"soft": "#7F77DD", "hard": "#9AA0A6"}
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 12.4))
    sc = None
    for col, (run, lab) in enumerate(cols):
        cl, P, T = data[lab]
        # row 0: action via PCA
        a = axes[0, col]
        for act, c in ACOL.items():
            m = [i for i, cc in enumerate(cl) if cc["action"] == act]
            a.scatter(P[m, 0], P[m, 1], s=16, c=c, label=act, edgecolor="none")
        a.set_title(f"{lab} — action (PCA)", fontsize=10, fontweight="bold")
        if col == 0:
            a.legend(fontsize=7.5, framealpha=0.9, loc="best")
        # row 1: soft / hard via t-SNE
        b = axes[1, col]
        for k, c in SOFTC.items():
            m = [i for i, cc in enumerate(cl) if ("soft" if cc["is_soft"] else "hard") == k]
            b.scatter(T[m, 0], T[m, 1], s=16, c=c, label=k, edgecolor="none")
        b.set_title(f"{lab} — soft / hard (t-SNE)", fontsize=10, fontweight="bold")
        if col == 0:
            b.legend(fontsize=7.5, framealpha=0.9, loc="best")
        # row 2: dealer upcard via t-SNE
        d = axes[2, col]
        sc = d.scatter(T[:, 0], T[:, 1], s=16, c=[cc["dealer_upcard"] for cc in cl], cmap="coolwarm")
        d.set_title(f"{lab} — dealer upcard (t-SNE)", fontsize=10, fontweight="bold")
        for ax in (a, b, d):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes[2, :].tolist(), fraction=0.045, pad=0.03, label="dealer upcard (2 → A)")
    fig.suptitle("The net lays the structure out cleanly — action regions, soft/hard, and dealer-strength ordering",
                 fontweight="bold", fontsize=11, y=0.995)
    return save(fig, "embedding")

def chart_peak_backhalf():
    runs = [("naive\nonehot 2M", RUN["naive"]), ("scalar\n[64,64] 5M", RUN["scalar_match"]),
            ("one-hot\n[16,16] 5M", RUN["cap16"]), ("best\nonehot 5M", RUN["best_trim"])]
    peaks, bhs, labels = [], [], []
    for lab, r in runs:
        pk, bh = peak_backhalf(r)
        peaks.append(pk); bhs.append(bh); labels.append(lab)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w / 2, peaks, w, color="#90CAF9", label="peak checkpoint")
    ax.bar(x + w / 2, bhs, w, color="#0D47A1", label="back-half mean (reported)")
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, peaks[i] + 0.3, f"{peaks[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, bhs[i] + 0.3, f"{bhs[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("agreement (%)"); ax.set_ylim(70, 100)
    ax.set_title("The best checkpoint is not the stable policy", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    return save(fig, "peakbh")

def chart_capacity():
    runs = [("[8, 8]", RUN["cap8"]), ("[16, 16]", RUN["cap16"]), ("[64, 64]", RUN["cap64"])]
    vals = [diff_agreement(r) for _, r in runs]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.bar([l for l, _ in runs], vals, color=["#90CAF9", "#42A5F5", "#0D47A1"], edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(80, 95); ax.set_ylabel("agreement (%)")
    ax.set_title("More width is not the missing ingredient\n(matched one-hot, harmonic schedule)",
                 fontweight="bold", fontsize=11)
    return save(fig, "capacity")

def chart_gamegap():
    """The representation cost grows with the action set: trimmed vs complete, tabular vs DQN."""
    labels = ["trimmed\n(no split)", "complete\n(+split +surrender)"]
    tab = [DATA["trim_tab_agree"], DATA["full_tab_agree"]]
    dqn = [DATA["trim_best_agree"], DATA["full_dqn_agree"]]
    x = np.arange(2); w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    a1.bar(x - w / 2, tab, w, color="#2E7D32", label="tabular")
    a1.bar(x + w / 2, dqn, w, color="#0D47A1", label="DQN")
    for i in range(2):
        a1.text(x[i] - w / 2, tab[i] + 0.4, f"{tab[i]:.0f}", ha="center", fontsize=8.5)
        a1.text(x[i] + w / 2, dqn[i] + 0.4, f"{dqn[i]:.0f}", ha="center", fontsize=8.5)
        a1.annotate(f"gap {tab[i]-dqn[i]:.0f} pt", (x[i], min(tab[i], dqn[i]) - 4), ha="center",
                    fontsize=8, color="#B71C1C")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8.5); a1.set_ylim(70, 100)
    a1.set_ylabel("agreement (%)"); a1.set_title("agreement", fontsize=10, fontweight="bold")
    a1.legend(fontsize=8.5, loc="lower left")
    tab_e = [DATA["trim_tab_edge"], DATA["full_tab_edge"]]
    dqn_e = [DATA["trim_best_edge"], DATA["full_dqn_edge"]]
    a2.bar(x - w / 2, tab_e, w, color="#2E7D32"); a2.bar(x + w / 2, dqn_e, w, color="#0D47A1")
    for i in range(2):
        a2.text(x[i] - w / 2, tab_e[i] + 0.02, f"{tab_e[i]:.2f}", ha="center", fontsize=8.5)
        a2.text(x[i] + w / 2, dqn_e[i] + 0.02, f"{dqn_e[i]:.2f}", ha="center", fontsize=8.5)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8.5)
    a2.set_ylabel("house edge (% / hand, lower better)")
    a2.set_title("edge (band, not ranking)", fontsize=10, fontweight="bold")
    fig.suptitle("The gap is small in the trimmed game and widens when the action set adds walls",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, "gamegap")

def build_charts():
    return {
        "timeline": chart_timeline(),
        "qtraj": chart_qtraj(),
        "instability": chart_instability(),
        "lr": chart_lr(),
        "encoding": chart_encoding(),
        "crossover": chart_crossover(),
        "embedding": chart_embedding(),
        "peakbh": chart_peak_backhalf(),
        "capacity": chart_capacity(),
        "gamegap": chart_gamegap(),
    }


# ============================ PDF ============================
def build_pdf(C):
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    W = A4[0] - 40 * mm

    title = ParagraphStyle("T", fontSize=18, leading=23, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=8)
    sub   = ParagraphStyle("Sub", fontSize=12.5, leading=17, textColor=HexColor("#BBDEFB"), alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
    meta  = ParagraphStyle("Meta", fontSize=10.5, leading=15, textColor=HexColor("#90CAF9"), alignment=TA_CENTER, fontName="Helvetica")
    h1    = ParagraphStyle("H1", fontSize=15, leading=19, textColor=C_ACCENT, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)
    body  = ParagraphStyle("B", fontSize=10, leading=15, textColor=C_INK, alignment=TA_JUSTIFY, fontName="Helvetica", spaceAfter=8, firstLineIndent=16)
    lead  = ParagraphStyle("Lead", fontSize=11.5, leading=17, textColor=C_INK, alignment=TA_JUSTIFY, fontName="Helvetica", spaceAfter=9, firstLineIndent=16)
    cap   = ParagraphStyle("Cap", fontSize=8, leading=12, textColor=HexColor("#757575"), alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceAfter=10)
    take  = ParagraphStyle("Take", fontSize=10.5, leading=14, textColor=HexColor("#455A64"), fontName="Helvetica-Oblique", spaceBefore=1, spaceAfter=9)
    boxh  = ParagraphStyle("BoxH", fontSize=12, leading=16, textColor=C_ACCENT, fontName="Helvetica-Bold", spaceAfter=5)
    boxb  = ParagraphStyle("BoxB", fontSize=9.5, leading=14, textColor=C_INK, fontName="Helvetica", spaceAfter=4)
    th    = ParagraphStyle("th", fontSize=9.5, leading=12, textColor=white, fontName="Helvetica-Bold")
    tc    = ParagraphStyle("tc", fontSize=9, leading=12, textColor=C_INK, fontName="Helvetica")
    tcb   = ParagraphStyle("tcb", fontSize=9, leading=12, textColor=C_INK, fontName="Helvetica-Bold")

    story = []
    from reportlab.lib.utils import ImageReader

    def P(t, s=body): story.append(Paragraph(t, s))
    def fig(key, caption, width=None):
        if C.get(key) is None:
            return
        width = width or W * 0.84
        iw, ih = ImageReader(C[key]).getSize()
        story.append(Image(C[key], width=width, height=width * ih / iw, hAlign="CENTER"))
        P(caption, cap)
    def rule(): story.append(HRFlowable(width=W, thickness=1, color=C_ACCENT, spaceAfter=10))
    def panel(flow, bg=C_DARK, height=None, pad=20):
        kw = {"colWidths": [W]}
        if height: kw["rowHeights"] = [height]
        t = Table([[flow]], **kw)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 26), ("RIGHTPADDING", (0, 0), (-1, -1), 26),
            ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
        return t
    def styled_table(data, widths, hl_rows=()):
        t = Table(data, colWidths=widths)
        sty = [("BACKGROUND", (0, 0), (-1, 0), C_DARK),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F5F5F5"), white]),
               ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")), ("TOPPADDING", (0, 0), (-1, -1), 5),
               ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 8),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        for r in hl_rows:
            sty.append(("BACKGROUND", (0, r), (-1, r), HexColor("#E3F2FD")))
        t.setStyle(TableStyle(sty))
        return t
    def thr(cellsrow): return [Paragraph(x, th) for x in cellsrow]
    def tr(cellsrow, bold0=True):
        return [Paragraph(x, tcb if (i == 0 and bold0) else tc) for i, x in enumerate(cellsrow)]

    # ----- COVER -----
    story.append(Spacer(1, 42 * mm))
    cover = [Paragraph("From Table to Network", title),
             Spacer(1, 2 * mm),
             Paragraph("Testing what neural function approximation adds &mdash;", sub),
             Paragraph("and destabilizes &mdash; after a tabular policy audit", sub),
             Spacer(1, 8 * mm),
             Paragraph("A DQN vs. lookup-table study on blackjack, audited cell-by-cell against basic strategy", meta),
             Spacer(1, 2 * mm),
             Paragraph("Part of AI Journey &mdash; Phase 3: Deep Learning &amp; Reinforcement Learning", meta),
             Spacer(1, 10 * mm),
             Paragraph("github.com/arda-basarici/ai-journey",
                       ParagraphStyle("L", fontSize=10.5, textColor=HexColor("#64B5F6"), alignment=TA_CENTER, fontName="Helvetica"))]
    story.append(panel(cover, height=118 * mm)); story.append(PageBreak())

    # ----- OPENING -----
    P("The third movement of the project", h1); rule()
    P("This report is the third step in a longer blackjack investigation. The first built a Monte "
      "Carlo simulator and treated blackjack as a controlled system: fixed rules, reproducible seeds, "
      "pluggable strategies, millions of hands. The second turned that simulator into a reinforcement-"
      "learning environment, where a tabular Monte Carlo agent learned from terminal outcomes and was "
      "audited cell-by-cell against basic strategy. The interesting result there was not that the agent "
      "learned a strong policy &mdash; it was <i>where</i> it fell short: most error lived in rare, "
      "coverage-starved regions. The table mastered common decisions and struggled where its own play "
      "gave it too little evidence.", lead)
    P("That sets up the question this report asks. If the table&rsquo;s weakness was coverage, what "
      "happens when the lookup table is replaced by a neural network? A table learns each cell "
      "separately; a network shares parameters and generalizes. Generalization might help &mdash; it "
      "produces an answer even for cells natural play visits rarely. But it has a cost: the network "
      "represents the grid through shared parameters rather than independent cell values, and blackjack&rsquo;s decision boundaries are not always "
      "smooth. So the central question is not &lsquo;can a DQN play blackjack?&rsquo; It can. The better "
      "question is:", body)
    
    P("<b>Can neural generalization repair the tabular learner&rsquo;s coverage problem, and what does "
      "it distort in exchange?</b>", take)
    story.append(Spacer(1, 15 * mm))
    fig("timeline", "The same blackjack platform, three movements. The simulator came first, the tabular "
        "learner exposed the coverage problem, and the DQN tests whether generalization changes that failure mode.")
    
    # ----- 1. why blackjack + two metrics -----
    story.append(PageBreak())
    P("1. The instrument, and the two-metric lens", h1); rule()
    P("Blackjack is a useful RL sandbox for one specific reason: in the controlled single-hand setting "
      "there is a <i>known reference policy</i>. Basic strategy gives the value-maximizing action for "
      "each total, soft/hard status, and dealer upcard under a fixed rule set. Most ML problems lack a "
      "trustworthy answer key; here the table is a ruler, so a learned policy can be checked cell by "
      "cell. This is not a study of casino blackjack &mdash; counting, bet sizing, and bankroll belong "
      "to the next stage. The cards are the surface; the deeper question is representation. A table "
      "represents the strategy grid directly, one value per cell; a network represents it through a "
      "shared function that generalizes but couples decisions together. The known table lets us measure "
      "that tradeoff rather than guess at it.", body)
    P("Two metrics run through the whole report, and they must be read differently. <b>Agreement</b> is "
      "how often the learned policy picks the same action as basic strategy across the grid &mdash; an "
      "exact, deterministic, policy-to-policy comparison, and the load-bearing metric here. <b>Edge</b> "
      "is the house edge measured by simulating hands: it answers how much money the policy loses, but "
      "it is noisy and weights mistakes by how often and how expensively they occur. Agreement tells us "
      "how closely the agent recovered the table; edge tells us how much the differences mattered. "
      "Neither alone is enough.", body)
    P("In the trimmed no-split game, the tabular Q-learner reaches about 92.8% agreement; the naive DQN "
      "starts near 82.1%; the best no-split DQN reaches 91.1%. The careful reading is therefore not "
      "&lsquo;DQN fails.&rsquo; It is close to the table &mdash; but it takes a full stack of "
      "stabilizers to get there, while the table is simple and direct.", body)

    sb = [thr(["method", "agreement", "edge (%/hand)", "tuning"]),
          tr(["tabular Q-learner", f"{DATA['trim_tab_agree']}%", f"{DATA['trim_tab_edge']:.2f}", "none"]),
          tr(["naive DQN", f"{DATA['trim_naive_agree']}%", f"{DATA['trim_naive_edge']:.2f}", "defaults"]),
          tr(["best DQN", f"{DATA['trim_best_agree']}%", f"{DATA['trim_best_edge']:.2f}", "full stack"]),
          tr(["basic strategy (no-split optimum)", "&mdash;", f"{DATA['trim_optimum']:.2f}", "&mdash;"])]
    story.append(Spacer(1, 15 * mm))
    story.append(styled_table(sb, [W * 0.40, W * 0.18, W * 0.24, W * 0.18], hl_rows=(3,)))
    P("Table 1: Trimmed-game scoreboard. Agreement (exact match on the common 240-cell grid) separates the "
      "policies; edge is read as a noisy band, not a precise ranking. See the edge-methodology note below.", cap)
    story.append(Spacer(1, 15 * mm))
    edgebox = [Paragraph("How to read the edge column", boxh),
               Paragraph("The learner edges are millions-of-hands re-evaluations (5M hands, seed 0, "
                         "&plusmn;0.05%); the optimum row is the seed-independent literature value for no-split "
                         "basic. The single evaluation seed draws low &mdash; the same audit measured a 200k "
                         "basic sample at 0.09% and a 5M basic full-game sample at 0.33% against a ~0.5% "
                         "literature figure &mdash; so the agents&rsquo; absolute edges sit just below the "
                         "literature optimum as an evaluation artifact, not by beating optimal play. The honest "
                         "reading is &lsquo;all close, near the no-split ceiling&rsquo;; agreement does the separating.", boxb)]
    story.append(panel(edgebox, bg=C_PANEL, pad=12))

    # ----- 2. tabular baseline + why DQN -----
    story.append(PageBreak())
    P("2. The tabular baseline, and why try a network at all", h1); rule()
    P("The tabular learner is the natural baseline because the problem itself is tabular. For the "
      "total-based abstraction the state space is small and the reference strategy is a grid; a lookup "
      "table matches that structure exactly. Each value is stored, inspected, and compared "
      "independently. The table is not more intelligent &mdash; it is better matched to the shape of the "
      "problem. It wins on exactness, inspectability, and simplicity. The DQN has to justify its extra complexity by "
      "doing something the table cannot: generalizing across cells.", body)
    P("So the reason to try a DQN is not that the problem needs one. It is that the network changes the "
      "representation. The tabular audit found a coverage story &mdash; the learner struggled where "
      "natural play gave it too little evidence &mdash; and a network might answer rare cells better "
      "because it does not learn each from scratch. That is the possible upside. The cost is that "
      "blackjack&rsquo;s optimal policy is not always smooth: a total one point higher, or a dealer "
      "card one rank stronger, can flip the correct action. Doubling, splitting and surrender are "
      "narrow bands with hard edges, and a network &mdash; especially with scalar inputs &mdash; tends "
      "to interpolate, which can smear a sharp boundary. The DQN study is a tradeoff test: <b>does "
      "generalization fill the holes left by tabular coverage, or blur the edges basic strategy "
      "depends on?</b>", body)
    P("One honesty note on the comparison: moving from tabular Monte Carlo to a DQN changes <i>both</i> the "
      "representation and the update rule (bootstrapped temporal-difference learning rather than "
      "terminal-return averaging). Because blackjack hands are short and dominated by the terminal "
      "reward, this report treats the network representation as the main object of study, rather than "
      "presenting bootstrapping itself as the improvement.", body)
    story.append(Spacer(1, 15 * mm))
    auditbox = [Paragraph("How the DQN was audited", boxh),
                Paragraph("The tabular audit reads stored Q-values and training visit counts directly. The "
                          "DQN has no such table, so the trained network is queried on every strategy cell "
                          "and its greedy action is materialized onto the same audit grid &mdash; making the "
                          "two methods comparable cell-for-cell against basic strategy.", boxb)]
    story.append(panel(auditbox, bg=C_PANEL, pad=12))

    # ----- 3. naive result + wild double -----
    story.append(PageBreak())
    P("3. First result: the naive DQN learns, but its residual has structure", h1); rule()
    P("The naive DQN learns a recognizable policy &mdash; about 82.1% agreement, well behind the table. "
      "The informative question is not how far behind, but <i>where</i> the disagreements sit, and they "
      "are not spread evenly. The early symptom is the <b>double</b> action in boundary cells. Doubling "
      "is a high-variance terminal move: the player commits twice the stake, draws one card, and the "
      "hand resolves &mdash; so its value estimate is noisy. In the soft-20 vs dealer-8 probe cell, the "
      "value of doubling keeps swinging across training while the value of standing is settled. The "
      "agent happens to land on the correct action (stand), but the value beneath the decision is "
      "unstable: a policy can look stable while the estimates underneath are still moving.", body)
    story.append(Spacer(1, 10 * mm))
    fig("qtraj", "Figure 1: One probe cell over training. Q(stand) is calm; Q(double) keeps swinging. The "
        "instability is concentrated on the high-variance terminal action, not on every value.")
    nrun = RUN["naive"]
    def _cell_act(soft, total, up):
        for c in cells(nrun):
            if bool(c["is_soft"]) == soft and c["player_value"] == total and c["dealer_upcard"] == up:
                return c["agent_action"], c["basic_action"]
        return "?", "?"
    probes = [("soft 20 vs 8", True, 20, 8, "unstable value, correct action"),
              ("soft 19 vs 6", True, 19, 6, "stable value, wrong action (over-double)"),
              ("hard 16 vs 10", False, 16, 10, "stable value, correct action")]
    prow = [thr(["probe cell", "std Q(double)", "std Q(stand)", "basic", "agent", "reading"])]
    for lab, soft, tot, up, reading in probes:
        nm = ("soft" if soft else "hard") + str(tot) + "_v" + str(up)
        sd, ss = probe_backhalf_std(nrun, nm, "double"), probe_backhalf_std(nrun, nm, "stand")
        a, b = _cell_act(soft, tot, up)
        prow.append(tr([lab, f"{sd:.2f}", f"{ss:.2f}", b, a, reading]))
    story.append(Spacer(1, 10 * mm))
    story.append(styled_table(prow, [W * 0.16, W * 0.15, W * 0.15, W * 0.12, W * 0.12, W * 0.30]))
    P("Table 2: Three probe cells from the naive run. The instability does not line up with the errors: the "
      "agent is right where its value is wildest (soft-20) and wrong where its value is calm (soft-19, a "
      "stable over-double). That mismatch is why the wild double is a symptom, not the whole explanation.", cap)
    story.append(Spacer(1, 10 * mm))
    P("This is the &lsquo;wild double,&rsquo; and it is a real symptom &mdash; but not the whole "
      "explanation. In some cells the value swings yet the action is right; in others the agent over-"
      "doubles with low variance; in others still it doubles correctly and stably. So instability "
      "around the terminal action is the first visible thing, not the diagnosis. The next step tests "
      "whether fixing the symptom also fixes the policy.", body)
    fig("instability", "Figure 2: Back-half std of Q across the grid. Q(double) (left column) wobbles where doubling "
        "is live and marginal; Q(stand) (right column) is calm almost everywhere &mdash; the instability is specific "
        "to the high-variance terminal action, not to every value. A real symptom, but not yet proof that it is the "
        "main cause of policy disagreement.")


    # ----- 4. controlled tests -----
    story.append(PageBreak())
    P("4. Controlled tests: Useful fixes, but no single explanation", h1); rule()
    P("The investigation then tests the obvious suspects one at a time &mdash; form a hypothesis, change "
      "one thing where possible, read the result against both metrics, and keep the method even when it "
      "does not fully solve the problem. The point is the elimination pattern, not any one run.", body)
    sv = [thr(["suspect", "why plausible", "what it changed"]),
          tr(["learning-rate schedule", "noisy double keeps moving", "decay calms the oscillation; over-decay starves the rest"]),
          tr(["curriculum (mask double early)", "learn the map first", "smoother, more legible training; not a score breakthrough"]),
          tr(["network width / capacity", "maybe underfitting", "tiny nets underfit; bigger nets do not close the gap"]),
          tr(["exploring starts", "the tabular coverage fix", "no clean win &mdash; unlike the table, forced coverage doesn&rsquo;t just fill blank cells; it shifts the training distribution and didn&rsquo;t reliably improve the final policy here"]),
          tr(["reward control variate", "remove dealer variance", "helps only part &mdash; the drawn card&rsquo;s own variance remains"]),
          tr(["Double DQN / SWA", "overestimation / readout noise", "modest; SWA stabilizes the readout, not the cause"])]
    story.append(styled_table(sv, [W * 0.26, W * 0.30, W * 0.44]))
    P("Table 3: Suspects and verdicts. Several interventions help a specific symptom; none alone removes "
      "the residual. Exploring starts is the sharpest contrast with the tabular report &mdash; forced "
      "coverage was the tabular fix, but the network already generalizes to every cell, so extra double-"
      "heavy starts mostly add exposure to the noisy action.", cap)
    fig("lr", "Figure 3: Left &mdash; a matched pair (same config, only the schedule differs): the constant "
        "step keeps Q(double) swinging while the decaying one settles. Right &mdash; across every logged run, "
        "decaying schedules carry far lower back-half std (median 0.16 vs 0.52). A real, useful effect, and "
        "still not the whole residual.")
    fig("capacity", "Figure 4: Matched runs at three widths. Capacity is not the missing ingredient; the "
        "residual survives a 64-wide network.", width=W * 0.45)

    # ----- 5. the reframe -----
    story.append(PageBreak())
    P("5. The important revision: not a hard DQN ceiling", h1); rule()
    P("An earlier reading would have been too strong &mdash; the DQN appeared to plateau far below the "
      "table, suggesting a hard function-approximation floor. The later runs weaken that claim. With a "
      "stronger stack of stabilizers the no-split DQN reaches about 91.1% agreement (back-half mean), "
      "against the table&rsquo;s 92.8%, and on edge it lands within the evaluation band of the no-split "
      "optimum. Individual checkpoints peak higher still &mdash; around 93% &mdash; though those peaks "
      "are transient and seed-sensitive, which is why the report quotes the back-half level, not the "
      "best checkpoint.", body)
    P("So the honest claim is not &lsquo;DQN cannot match the table.&rsquo; It is: <b>for this small, "
      "exactly tabulatable abstraction, the table is simpler and more robust; the DQN can approach it, "
      "but only after stabilizers, careful readout, and attention to seed-sensitive dynamics.</b> The "
      "table gets exactness for free because its representation matches the problem; the network has to "
      "learn a shared approximation of the same map, and its path is less direct and more sensitive to "
      "training dynamics. The DQN is not defeated &mdash; it is expensive, delicate, and informative. "
      "Crucially, this reframe does not erase a representation cost: that cost is small in the trimmed "
      "game and, as the next sections show, grows precisely when the action set adds more hard edges.", body)
    story.append(Spacer(1, 15 * mm))
    fig("peakbh", "Figure 5: Peak vs back-half agreement. Several runs touch a high checkpoint and settle "
        "lower; reporting the peak would overstate the policy the agent reliably holds.")

    # ----- 6. representation -----
    story.append(PageBreak())
    P("6. Representation: where the network still pays", h1); rule()
    P("Once the easy explanations are bounded, the encoding comparison shows that part of the residual is representation-shaped. The table is "
      "hard-edged &mdash; one cell says double, the neighbour says stand, each independent. The network "
      "is shared: it learns a continuous function over inputs, which generalizes but pressures "
      "neighbouring states to behave alike. In these runs, sharp policy boundaries are where the network most visibly pays for its smooth approximation. "
      "With <i>scalar</i> inputs, totals "
      "and upcards are ordered quantities and the smooth net over-extends the double region. With "
      "<i>one-hot</i> inputs they are categorical, and the net is free to place sharper boundaries.", body)
    oh_o, oh_u = genuine_over_under(RUN["best_trim"])
    sc_o, sc_u = genuine_over_under(RUN["scalar_match"])
    P(f"In the matched comparison, one-hot reaches 92.9% agreement with {oh_o} over-doubles and "
      f"{oh_u} under-doubles; scalar reaches 87.5% with {sc_o} over-doubles and "
      f"{sc_u} under. This is not &lsquo;one-hot is always better&rsquo; &mdash; it is that the "
      "encoding should match the structure of the decision. Where adjacent values change meaning "
      "sharply, a sharper encoding helps; the residual lives at the edge of the double region, exactly "
      "where the optimal table has a hard boundary and the smooth network tries to approximate it.", body)
    story.append(Spacer(1, 15 * mm))
    fig("encoding", "Figure 6: Scalar vs one-hot, same architecture and schedule. Scalar smears the double "
        "region outward (more over-doubles); one-hot sharpens the boundary. The residual is partly a "
        "representation choice, not a fixed limit on what the network can learn.")
    story.append(PageBreak())
    P("“The clearest diagnostic evidence that this is specifically about the double comes from watching agreement over "
      "training. The two encodings are trained identically, with the double action masked until episode "
      "500k. While only hit and stand exist, the smooth scalar net actually <i>leads</i> &mdash; ordinal "
      "interpolation is an asset there. The instant the double is introduced, one-hot overtakes and stays "
      "ahead. The crossover is the signature: encoding becomes most visible once a hard-edged decision enters.", body)
    story.append(Spacer(1, 15 * mm))
    fig("crossover", "Figure 7: Agreement over training, matched scalar vs one-hot (double introduced at the "
        "dashed line). Scalar leads the hit/stand phase; one-hot overtakes once the double enters &mdash; "
        "isolating the double as the one place the smooth boundary costs.")
    P("(The network&rsquo;s own feature geometry &mdash; which shows it has learned the structure of the "
      "problem, with the residual confined to the double&rsquo;s edge &mdash; is shown in the appendix.)", body)

    # ----- 7. complete game -----
    story.append(PageBreak())
    P("7. Completing the game: split and surrender as supporting evidence", h1); rule()
    P("The full game adds split and surrender. This section is deliberately lighter: split and surrender "
      "are <i>supporting evidence</i>, not a separately tuned study &mdash; the deep diagnostic work was "
      "the no-split double. The grid grows from 240 cells to 340 (the +100 pair cells; surrender adds an "
      "action, not cells, that basic chooses in just three of them), and basic strategy&rsquo;s own edge "
      "falls, because every option added is one more way to play a hand correctly. The complete-game "
      "result should be read as an extension check, not as the final optimized DQN configuration for "
      "split and surrender.", body)
    story.append(Spacer(1, 15 * mm))
    fg = [thr(["method (game)", "agreement", "edge (%/hand)", "residual"]),
          tr(["tabular &mdash; trimmed", f"{DATA['trim_tab_agree']}%", f"{DATA['trim_tab_edge']:.2f}", "&mdash;"]),
          tr(["best DQN &mdash; trimmed", f"{DATA['trim_best_agree']}%", f"{DATA['trim_best_edge']:.2f}", "double boundary"]),
          tr(["tabular &mdash; complete", f"{DATA['full_tab_agree']}%", f"{DATA['full_tab_edge']:.2f}", "&mdash;"]),
          tr(["DQN &mdash; complete", f"{DATA['full_dqn_agree']}%", f"{DATA['full_dqn_edge']:.2f}", "over-split, surrender-blind"])]
    story.append(styled_table(fg, [W * 0.34, W * 0.18, W * 0.22, W * 0.26], hl_rows=(4,)))
    P("Table 4: The DQN nearly matches the table in the trimmed game and falls about 11 points behind "
      "once split and surrender add hard-edged actions. The exact tabular method remains close to its optimum in "
      "in both games; the DQN falls further behind once the action set adds sharper boundaries." 
      "(Optimum: full basic ~0.54% lit.; complete edges are 5M-hand "
      "re-evaluations, seed-sensitive ~&plusmn;0.05%.)", cap)
    fig("gamegap", "Figure 8: In this extension check, the gap widens as the action set adds sharper decisions. The agreement gap widens "
        "from a few points (trimmed) to about eleven (complete), with the network held in the same family "
        "&mdash; increasing capacity did not remove the gap in these runs. ([64,64] complete edge ~1.17%, no better than [16,16]).")
    # surrender cells, read live
    surr = [c for c in cells(RUN["dqn_full16"]) if c.get("basic_action") == "surrender"]
    if surr:
        rows = [thr(["cell", "basic", "DQN plays"])]
        for c in sorted(surr, key=lambda c: (c["player_value"], c["dealer_upcard"])):
            lab = f"{'soft' if c['is_soft'] else 'hard'} {c['player_value']} vs {('A' if c['dealer_upcard']==11 else c['dealer_upcard'])}"
            rows.append(tr([lab, "surrender", c["agent_action"]]))
        story.append(styled_table(rows, [W * 0.42, W * 0.29, W * 0.29]))
        P("Table 5: Surrender is a small defensive island &mdash; correct in only a few cells. In the "
          "observed complete-game run the DQN enters none of them, exactly the kind of rare, narrow "
          "decision a smooth function struggles to carve out. Split and surrender remain confirming "
          "evidence, not a fully tuned study.", cap)

    # ----- 8. measurement discipline -----
    story.append(PageBreak())
    P("8. Measurement discipline: what kept the report honest", h1); rule()
    P("This report is as much about measurement discipline as about DQN performance, because several "
      "tempting shortcuts would have produced a stronger but less honest story. <b>Reporting the best "
      "checkpoint:</b> runs spike high and settle lower (one one-hot [64,64] peaks ~93% but holds ~91%; "
      "a scalar one peaks ~93% and lives nearer 88%), so back-half means are used throughout. "
      "<b>Trusting the agent&rsquo;s own confidence:</b> the audit separates genuine disagreements from "
      "near-ties using the agent&rsquo;s Q-values, but longer training can shrink self-estimated gaps "
      "and move disagreements into the near-tie bucket without improving agreement or edge &mdash; the "
      "policy did not improve, it just grew more confident about its estimates. <b>Over-ranking edge:</b> "
      "edge is noisy and seed-sensitive, so it supports the policy story, it does not rank close "
      "configurations. <b>Attributing a result before matching the comparison:</b> capacity, schedule, "
      "Double DQN and training length confound easily, so the report separates what a table shows from "
      "what it suggests. The rules are simple: agreement is the audit metric; edge is secondary; prefer "
      "back-half means to peaks; treat close edges as ties; do not let the agent grade itself; and do "
      "not claim a cause the comparison does not support.", body)

    # ----- 9. conclusion -----
    story.append(PageBreak())
    P("9. What this report actually shows", h1); rule()
    P("It does not show that neural networks cannot play blackjack, that DQN has a universal limit, or "
      "that one encoding or schedule is always best. The narrower, more useful result: in this "
      "controlled, known-answer abstraction &mdash; small discrete state space, known reference table, "
      "hard decision boundaries, independently inspectable cells &mdash; the lookup table is the simpler "
      "and more robust representation. The DQN is capable: in the no-split game, with stabilizers, it "
      "approaches the table and its edge lands near the optimum. But it pays for generalization. It is "
      "more sensitive to training dynamics; its high-variance double can oscillate; its readout depends "
      "on checkpoint and seed; its encoding changes the decision geometry; and when the game adds harder, "
      "rarer actions, the gap widens.", body)
    P("The strongest lesson is about <b>tool fit</b>. A table is the right tool when the problem is "
      "small, discrete, known, and exactly auditable. A network becomes interesting when the state space "
      "grows past what a table can hold, or when the environment carries information that changes over "
      "time &mdash; which is exactly where the next stage goes. Card counting, bet sizing and bankroll "
      "make the remaining shoe and survival part of the state; expected value and risk of ruin become "
      "two objectives, and the clean basic-strategy answer key disappears. This report does the known-"
      "answer problem first, on purpose: it establishes the audit habits before removing the key.", body)
    story.append(Spacer(1, 15 * mm))
    tb = [thr(["representation", "strength", "cost"]),
          tr(["lookup table", "exact, inspectable, stable", "needs coverage; does not generalize"]),
          tr(["DQN", "generalizes, compact, extensible", "needs stabilization;can blur boundaries; seed-sensitive"])]
    story.append(styled_table(tb, [W * 0.20, W * 0.36, W * 0.44]))
    P("Table 6: The table wins on exactness and simplicity. The DQN is valuable here because it makes the "
      "cost of generalization measurable &mdash; before moving to a problem where the table no longer "
      "cleanly applies.", cap)

    # ----- limitations & methodology -----
    story.append(PageBreak())
    P("Limitations, scope, and methodology", h1); rule()
    P("This is an exploratory RL investigation, not an exhaustive DQN benchmark, and a few choices and "
      "limits should be explicit.", body)
    notes = [
        ("Scope", "The core experiments use fresh-shoe blackjack and compare learned policies against "
         "total-based basic strategy. Card counting and bankroll management are deliberately out of scope."),
        ("State features", "The agent sees player total, soft/hard flag, dealer upcard (plus split "
         "availability). For fresh-shoe play with no counting this is a sufficient statistic for the "
         "optimal action, which is why a compact network &mdash; not a convolutional or sequence model "
         "&mdash; is the right class: there is no spatial or temporal structure over cards to exploit."),
        ("Primary metric", "Agreement (an exact policy comparison) is primary; edge is noisier, seed-"
         "sensitive, and is read as a band, not used to rank close configurations."),
        ("Seeds", "Some results rest on a limited number of seeds, and seed variance is meaningful for "
         "DQN final checkpoints &mdash; hence back-half means, matched comparisons, and residual patterns "
         "over single final numbers."),
        ("Confounds", "The DQN runs combine several stabilizers (schedule, Double DQN, batch size, reward "
         "baseline, encoding, curriculum, readout averaging); unless a comparison is matched, no single "
         "factor is credited."),
        ("Reproducibility", "Some early runs could not be fully reconstructed because model weights or "
         "full per-cell Q-vectors were not always saved. This does not overturn the agreement-based "
         "storyline, but the lesson &mdash; persist the weights and the full Q-vector for every run "
         "&mdash; is the kind of discipline the next stage will need."),
    ]
    for h, t in notes:
        P(f"<b>{h}.</b> {t}", body)
    P("The conclusion stays narrow on purpose: <b>for this small, known, exactly tabulatable abstraction, "
      "the table is the simpler and more robust tool; the DQN is valuable because it shows what "
      "generalization adds, what it destabilizes, and why representation choices matter &mdash; before "
      "moving to a harder problem where the table no longer cleanly applies.</b>", take)
    story.append(Spacer(1, 15 * mm))
    repbox = [Paragraph("Reproducibility rule for the next stage", boxh),
              Paragraph("Save, for every run: model weights, the full per-cell Q-vector (not just the chosen "
                        "action), run metadata, all random seeds, and the exact evaluation configs. Several "
                        "early runs here could only be partially reconstructed for want of these &mdash; cheap "
                        "to store, expensive to recreate.", boxb)]
    story.append(panel(repbox, bg=C_PANEL, pad=12))

    # ----- appendix: learned geometry -----
    story.append(PageBreak())
    P("Appendix: the geometry of what the network learned", h1); rule()
    P("These projections are complementary evidence for Section 6, kept out of the main flow because they "
      "illustrate rather than prove. For every strategy cell, the trained network computes a penultimate-"
      "layer feature vector; we project those vectors to two dimensions and colour them three ways. "
      "<b>Action</b> uses PCA, whose linear axes preserve the global hit&rarr;double&rarr;stand ordering; "
      "<b>soft/hard</b> and <b>dealer upcard</b> use t-SNE. The point is that the map is not a blur: both "
      "encodings carve coherent action regions, separate soft from hard, and lay the dealer-strength "
      "ordering out as a smooth gradient &mdash; so the network has genuinely learned the structure of the "
      "problem. One-hot separates a little more sharply than scalar, consistent with the encoding result "
      "&mdash; and the residual Section 6 dissects is the one soft boundary at the double&rsquo;s edge, not "
      "missing structure.", body)
    fig("embedding", "Figure 9: Penultimate-layer embeddings, one-hot (left) vs scalar (right). Top &mdash; "
        "action regions via PCA (which preserves the hit&rarr;double&rarr;stand ordering); middle &mdash; "
        "soft vs hard (t-SNE); bottom &mdash; dealer upcard (t-SNE), a smooth strength gradient. Both nets "
        "organise the space by the right structure; one-hot&rsquo;s separation is slightly sharper. Rendered "
        "live from the trained weights.", width=W * 0.7)

    # ----- closing panel -----
    story.append(Spacer(1, 42 * mm))
    closing = [Paragraph("Built as Part of AI Journey",
                         ParagraphStyle("CT", fontSize=16, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=14, leading=20)),
               Paragraph("A structured path from Python foundations to AI engineering.",
                         ParagraphStyle("CB", fontSize=10, textColor=HexColor("#BBDEFB"), alignment=TA_CENTER, fontName="Helvetica", spaceAfter=14, leading=15)),
               Paragraph("github.com/arda-basarici/ai-journey",
                         ParagraphStyle("CL", fontSize=11, textColor=HexColor("#64B5F6"), alignment=TA_CENTER, fontName="Helvetica-Bold", leading=14))]
    story.append(panel(closing, height=150, pad=24))

    doc.build(story)
    print("Report generated:", OUTPUT_PDF)


def main():
    print("building charts from run records ...")
    C = build_charts()
    print("building PDF ...")
    build_pdf(C)


if __name__ == "__main__":
    main()
