"""Shared loader for the analysis notebooks — saved run records → a tidy DataFrame.

Every notebook imports ``load_runs()`` so run selection is consistent and the canonical-run choices
are explicit (filter the frame, don't hardcode timestamped paths). One source of truth for the
numbers that go into the report.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from pathlib import Path

import pandas as pd

# repo root = the dir containing runs/
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir())

# --- corrected edge benchmark (audit) ---------------------------------------------------------
# A run's saved ``basic_edge_pct`` is ONE eval_seed=0, 200k-hand sample (per-hand reward SD ~1.15
# => SE ~0.26%/hand). seed 0 drew ~1.9 SE LOW (~0.09%), and because every run reuses seed 0 that
# low draw never averaged out — so ``basic_edge_pct`` is NOT the optimum. Measured here over 1.4M
# fresh-shoe hands, basic strategy's true edge in this 6-deck S17 3:2 game is 0.58% +/- 0.10%,
# matching the literature "new 6-deck, standard rules" figure (~0.54%). The reachable optimum
# differs by game because the no-split agents cannot split:
EDGE_SE_PCT = 0.26                 # 1 SE on any single 200k-hand edge eval; don't rank gaps < ~0.5%
OPTIMUM_PCT = {                    # basic-strategy house edge, %/hand, fresh 6-deck S17 3:2
    "complete": 0.54,             # full action set (split [+surrender]): lit. row 2; our 1.4M = 0.58 +/-0.10
    "trimmed": 1.11,              # no split/surrender — the no-split agents' reachable optimum: lit. row 6
}


def reeval_edges() -> dict:
    """Tight re-evaluated edges (%/hand) from ``reeval_results.json`` (written by ``reeval_edges.py`` over
    millions of hands), keyed by BOTH policy label and run-id. Returns ``{}`` until that file exists, so a
    scoreboard cleanly falls back to the recorded 200k edge in the meantime — no hardcoded numbers."""
    p = ROOT / "reeval_results.json"
    out = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:                      # partial/corrupt write mid-run — fall back to recorded edges
        return {}
    for r in data.get("results", []):
        if r.get("edge_pct") is None:
            continue
        out[r["policy"]] = r["edge_pct"]
        if r.get("run"):
            out[r["run"]] = r["edge_pct"]
    return out


def tight_edge(key, default=None):
    """The re-evaluated (tight, millions-of-hands) edge for a run-id or policy label, else ``default`` —
    so a scoreboard reads the re-eval when present and the recorded 200k value until then."""
    return reeval_edges().get(key, default)


def _soft20_double_std(lc: list) -> float | None:
    """Back-half std of Q(double | soft-20 vs dealer-8) — the oscillation metric. None if not logged."""
    d = [cp["probe_q"]["soft20_v8"]["double"] for cp in lc if cp.get("probe_q")]
    return st.pstdev(d[len(d) // 2:]) if len(d) >= 2 else None


def load_runs(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """All saved runs as one DataFrame. ``method`` is 'dqn' if the config has an ``encoding`` field,
    else 'tabular'. Sorted by wall-clock save time."""
    runs_dir = Path(runs_dir) if runs_dir else ROOT / "runs"
    rows = []
    for f in sorted(glob.glob(str(runs_dir / "*" / "record.json")), key=os.path.getmtime):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        c = r.get("config") or {}
        lc = r.get("learning_curve", [])
        is_dqn = "encoding" in c
        rows.append({
            "run": Path(f).parent.name,
            "method": "dqn" if is_dqn else "tabular",
            "encoding": c.get("encoding"),
            "hidden": str(c.get("hidden")) if c.get("hidden") else None,
            # old runs predate these fields; their effective defaults are constant lr / no curriculum
            "lr_schedule": c.get("lr_schedule") or "constant",
            "lr": c.get("lr"),
            "lr_end": c.get("lr_end"),
            "lr_hold_until": c.get("lr_hold_until") or 0,   # >0 = flat-then-decay (paired with double_after)
            "episodes": c.get("num_episodes"),
            "batch": c.get("batch_size"),
            "double_dqn": bool(c.get("double_dqn")),
            "swa": bool(c.get("swa")),
            "exploring_starts": bool(c.get("exploring_starts")),
            "train_every": c.get("train_every") or 4,
            "target_sync_every": c.get("target_sync_every"),
            "target_tau": c.get("target_tau") or 0.0,
            "double_after": c.get("double_after") or 0,
            "reward_baseline": c.get("reward_baseline", "none"),
            "with_splits": bool(c.get("with_splits")),
            "with_surrender": bool(c.get("with_surrender")),
            "seed": c.get("seed"),
            "agreement": round(r["diff"]["agreement_unweighted"], 4),
            "edge_pct": round(r["metrics"]["agent"]["edge"] * 100, 3),
            # basic strategy's edge measured IN THE SAME run: same fresh-shoe harness, same eval
            # hands, same seed as the agent edge above — a paired comparison. The report's "optimum"
            # floor reads this column instead of a hardcoded constant, so every row is one metric.
            "basic_edge_pct": round((r.get("metrics") or {}).get("basic", {}).get("edge", float("nan")) * 100, 3)
                if (r.get("metrics") or {}).get("basic") else None,
            "basic_edge_se_pct": round((r.get("metrics") or {}).get("basic", {}).get("std_error", 0.0) * 100, 3)
                if (r.get("metrics") or {}).get("basic") else None,
            "soft20_double_std": _soft20_double_std(lc),
            "has_qgrid": bool(c.get("log_q_grid")),
            "has_model": (Path(f).parent / "model.pt").exists(),
            "path": str(Path(f).parent),
        })
    df = pd.DataFrame(rows)
    # Fold the tight re-evaluated edges (reeval_results.json, millions of hands) over the noisy 200k
    # record edges, keyed by run-id — so every scoreboard reads one consistent source. No-op until the
    # file exists. Only the re-evaluated runs change; everything else keeps its recorded edge.
    _re = reeval_edges()
    if _re and len(df):
        df["edge_pct"] = [round(_re.get(run, e), 3) for run, e in zip(df["run"], df["edge_pct"])]
    return df


def learning_curve(path: str | Path) -> list[dict]:
    """The per-checkpoint learning curve for one run (episode, agreement, probe_q, q_grid, ...)."""
    return json.load(open(Path(path) / "record.json", encoding="utf-8")).get("learning_curve", [])


def probe_trajectory(path: str | Path, cell: str = "soft20_v8", action: str = "double"):
    """(episodes, values) for one probe cell's action-Q over training — the oscillation trace."""
    lc = learning_curve(path)
    eps = [cp["episode"] for cp in lc if cp.get("probe_q")]
    vals = [cp["probe_q"][cell][action] for cp in lc if cp.get("probe_q")]
    return eps, vals


def sample_counts(path: str | Path) -> dict:
    """{(player_value, is_soft, dealer_upcard, action): count} — how often each cell/action was trained."""
    rec = json.load(open(Path(path) / "record.json", encoding="utf-8"))
    return {(s["player_value"], s["is_soft"], s["dealer_upcard"], s["action"]): s["count"]
            for s in rec.get("sample_counts", [])}


# --- cell-level analysis (the diff cells saved in each record) --------------------------------------

def diff_cells(path: str | Path) -> list[dict]:
    """Every compared cell from a run's saved diff (player_value, is_soft, dealer_upcard, can_split,
    agent_action, basic_action, category, agent_q, basic_q, visits)."""
    return json.load(open(Path(path) / "record.json", encoding="utf-8"))["diff"]["cells"]


def _cell_key(c: dict) -> tuple:
    return (c["player_value"], c["is_soft"], c["dealer_upcard"], c.get("can_split", False))


def cell_categories(path: str | Path) -> dict:
    """{cell_key: category} for one run — the per-cell agree/disagree verdict."""
    return {_cell_key(c): c["category"] for c in diff_cells(path)}


def agreement_on(path: str | Path, keys=None) -> float:
    """Agreement over a given set of cell keys (default: all of the run's cells). Used to score two
    methods on the *same* cells — the common grid — when their native cell-sets differ."""
    cats = cell_categories(path)
    ks = [k for k in (keys if keys is not None else cats) if k in cats]
    return sum(1 for k in ks if cats[k] == "agree") / len(ks) if ks else float("nan")


def common_grid_agreement(path_a, path_b):
    """(agree_a, agree_b, n_common): each run's agreement over the cells they *both* evaluate.
    The apples-to-apples tabular-vs-network number (their native grids differ; this intersects them)."""
    a, b = cell_categories(path_a), cell_categories(path_b)
    common = set(a) & set(b)
    def ag(cats):
        return sum(1 for k in common if cats[k] == "agree") / len(common) if common else float("nan")
    return ag(a), ag(b), len(common)


# --- display: render a tidy aligned table instead of a fragile print -------------------------------

def show(df, pct=(), num=(), caption=None, source=None):
    """Render a DataFrame as a clean, aligned HTML table (no collapsed columns): ``pct`` columns as
    percentages, ``num`` columns to 2 decimals, the row index hidden. Build tables with raw numbers,
    format them here.

    ``caption`` is the descriptive title (above the table); ``source`` is the provenance line, rendered
    as a small muted **note below** the table — kept separate so the two never merge into one blob.

    Renders the table **immediately** (via ``IPython.display``) and returns None — so a cell can hold
    several tables plus prints/figures and every table still shows (not just the last expression). Falls
    back to a plain formatted DataFrame if jinja2/IPython is unavailable, so a table never crashes."""
    fmt = {**{c: "{:.1%}".format for c in pct}, **{c: "{:.2f}".format for c in num}}
    try:
        styler = df.style.format(fmt).hide(axis="index")
        if caption:
            styler = styler.set_caption(caption)
        from IPython.display import display, HTML
        if source:
            try:
                body = styler.to_html()
            except Exception:
                body = styler._repr_html_()
            note = ('<div style="font-size:0.8em; color:#8a8a8a; font-style:italic; '
                    'margin-top:4px">%s</div>' % source)
            display(HTML(body + note))
        else:
            display(styler)
    except Exception:  # jinja2 / IPython missing — format in place and show a plain (still tidy) table
        out = df.copy()
        for c, f in fmt.items():
            out[c] = out[c].map(lambda v: f(v) if pd.notna(v) else v)
        try:
            from IPython.display import display
            display(out.reset_index(drop=True))
        except Exception:
            print(out.reset_index(drop=True).to_string(index=False))


# --- provenance: every table/number states where it came from (config, not just a hash) -------------

def _fmt_lr(x) -> str:
    return "%g" % x if x is not None else "?"


def describe(run, seed: bool = True) -> str:
    """A compact, human-readable **config** for one run (a Series) — what is *special* about it, not its
    opaque id. e.g. ``DQN · onehot · [64, 64] · constant lr 0.001 · 2M ep · seed 42``. This is what a
    reader actually needs to know which run a figure came from and why it was chosen."""
    g = run.get
    if g("method") == "tabular":
        bits = ["tabular Q-learner", "1/n step"]          # tabular uses a 1/n step, not a fixed lr
    else:
        bits = ["DQN"]
        if g("encoding"):
            bits.append(g("encoding"))
        if g("hidden"):
            bits.append(str(g("hidden")))
        sch = g("lr_schedule") or "constant"
        if sch == "constant":
            bits.append("constant lr %s" % _fmt_lr(g("lr")))
        else:
            bits.append("%s lr %s→%s" % (sch, _fmt_lr(g("lr")), _fmt_lr(g("lr_end"))))
    if g("episodes"):
        bits.append("%gM ep" % (g("episodes") / 1e6))
    if g("reward_baseline") not in (None, "none"):
        bits.append("%s-baseline" % g("reward_baseline"))
    if g("double_after"):
        bits.append("double@%gk" % (g("double_after") / 1e3))
    if g("double_dqn"):
        bits.append("Double-DQN")
    if g("swa"):
        bits.append("SWA")
    if g("target_tau"):
        bits.append("soft-target")
    if g("with_splits"):
        bits.append("+splits")
    s = " · ".join(str(b) for b in bits)
    if seed and g("seed") is not None:
        s += " · seed %s" % g("seed")
    return s


def provenance(sel, role: str | None = None) -> str:
    """A standard source line that reads on its own: WHAT the run is (its config via ``describe``), not
    just an id. Pass a row (single run) or a group (DataFrame). Optional ``role`` is a short human label
    of why this run is here ('naive exemplar', 'best stacked run'). Render it *below* the data via
    ``show(..., source=provenance(...))`` (muted note under a table) or ``fignote(provenance(...))``
    (under a figure) — never merged into the descriptive caption.

    Report rule: a *single-run* figure illustrates a mechanism (trajectories, q-grids — not averageable);
    an *averaged* number is a quantitative claim (edges, agreements). The line says which."""
    if isinstance(sel, pd.Series) or len(sel) == 1:
        # a single run: its full config is exactly what the reader needs
        row = sel if isinstance(sel, pd.Series) else sel.iloc[0]
        body = "single run — " + describe(row)
    else:
        # a group may mix configs (encoding, seed, ...), so describe only what is *shared* across it and
        # name the fields that vary — never pin one run's config on the whole average.
        body = "mean over %d runs — %s" % (len(sel), _shared_describe(sel))
    return "source: %s%s" % (("%s · " % role) if role else "", body)


def _shared_describe(df) -> str:
    """Describe a group by the config fields that are *constant* across it, listing which key fields vary
    — so an averaged row never claims a single run's config."""
    fields = ["method", "encoding", "hidden", "lr_schedule", "lr", "lr_end", "episodes",
              "reward_baseline", "double_after", "double_dqn", "swa", "target_tau", "with_splits"]
    rep, varies = {}, []
    for f in fields:
        if f not in df:
            continue
        u = [x for x in df[f].dropna().unique()]
        if len(u) == 1:
            rep[f] = u[0]
        elif len(u) > 1 and f in ("encoding", "lr_schedule", "double_dqn", "reward_baseline", "seed"):
            varies.append(f)
    desc = describe(pd.Series(rep), seed=False)
    seeds = sorted({s for s in df.get("seed", pd.Series(dtype=object)).tolist() if s is not None})
    desc += " · seeds %s" % (", ".join(map(str, seeds)) or "?")
    if varies:
        desc += " · (varies: %s)" % ", ".join(varies)
    return desc


def fignote(text: str) -> None:
    """Drop a muted provenance note beneath the current matplotlib figure (call before ``plt.show()``)."""
    import matplotlib.pyplot as plt
    plt.gcf().text(0.99, -0.02, text, ha="right", va="top", fontsize=8, color="#8a8a8a", style="italic")
