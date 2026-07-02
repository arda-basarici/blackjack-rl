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

# repo root = the dir containing runs/. Walk up from cwd (so notebooks run from any subdir resolve
# it); fall back to this package's project root when no runs/ ancestor exists, so importing this
# module never crashes when artifacts are absent (fresh checkout, CI, pdoc importing every module).
ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir()),
    Path(__file__).resolve().parents[1],
)

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
    p = ROOT / "results" / "reeval_results.json"
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


# --- Problem B (B2d) — the learned bettor. Bet-aware companions to the loaders above (the play-side
# ``load_runs`` reads a ``diff``/``edge`` record shape the ``kind=='bet_agent'`` records don't have). ----

BET_COUNTS: list[int] = [-4, -2, 0, 2, 4, 6, 8]     # the checkpoint probe counts
KELLY_LADDER: dict[int, int] = {-4: 1, -2: 1, 0: 1, 2: 2, 4: 5, 6: 8, 8: 8}  # discrete-Kelly target


def _bet_at(checkpoint: dict, count: int) -> float:
    bets = checkpoint["bet_by_count"]                 # JSON stringifies the int keys
    return float(bets.get(str(count), bets.get(count)))


def load_bet_runs(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """Every ``kind=='bet_agent'`` run as one tidy frame (config + trajectory length)."""
    runs_dir = Path(runs_dir) if runs_dir else ROOT / "runs"
    rows = []
    for f in sorted(glob.glob(str(runs_dir / "*" / "record.json")), key=os.path.getmtime):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if r.get("kind") != "bet_agent":
            continue
        c = r.get("config", {}) or {}
        m = r.get("metrics", {}) or {}
        sess = c.get("session", {}) or {}
        rows.append({
            "run": Path(f).parent.name,
            "regime": "growth" if (sess.get("starting_bankroll") or 0) >= 400 else "ruin",
            "gamma": c.get("gamma"), "double": bool(c.get("double_dqn")), "scale": c.get("reward_scale"),
            "batch": c.get("batch_size"), "lr_sched": c.get("lr_schedule"),
            "bankroll_feature": c.get("bankroll_feature", "raw"), "seed": c.get("seed"),
            "n_sess": c.get("n_sessions"), "n_ckpt": len(m.get("learning_curve", [])),
            "path": str(Path(f).parent),
        })
    return pd.DataFrame(rows)


def oracle_run(runs_dir: str | Path | None = None) -> str | None:
    """Path of the latest denoised-reward **oracle** diagnostic (``kind=='bet_oracle'``) — the positive
    control: fed the *expected* (noise-free) log-reward, the SAME DQN learns a clean, stable Kelly ramp, so
    the real-reward flatline is the signal/coverage wall, not a bug or a capacity limit. ``None`` if absent.
    Its record carries a ``learning_curve`` in the bet-run shape, so ``bet_trajectory`` / ``plot_bet_orbit``
    read it directly (it is tagged ``bet_oracle``, so ``load_bet_runs`` correctly skips it)."""
    runs_dir = Path(runs_dir) if runs_dir else ROOT / "runs"
    hits = []
    for f in glob.glob(str(runs_dir / "*bet-oracle*" / "record.json")):
        try:
            if json.load(open(f, encoding="utf-8")).get("kind") == "bet_oracle":
                hits.append(Path(f).parent)
        except Exception:
            continue
    return str(max(hits, key=os.path.getmtime)) if hits else None


# --- coverage: the true-count visit distribution (Ch3.4) — direct evidence for "high counts are rare" ----

def count_frequency() -> pd.DataFrame:
    """True-count visit frequency from the committed 20M-hand edge reference (each per-count bucket carries
    its sample count ``n``). The evidence behind Ch3.4's coverage wall: the counts where Kelly bets big are
    the counts the agent almost never sees. Columns: ``true_count``, ``n``, ``frac`` (share of all hands)."""
    from blackjack_rl.session.references import load_edge_reference
    edges = load_edge_reference().edges
    df = pd.DataFrame([{"true_count": c, "n": e.n} for c, e in edges.items()]).sort_values("true_count")
    df["frac"] = df["n"] / df["n"].sum()
    return df.reset_index(drop=True)


def plot_count_frequency(lo: int = -8, hi: int = 12, note: str | None = None) -> None:
    """The true-count visit distribution (20M-hand reference), log-y so the rarity is legible — the coverage
    wall behind Ch3.4. The region where discrete-Kelly bets above the table minimum (TC >= +2) is shaded:
    exactly the counts that call for a big bet are visited a fraction of a percent of the time."""
    import matplotlib.pyplot as plt
    d = count_frequency()
    d = d[(d.true_count >= lo) & (d.true_count <= hi)]
    plt.figure(figsize=(8, 4))
    plt.bar(d.true_count, d.frac * 100, color=_LADDER_COLORS["flat"], alpha=0.85, width=0.9)
    plt.axvspan(1.5, hi + 0.5, color=_LADDER_COLORS["kelly"], alpha=0.12, label="Kelly bets > 1u (TC >= +2)")
    plt.gca().set(xlabel="true count", ylabel="% of all hands (log scale)", yscale="log",
                  title="high counts are rare — the coverage wall (20M-hand reference)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    if note:
        fignote(note)
    plt.show()


def bet_trajectory(path: str | Path):
    """(sessions, loss, bets) for one run — ``bets`` is the (checkpoint, count) matrix (the orbit)."""
    import numpy as np
    lc = json.load(open(Path(path) / "record.json", encoding="utf-8"))["metrics"]["learning_curve"]
    sessions = np.array([cp["session"] for cp in lc])
    loss = np.array([cp["recent_loss"] if cp.get("recent_loss") is not None else np.nan for cp in lc], float)
    bets = np.array([[_bet_at(cp, c) for c in BET_COUNTS] for cp in lc], dtype=float)
    return sessions, loss, bets


def bet_kelly_distance(bets):
    """L1 distance of each checkpoint's bet curve to the discrete-Kelly ladder (low = near-Kelly)."""
    import numpy as np
    return np.abs(bets - np.array([KELLY_LADDER[c] for c in BET_COUNTS])).sum(axis=1)


def near_kelly_runs(regime: str | None = None, n: int = 4, runs_dir: str | Path | None = None) -> pd.DataFrame:
    """The ``n`` bet runs whose training trajectory gets **closest** to the discrete-Kelly ladder (smallest
    min L1-distance over all its checkpoints) — the runs that actually *visit* Kelly, for the distance-to-
    Kelly figure (so the dips reach the ~Kelly line, not an arbitrary last-4). Optional regime filter; ranked
    closest-first with a ``min_kelly_dist`` column."""
    df = load_bet_runs(runs_dir)
    if regime:
        df = df[df.regime == regime]
    dists = []
    for _, r in df.iterrows():
        try:
            dists.append(float(bet_kelly_distance(bet_trajectory(r.path)[2]).min()))
        except Exception:
            dists.append(float("inf"))
    df = df.assign(min_kelly_dist=dists).sort_values("min_kelly_dist")
    # one run per config (drop re-runs of the same cell) so the figure's traces are distinct
    return df.drop_duplicates(subset=["regime", "double", "lr_sched", "seed"], keep="first").head(n)


def bet_native_curve(path: str | Path, counts=tuple(BET_COUNTS), decks_remaining: float = 3.0) -> dict:
    """The trained agent's greedy bet-vs-count at its NATIVE bankroll (the honest policy probe — not the
    OOD sweep). Rebuilds the agent from the run dir and reads its deterministic policy."""
    from blackjack_rl.session.bet_agent import greedy_bet_curve
    from blackjack_rl.session.persistence import load_bet_agent
    cfg = json.load(open(Path(path) / "record.json", encoding="utf-8"))["config"]
    bankroll = float((cfg.get("session") or {}).get("starting_bankroll") or 400.0)
    return greedy_bet_curve(load_bet_agent(path), counts, bankroll=bankroll, decks_remaining=decks_remaining)


def load_bet_evals(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """Saved four-axis evals (``runs/<id>/eval_*.json``) joined with each run's config — one row per
    (run, phase, regime, bettor). ``phase`` is 'final' or 'best-ckpt'; ``train_regime`` is the native cell."""
    runs_dir = Path(runs_dir) if runs_dir else ROOT / "runs"
    rows = []
    for f in sorted(glob.glob(str(runs_dir / "*" / "eval_*.json"))):
        try:
            rec = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if rec.get("kind") != "bet_eval":
            continue
        cfg = json.load(open(Path(f).parent / "record.json", encoding="utf-8"))["config"]
        train_regime = "growth" if (cfg.get("session", {}).get("starting_bankroll", 0) >= 400) else "ruin"
        phase = "final" if rec.get("checkpoint_session") is None else "best-ckpt"
        for label, m in rec["metrics"].items():
            regime, bettor = label.split("/")
            ddk = next(k for k in m if k.startswith("drawdown_breach"))
            gr = m["growth_rate"]
            rows.append({
                "run": Path(f).parent.name, "phase": phase, "regime": regime, "bettor": bettor,
                "train_regime": train_regime, "seed": cfg.get("seed"), "gamma": cfg.get("gamma"),
                "double": "on" if cfg.get("double_dqn") else "off",
                "bankroll_feature": cfg.get("bankroll_feature", "raw"),
                "growth_1e4": gr["value"] * 1e4, "ruin_pct": m["ruin"]["estimate"] * 100,
                # the eval's own Monte-Carlo 95% CI on the growth estimate — the honest margin for a
                # deterministic (cached) baseline, which has no seed spread of its own.
                "growth_lo_1e4": gr.get("low", float("nan")) * 1e4, "growth_hi_1e4": gr.get("high", float("nan")) * 1e4,
                "dd_pct": m[ddk]["estimate"] * 100, "n_sessions": rec["n_sessions"],
            })
    return pd.DataFrame(rows)


def bet_multiseed_summary(evals: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean/std across seeds for the native-regime agent cell — the CI-backed view. Carries all
    three scalar axes (growth, ruin, deep-drawdown) so the four-axis table reads complete. Grouped by
    ``bankroll_feature`` too, so the growth encodings (raw/logratio/none) stay **separate** rows instead of
    silently pooling into one — they are different agents, and the ``raw`` cell is the headline."""
    native = evals[(evals.bettor == "agent") & (evals.regime == evals.train_regime)]
    return (native.groupby(["train_regime", "gamma", "double", "bankroll_feature", "phase"])
            .agg(n=("seed", "nunique"), growth=("growth_1e4", "mean"), growth_sd=("growth_1e4", "std"),
                 ruin=("ruin_pct", "mean"), dd=("dd_pct", "mean"), dd_sd=("dd_pct", "std"))
            .round(2).reset_index())


# --- committed high-n bet-ladder: the TIGHT Kelly/Flat baseline ------------------------------------
# The four-axis agent evals cache Kelly/Flat at the *agent's* eval size (2000 sess) → a single noisy MC
# estimate whose growth CI spans zero (and whose growth point can land on the wrong side of 0). The B2c
# bet-ladder measured the SAME discrete-Kelly / Flat policies at 20,000 sess/cell (10× tighter, verified
# byte-identical policy: same edge reference, KellyBet(discrete)@400u = 1 1 1 2 5 8 8). Report baselines
# read THIS; only the agent (whose real error bar is the 6-seed spread, not eval-MC) reads the 2000-sess
# evals. Select by config (largest n_sessions_per_cell), never by timestamp.

def _latest_ladder_record(runs_dir: str | Path | None = None) -> dict | None:
    """The committed bet-ladder record with the most sessions/cell (the 20k run, not a 2k sanity pass)."""
    runs_dir = Path(runs_dir) if runs_dir else ROOT / "runs"
    best = None
    for f in glob.glob(str(runs_dir / "*bet-ladder*" / "record.json")):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if r.get("kind") != "bet_ladder":
            continue
        n = (r.get("config") or {}).get("n_sessions_per_cell", 0)
        if best is None or n > best[0]:
            best = (n, r, Path(f).parent.name)
    return None if best is None else {"n": best[0], "record": best[1], "run": best[2]}


def ladder_baselines(runs_dir: str | Path | None = None) -> pd.DataFrame:
    """The committed high-n bet-ladder's baselines as a tidy frame (one row per regime × bettor) — the
    tight reference the report scores the agent against. ``bettor`` is 'flat' / 'kelly' (= discrete Kelly,
    the report's Kelly) / 'kelly-cont' / 'flat-8'. Growth in ×1e-4/hand with its MC 95% CI; deep-drawdown %
    with its Wilson CI. Empty frame if no ladder run exists (so a notebook degrades cleanly)."""
    found = _latest_ladder_record(runs_dir)
    if not found:
        return pd.DataFrame()
    cells = found["record"]["cells"]
    ddk = "drawdown_breach_0.5"
    rows = []
    for regime in ("growth", "ruin"):
        for bettor, cell_key in (("flat", "flat"), ("kelly", "kelly-disc"),
                                 ("kelly-cont", "kelly-cont"), ("flat-8", "flat-8")):
            cell = cells.get(f"{regime}/{cell_key}")
            if cell is None:
                continue
            g, dd = cell["growth_rate"], cell[ddk]
            rows.append({
                "regime": regime, "bettor": bettor,
                "growth_1e4": g["value"] * 1e4, "growth_lo_1e4": g["low"] * 1e4, "growth_hi_1e4": g["high"] * 1e4,
                "dd_pct": dd["estimate"] * 100, "dd_lo_pct": dd["low"] * 100, "dd_hi_pct": dd["high"] * 100,
                "ruin_pct": cell["ruin"]["estimate"] * 100, "n_sessions": g["n"], "run": found["run"],
            })
    return pd.DataFrame(rows)


def ladder_provenance(role: str | None = None, runs_dir: str | Path | None = None) -> str:
    """Source line for a ladder-baseline figure/table: the committed high-n bet-ladder + its edge reference
    — honest about being ONE tight measurement (n sessions/cell), not a run count."""
    found = _latest_ladder_record(runs_dir)
    if not found:
        return "source: bet-ladder (not found)"
    er = (found["record"].get("config") or {}).get("edge_reference", {})
    body = ("%s-session bet-ladder — discrete-Kelly / Flat on basic play; Kelly sized from the %s-hand "
            "edge reference (git %s)" % (f"{found['n']:,}", f"{er.get('n_total', 0):,}", er.get("git_hash", "?")))
    return "source: %s%s" % (("%s · " % role) if role else "", body)


def diff_significance(value_a: float, lo_a: float, hi_a: float,
                      value_b: float, lo_b: float, hi_b: float) -> dict:
    """Two-sided z-test on ``a − b`` for two **independent** estimates given their 95% CIs (the bettors
    share no shoes — ``cell_eval`` seeds each cell from a disjoint range — so the difference is unpaired).
    Recovers each SE from its half-width (95% CI ⇒ ÷1.96); ``SE_diff = hypot(SE_a, SE_b)``. Dependency-light
    (``math.erfc`` for the p-value). Returns ``{gap, z, p, se_diff}``."""
    import math
    se_a, se_b = (hi_a - lo_a) / 2 / 1.96, (hi_b - lo_b) / 2 / 1.96
    gap, se = value_a - value_b, math.hypot(se_a, se_b)
    z = gap / se if se else float("inf")
    return {"gap": gap, "z": z, "p": math.erfc(abs(z) / math.sqrt(2)), "se_diff": se}


def kelly_beats_flat(regime: str, runs_dir: str | Path | None = None) -> dict:
    """The report's headline test on the TIGHT 20k ladder: is discrete-Kelly's growth > Flat's in ``regime``?
    Independent-sample z-test → ``{regime, kelly_1e4, flat_1e4, gap, z, p, se_diff}``. p<0.05 ⇒ resolved."""
    b = ladder_baselines(runs_dir)
    k = b[(b.regime == regime) & (b.bettor == "kelly")].iloc[0]
    f = b[(b.regime == regime) & (b.bettor == "flat")].iloc[0]
    out = diff_significance(k.growth_1e4, k.growth_lo_1e4, k.growth_hi_1e4,
                            f.growth_1e4, f.growth_lo_1e4, f.growth_hi_1e4)
    return {"regime": regime, "kelly_1e4": k.growth_1e4, "flat_1e4": f.growth_1e4, **out}


# --- provenance for the bet frames -----------------------------------------------------------------
# The play-side ``provenance`` reads a runs frame with ``method``/``encoding`` columns; the bet frames
# (``load_bet_runs`` / ``load_bet_evals``) carry a different config shape, so they get their own reader.

_BET_FIELDS = ["regime", "train_regime", "gamma", "double", "bankroll_feature", "scale", "lr_sched",
               "n_sess", "phase"]


def _bet_bits(cfg) -> list[str]:
    """Config (a Series or dict) → ordered human-readable bits — what is *special* about a bet run, not
    its opaque id. Skips missing fields; normalises the two double flavours (bool / 'on'|'off')."""
    g = cfg.get
    bits = ["DQN bettor"]
    regime = g("regime") if g("regime") in ("growth", "ruin") else g("train_regime")
    if regime:
        bits.append(str(regime))
    if pd.notna(g("gamma")):
        bits.append("g=%g" % g("gamma"))
    double = g("double")
    if double is not None:
        bits.append("double-DQN" if double in (True, "on") else "single-DQN")
    if g("bankroll_feature"):
        bits.append("%s-enc" % g("bankroll_feature"))
    if pd.notna(g("scale")):
        bits.append("scale%g" % g("scale"))
    if g("lr_sched"):
        bits.append("%s-lr" % g("lr_sched"))
    if g("phase"):
        bits.append("%s policy" % g("phase"))
    return bits


def bet_provenance(sel, role: str | None = None) -> str:
    """A source line for a bet figure/table: WHAT the run(s) are (config via ``_bet_bits``), not just an id.
    Pass a Series (single run) or a DataFrame (a multi-seed group → shared config + the seed list, naming
    any fields that vary). Render *below* the artifact via ``show(..., source=bet_provenance(...))`` or
    ``fignote(bet_provenance(...))`` — the bet-side twin of ``provenance``."""
    if isinstance(sel, pd.Series) or len(sel) == 1:
        row = sel if isinstance(sel, pd.Series) else sel.iloc[0]
        body = "single run — " + " · ".join(_bet_bits(row))
        if row.get("seed") is not None:
            body += " · seed %s" % row.get("seed")
    else:
        rep, varies = {}, []
        for f in _BET_FIELDS:
            if f not in sel:
                continue
            uniq = [x for x in sel[f].dropna().unique()]
            if len(uniq) == 1:
                rep[f] = uniq[0]
            elif len(uniq) > 1:
                varies.append(f)
        seeds = sorted({s for s in sel.get("seed", pd.Series(dtype=object)).tolist() if s is not None})
        n_runs = sel["run"].nunique() if "run" in sel else len(sel)   # rows may double per run (final/best-ckpt)
        # "across N runs" not "mean over N" — the group may feed several per-config means or a table, not one average
        body = "across %d runs — %s · seeds %s" % (
            n_runs, " · ".join(_bet_bits(rep)), ", ".join(map(str, seeds)) or "?")
        if varies:
            body += " · (varies: %s)" % ", ".join(varies)
    return "source: %s%s" % (("%s · " % role) if role else "", body)


# --- bet-model figures — clean, self-contained plotters (the chapters call these, one line each) --------

_LADDER_COLORS = {"agent": "#d95f02", "kelly": "#1b9e77", "flat": "#7570b3"}


def plot_bet_orbit(path: str | Path, note: str | None = None) -> None:
    """Heatmap of greedy bet level vs true count over training, loss beneath — the 'training orbit'.
    `note` (e.g. ``bet_provenance(row)``) is rendered as a muted source line under the figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    sessions, loss, bets = bet_trajectory(path)
    fig, (heat, lo) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    mesh = heat.pcolormesh(sessions, np.arange(len(BET_COUNTS)), bets.T, cmap="viridis", shading="nearest", vmin=1, vmax=8)
    heat.set_yticks(np.arange(len(BET_COUNTS)))
    heat.set_yticklabels([f"{c:+d}" for c in BET_COUNTS])
    heat.set(ylabel="true count", title=f"training orbit — bet vs count  ({Path(path).name[:18]})")
    fig.colorbar(mesh, ax=heat, label="bet level")
    lo.plot(sessions, loss, lw=1)
    lo.set(xlabel="session", ylabel="loss")
    lo.grid(alpha=0.3)
    fig.tight_layout()
    if note:
        fignote(note)
    plt.show()


def plot_kelly_distance(runs: pd.DataFrame, label=lambda r: f"{'on' if r.double else 'off'}/s{r.seed}",
                        note: str | None = None) -> None:
    """L1-distance-to-Kelly over training for each run — dips mark near-Kelly ramps the orbit visits."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 3.6))
    for _, r in runs.iterrows():
        sessions, _, bets = bet_trajectory(r.path)
        plt.plot(sessions, bet_kelly_distance(bets), lw=1.2, label=label(r))
    plt.axhline(2, ls="--", c="green", alpha=0.6, label="dist<=2 (~Kelly)")
    plt.gca().set(xlabel="session", ylabel="L1 distance to Kelly",
                  title="the orbit visits near-Kelly ramps (dips) but never settles")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    if note:
        fignote(note)
    plt.show()


def plot_ladder_bars(evals: pd.DataFrame, metric: str, agent_cfg: dict, title: str | None = None,
                     phase: str = "final", note: str | None = None) -> None:
    """Agent vs Kelly vs Flat bars for `metric`, grouped by regime (native cell). The **agent** bar is the
    2000-sess eval mean, with the **individual seeds drawn as dots** (honest at small n — a symmetric SD
    whisker there is ~the 95% CI and dwarfs the bar); **Kelly/Flat** are the tight 20k-ladder baselines
    (bar = value ± the eval's own 95% CI). `agent_cfg` picks the headline agent config per regime. `phase`
    selects the agent policy — 'final' (deployed) or 'best-ckpt' (H3 diagnostic). Baselines are phase-invariant."""
    import matplotlib.pyplot as plt
    import numpy as np
    fin = evals[(evals.phase == phase) & (evals.regime == evals.train_regime)]
    base = ladder_baselines()
    ci_cols = {"growth_1e4": ("growth_lo_1e4", "growth_hi_1e4"), "dd_pct": ("dd_lo_pct", "dd_hi_pct")}
    lo_col, hi_col = ci_cols.get(metric, (None, None))
    plt.figure(figsize=(7.5, 4))
    for i, regime in enumerate(("growth", "ruin")):
        agent = fin[(fin.train_regime == regime) & (fin.bettor == "agent")]
        for key, val in agent_cfg.get(regime, {}).items():
            agent = agent[agent[key] == val]
        # agent = mean bar + the individual seeds as dots. At n~6 a symmetric SD whisker is ~the 95% CI and
        # would dwarf the bar (and hide the real spread, which for best-ckpt IS the finding) — so show the
        # raw points instead. Baselines (below) keep their deterministic MC CI, a different uncertainty type.
        vals = agent[metric].to_numpy()
        plt.bar(i * 4, vals.mean() if len(vals) else 0, color=_LADDER_COLORS["agent"], alpha=0.5)
        if len(vals):
            plt.scatter(i * 4 + np.linspace(-0.24, 0.24, len(vals)), vals, s=15,
                        color=_LADDER_COLORS["agent"], edgecolor="black", linewidth=0.4, zorder=3)
        for j, bettor in enumerate(("kelly", "flat"), start=1):
            row = base[(base.regime == regime) & (base.bettor == bettor)]
            if row.empty:
                continue
            row = row.iloc[0]
            yerr = (row[hi_col] - row[lo_col]) / 2 if lo_col else 0
            plt.bar(i * 4 + j, row[metric], yerr=yerr, capsize=4, color=_LADDER_COLORS[bettor], alpha=0.85)
    plt.xticks([i * 4 + j for i in (0, 1) for j in range(3)], ["agent", "kelly", "flat"] * 2)
    plt.gca().set(title=title or metric)
    plt.gca().text(1, 1.0, "GROWTH", transform=plt.gca().get_xaxis_transform(), ha="center", fontsize=9, color="grey")
    plt.gca().text(5, 1.0, "RUIN", transform=plt.gca().get_xaxis_transform(), ha="center", fontsize=9, color="grey")
    plt.grid(alpha=0.3, axis="y")
    if note:
        fignote(note)
    plt.show()


def plot_signal_vs_noise(signal: float, noise_sd: float, unit: str = "per-hand reward",
                         signal_label: str = "edge", note: str | None = None) -> None:
    """Schematic (NOT fitted data): the per-hand outcome distribution, mean shifted by `signal` against
    spread `noise_sd`. Two near-identical gaussians — the point is they overlap almost perfectly, so the
    edge a value-learner must estimate is buried in per-hand noise (huge samples to resolve). Feed sourced
    numbers (e.g. per-hand reward SD ~1.15, basic edge ~0.0054/hand) and label the figure a schematic."""
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(-3.2 * noise_sd, 3.2 * noise_sd, 500)

    def bell(mu):
        return np.exp(-0.5 * ((x - mu) / noise_sd) ** 2)

    plt.figure(figsize=(7.5, 3.8))
    plt.plot(x, bell(0.0), color=_LADDER_COLORS["flat"], label="no edge (mean 0)")
    plt.plot(x, bell(signal), color=_LADDER_COLORS["kelly"], ls="--",
             label="with %s (mean +%.2g)" % (signal_label, signal))
    plt.axvline(0.0, color=_LADDER_COLORS["flat"], lw=0.6, alpha=0.5)
    plt.axvline(signal, color=_LADDER_COLORS["kelly"], lw=0.6, alpha=0.5)
    plt.gca().set(xlabel=unit, yticks=[],
                  title="schematic: the %s is sub-noise — shift %.2g vs SD %.2g  (%.2g of one SD)"
                        % (signal_label, signal, noise_sd, signal / noise_sd))
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    if note:
        fignote(note)
    plt.show()


def plot_prize_bar(regime: str = "growth", note: str | None = None) -> None:
    """Kelly vs Flat growth in one regime from the TIGHT 20k ladder, with the *prize* (the gap Kelly buys
    over Flat) annotated and both 95% CIs drawn. Chapter 3's opening number. Both bars are typically **below
    zero** — even optimal Kelly is net-negative here (the table-min tax); it merely loses *less* than Flat."""
    import matplotlib.pyplot as plt
    b = ladder_baselines()
    b = b[b.regime == regime]
    k, f = b[b.bettor == "kelly"].iloc[0], b[b.bettor == "flat"].iloc[0]
    kelly, flat = float(k.growth_1e4), float(f.growth_1e4)
    k_err, f_err = (k.growth_hi_1e4 - k.growth_lo_1e4) / 2, (f.growth_hi_1e4 - f.growth_lo_1e4) / 2
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [flat, kelly], yerr=[f_err, k_err], capsize=5,
           color=[_LADDER_COLORS["flat"], _LADDER_COLORS["kelly"]], alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flat", "Kelly"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.annotate("", xy=(1, kelly), xytext=(1, flat),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.08, (kelly + flat) / 2, "prize\n%.2fe-4/hand" % (kelly - flat), va="center", fontsize=9)
    ax.set(ylabel="growth rate (x1e-4/hand)", title="the entire prize Kelly buys over Flat (%s)" % regime)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if note:
        fignote(note)
    plt.show()


def plot_native_curves(runs: pd.DataFrame, title: str | None = None, label=None,
                       note: str | None = None) -> None:
    """Greedy bet-vs-count at each run's NATIVE bankroll — the honest policy probe (mostly flat), with the
    Kelly ladder dashed for reference. `label` maps a run row → a legend string."""
    import matplotlib.pyplot as plt
    label = label or (lambda r: f"s{r.seed}/{r.bankroll_feature}")
    plt.figure(figsize=(8, 4.2))
    for _, r in runs.iterrows():
        curve = bet_native_curve(r.path)
        counts = sorted(curve)
        plt.plot(counts, [curve[c] for c in counts], marker="o", lw=1.3, alpha=0.75, label=label(r))
    plt.plot(BET_COUNTS, [KELLY_LADDER[c] for c in BET_COUNTS], "k--", lw=1.6, label="Kelly ladder")
    plt.gca().set(xlabel="true count", ylabel="bet level", ylim=(0, 8.5),
                  title=title or "native bet-vs-count — mostly flat, never the ramp")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.3)
    if note:
        fignote(note)
    plt.show()


def plot_bet_replication(regime: str = "growth", feature: str = "raw", bankrolls=(400, 100),
                         counts=tuple(BET_COUNTS), note: str | None = None) -> None:
    """Bet-vs-count for EVERY seed of a config, at each bankroll, discrete-Kelly dashed per panel — the
    falsification of the embedding's 'the bet tracks count' read. If that structure were a real property of
    the policy the seeds would agree; instead it is **seed-specific noise** (native: flat / coarse gate /
    erratic; OOD-low: some seeds rise with count, some fall, some saturate), so no wealth-vs-count read
    survives replication. This is why the embedding is suggestive, not decisive."""
    import matplotlib.pyplot as plt
    from blackjack_rl.session.bet_agent import KellyBet, greedy_bet_curve
    from blackjack_rl.session.persistence import load_bet_agent
    from blackjack_rl.session.references import load_edge_reference
    kb = KellyBet(load_edge_reference().kelly_curve, discretize=True)
    sel = load_bet_runs()
    sel = sel[(sel.regime == regime) & (sel.bankroll_feature == feature)]
    agents = {int(s): load_bet_agent(g.iloc[-1].path) for s, g in sel.groupby("seed")}
    fig, axes = plt.subplots(1, len(bankrolls), figsize=(5.5 * len(bankrolls), 4), sharey=True)
    axes = list(axes) if len(bankrolls) > 1 else [axes]
    for ax, W in zip(axes, bankrolls):
        for s, ag in sorted(agents.items()):
            cur = greedy_bet_curve(ag, counts, bankroll=float(W), decks_remaining=3.0)
            ax.plot(counts, [cur[c] for c in counts], marker="o", lw=1.2, alpha=0.7, label=f"seed {s}")
        ax.plot(counts, [kb.bet(true_count=float(c), decks_remaining=3.0, bankroll=float(W)) for c in counts],
                "k--", lw=1.6, label="Kelly")
        ax.set(xlabel="true count", ylim=(0, 8.5),
               title=f"bankroll {W}u{' (native)' if W >= 400 else ' (out-of-distribution)'}")
        ax.grid(alpha=0.3)
    axes[0].set(ylabel="bet level")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("the embedding's bet-structure doesn't replicate — every seed bets differently")
    fig.tight_layout()
    if note:
        fignote(note)
    plt.show()


def plot_encoding_ablation(evals: pd.DataFrame, metric: str = "growth_1e4", title: str | None = None,
                           note: str | None = None) -> None:
    """Growth by bankroll encoding (raw / logratio / none) in the growth regime, Kelly & Flat dashed — the
    controlled test of the wealth hypothesis. All three bars ≈ Flat ⇒ encoding-invariant (falsified)."""
    import matplotlib.pyplot as plt
    fin = evals[(evals.phase == "final") & (evals.train_regime == "growth") & (evals.regime == "growth")]
    agent = fin[fin.bettor == "agent"]
    import numpy as np
    order = [e for e in ("raw", "logratio", "none") if (agent["bankroll_feature"] == e).any()]
    plt.figure(figsize=(6.5, 4))
    for i, enc in enumerate(order):
        vals = agent[agent["bankroll_feature"] == enc][metric].to_numpy()   # per-seed values
        plt.bar(i, vals.mean() if len(vals) else 0, color=_LADDER_COLORS["agent"], alpha=0.5)
        if len(vals):                                                        # dots, not a bar-dwarfing whisker
            plt.scatter(i + np.linspace(-0.22, 0.22, len(vals)), vals, s=15,
                        color=_LADDER_COLORS["agent"], edgecolor="black", linewidth=0.4, zorder=3)
    bg = ladder_baselines()  # tight 20k baselines, not the noisy 2k cached ones
    bg = bg[bg.regime == "growth"]
    plt.axhline(float(bg.loc[bg.bettor == "flat", metric].iloc[0]), ls="--", color=_LADDER_COLORS["flat"], label="Flat (20k)")
    plt.axhline(float(bg.loc[bg.bettor == "kelly", metric].iloc[0]), ls="--", color=_LADDER_COLORS["kelly"], label="Kelly (20k)")
    plt.xticks(range(len(order)), order)
    plt.gca().set(xlabel="bankroll encoding", ylabel=metric,
                  title=title or "encoding ablation — remove wealth, nothing changes (all ~ Flat)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    if note:
        fignote(note)
    plt.show()


# --- learned-representation probe (Ch4) — penultimate-layer embedding of a state grid -----------------
# Promoted from the scratch investigation notebook: the net's internal representation, projected to 2-D and
# coloured by bet / count / bankroll. Clusters by count ⇒ learned the edge; by bankroll ⇒ keyed on wealth.

def bet_embedding(path: str | Path, counts=range(-8, 11), depths=(1.0, 2.0, 3.0, 4.5, 6.0),
                  bankrolls=(50, 100, 150, 200, 300, 400, 500, 600)):
    """Penultimate-layer embedding (post final ReLU) of a grid of (count, depth, bankroll) states for one
    agent, plus its greedy bet at each. Returns (states_df, embedding[n, hidden], bet[n]). NOTE the
    bankroll grid is deliberately WIDE (50–600u) — an OOD sweep; the agents live at ~400u (see Ch4)."""
    import numpy as np
    import torch
    from blackjack_rl.session.persistence import load_bet_agent
    agent = load_bet_agent(path)
    states = pd.DataFrame([{"true_count": tc, "decks_remaining": d, "bankroll": b}
                           for tc in counts for d in depths for b in bankrolls])
    feats = torch.tensor([agent.encode_state(tc, d, b) for tc, d, b in
                          zip(states["true_count"], states["decks_remaining"], states["bankroll"])],
                         dtype=torch.float32)
    with torch.no_grad():
        embedding = agent.q_net.features(feats).numpy()
        bet = np.array(agent.levels)[agent.q_net(feats).argmax(1).numpy()]
    return states, embedding, bet


def bet_project(embedding, method: str = "pca", seed: int = 0):
    """2-D projection of an embedding: 'pca' (linear, fast) or 'tsne' (local structure)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embedding)
    return TSNE(n_components=2, init="pca", random_state=seed,
                perplexity=min(30, len(embedding) - 1)).fit_transform(embedding)


def plot_bet_embedding(coords, color, label: str, title: str, cmap: str = "viridis",
                       note: str | None = None) -> None:
    """One 2-D embedding scatter, coloured by `color` (bet / count / bankroll)."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 4.5))
    pts = plt.scatter(coords[:, 0], coords[:, 1], c=color, cmap=cmap, s=10, alpha=0.7)
    plt.colorbar(pts, label=label)
    plt.gca().set(xticks=[], yticks=[], title=title)
    plt.tight_layout()
    if note:
        fignote(note)
    plt.show()
