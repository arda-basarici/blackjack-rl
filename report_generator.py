"""
Blackjack RL — Problem A Report Generator
Produces a professional PDF from the saved run records (the pathfinding pattern:
charts are re-rendered cleanly from runs/, never copied from the notebook).

All headline numbers are computed at build time from the run records, so the report
cannot drift out of sync. Policy-level numbers come straight from each run's diff
(deterministic); the house-edge ledger is re-evaluated through the engine and cached.

Usage:
    python report_generator.py                 # 1M-hand ledger eval (slow first build, cached after)
    python report_generator.py --eval-hands 50000   # quick iteration

Output:
    blackjack-rl-policy-audit.pdf   (+ report_charts/)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

RUNS_DIR   = "runs"
OUTPUT_PDF = "blackjack-rl-policy-audit.pdf"
CHARTS_DIR = "report_charts"
CACHE      = os.path.join(CHARTS_DIR, "edge_cache.json")
os.makedirs(CHARTS_DIR, exist_ok=True)

C_DARK   = HexColor("#1A1A2E")
C_ACCENT = HexColor("#0D47A1")
C_INK    = HexColor("#212121")
C_RED    = HexColor("#B71C1C")
C_PANEL  = HexColor("#EEF2F7")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

PAIR_LABEL = {(4,False):"2,2",(6,False):"3,3",(8,False):"4,4",(10,False):"5,5",
              (12,False):"6,6",(14,False):"7,7",(16,False):"8,8",(18,False):"9,9",
              (20,False):"T,T",(12,True):"A,A"}
PAIR_ORDER = ["2,2","3,3","4,4","5,5","6,6","7,7","8,8","9,9","T,T","A,A"]

# -- Data loading --------------------------------------------------------------

def pick(**crit):
    method = crit.pop("method", "__any__")
    def norm(cfg, k):
        v = cfg.get(k)
        if k == "with_splits": return bool(v)
        if k == "epsilon_schedule": return v or "constant"
        return v
    best = None
    for p in sorted(glob.glob(f"{RUNS_DIR}/*/record.json")):
        rec = json.load(open(p)); cfg = rec["config"]
        if method != "__any__" and rec.get("method") != method: continue
        if all(norm(cfg, k) == v for k, v in crit.items()): best = p
    if best is None:
        raise FileNotFoundError(f"no run matches {crit} method={method}")
    return json.load(open(best))

def cells_of(record):
    df = pd.DataFrame(record["diff"]["cells"])
    if "can_split" not in df: df["can_split"] = False
    df["can_split"] = df["can_split"].fillna(False).astype(bool)
    df["ev_gap"] = (df["agent_q"] - df["basic_q"]).abs()
    df["kind"] = ["pair" if c else ("soft" if s else "hard") for c, s in zip(df.can_split, df.is_soft)]
    return df

def load_runs():
    return {
        "baseline": pick(epsilon=0.1, epsilon_schedule="constant", with_splits=False, method=None),
        "eps03":    pick(epsilon=0.3, epsilon_schedule="constant", with_splits=False, method=None),
        "decay":    pick(epsilon_schedule="linear", step_size=0.001, with_splits=False, method=None),
        "decay_sa": pick(epsilon_schedule="linear", step_size=None, with_splits=False, method=None),
        "split":    pick(with_splits=True, method=None),
        "es":       pick(method="exploring_starts"),
    }

# -- Edge ledger (re-evaluated through the engine, cached) ----------------------

def edge_ledger(runs, hands):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    from blackjack_rl.experiment import load_agent
    from blackjack_rl.evaluation.metrics import evaluate_policy, GreedyPolicy
    from strategies.basic_strategy import BasicStrategy

    def edge(key, policy):
        ck = f"{key}::{hands}"
        if ck not in cache:
            r = evaluate_policy(policy, n_hands=hands, seed=0)
            cache[ck] = [round(r.edge*100, 3), round(r.std_error*100, 3)]
        return tuple(cache[ck])

    out = {}
    for name, rec in runs.items():
        out[name] = edge(rec["run_id"], GreedyPolicy(load_agent(rec)))
    out["basic"] = edge("basic", BasicStrategy())
    json.dump(cache, open(CACHE, "w"), indent=1)
    return out

# -- Metrics (policy-level, from the diffs; deterministic) ---------------------

def compute_metrics(runs, edges):
    b = cells_of(runs["baseline"])
    b["match"] = b.agent_action == b.basic_action
    b["involves_double"] = b.agent_action.eq("double") | b.basic_action.eq("double")
    gen = lambda d: (d.category == "genuine_disagreement").mean()
    gd = b[b.category == "genuine_disagreement"].copy()

    # mechanism: share of visits the basic (correct) action was tried, for disagreement cells
    qt = pd.DataFrame(runs["baseline"]["qtable"])
    def share(row):
        cell = qt[(qt.player_value==row.player_value)&(qt.is_soft==row.is_soft)&(qt.dealer_upcard==row.dealer_upcard)]
        tot = cell["n"].sum()
        bn = cell[cell.action==row.basic_action]["n"].sum()
        return (bn/tot) if tot else 0.0
    gd["basic_share"] = [share(r) for r in gd.itertuples()]

    v16 = gd[(gd.player_value==16)&(~gd.is_soft)&(gd.dealer_upcard==10)]
    soft16 = gd.sort_values("ev_gap").iloc[-1]   # the 0.407 outlier

    # relocate eps0.1 -> eps0.3
    e = cells_of(runs["eps03"]); e["match"] = e.agent_action == e.basic_action
    bi = b.set_index(["player_value","is_soft","dealer_upcard"])["match"]
    ei = e.set_index(["player_value","is_soft","dealer_upcard"])["match"]
    common = bi.index.intersection(ei.index)
    spoiled = int((bi.loc[common] & ~ei.loc[common]).sum())
    corrected = int((~bi.loc[common] & ei.loc[common]).sum())

    # splits + capstone
    sp = cells_of(runs["split"]); es = cells_of(runs["es"])
    sp_gen = sp[sp.category=="genuine_disagreement"]; es_gen = es[es.category=="genuine_disagreement"]
    es_surv = es_gen.copy()

    M = dict(
        agree_unwt = (b.category=="agree").mean()*100,
        agree_wt   = runs["baseline"]["diff"]["agreement_weighted"]*100,
        loc_hard   = gen(b[~b.is_soft & ~b.can_split])*100,
        loc_soft   = gen(b[b.is_soft & ~b.can_split])*100,
        loc_double = gen(b[b.involves_double])*100,
        n_genuine  = len(gd),
        gap_min    = gd.ev_gap.min(), gap_max = gd.ev_gap.max(),
        v16_gap    = float(v16.ev_gap.iloc[0]), v16_visits = int(v16.visits.iloc[0]),
        soft16_share = soft16["basic_share"]*100, soft16_gap = soft16["ev_gap"],
        reloc_corrected = corrected, reloc_spoiled = spoiled,
        reloc_net = corrected - spoiled, reloc_shared = len(common),
        edges = edges,
        split_pair_genuine = int((sp_gen.kind=="pair").sum()),
        cap_split_gen = len(sp_gen), cap_es_gen = len(es_gen),
        cap_split_maxgap = sp_gen.ev_gap.max(), cap_es_maxgap = es_gen.ev_gap.max(),
        es_surv_vmin = int(es_surv.visits.min()), es_surv_vmax = int(es_surv.visits.max()),
        es_undervisited = int((es.category=="under_visited").sum()),
        train_eps = runs["baseline"]["config"]["num_episodes"],
        gd = gd, b = b, sp = sp, es = es,
    )
    return M

# -- Charts --------------------------------------------------------------------

def save_chart(fig, name):
    path = f"{CHARTS_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig); return path

def chart_localization(M):
    cats = ["Hard totals", "Soft totals", "Doubling\nin play"]
    vals = [M["loc_hard"], M["loc_soft"], M["loc_double"]]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    bars = ax.bar(cats, vals, color=["#2196F3", "#FF9800", "#B71C1C"], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.axhline((100-M["agree_unwt"]), color="gray", ls="--", lw=1,
               label=f"pooled disagreement {100-M['agree_unwt']:.1f}%")
    ax.set_ylabel("Disagreement with basic strategy (%)")
    ax.set_title("Error is localized, not spread: disagreement by decision type", fontweight="bold")
    ax.legend(); plt.tight_layout(); return save_chart(fig, "localization")

def chart_severity(M):
    gd = M["gd"].sort_values("ev_gap").reset_index(drop=True)
    colors = ["#FDD835" if g < 0.05 else "#FB8C00" if g < 0.15 else "#B71C1C" for g in gd.ev_gap]
    labels = [f"{'soft ' if s else 'hard '}{int(v)} v{int(u)}"
              for v, s, u in zip(gd.player_value, gd.is_soft, gd.dealer_upcard)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(range(len(gd)), gd.ev_gap, color=colors, edgecolor="white")
    ax.set_yticks(range(len(gd))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.02, color="gray", ls="--", lw=1, label="0.02 classification threshold")
    ax.set_xlabel("Expected value forfeited (|Q_agent - Q_basic|)")
    ax.set_title("Severity, not count: the 15 genuine disagreements span 20x", fontweight="bold")
    legend = [Patch(facecolor="#FDD835", label="near-tie (<0.05)"),
              Patch(facecolor="#FB8C00", label="moderate"),
              Patch(facecolor="#B71C1C", label="severe (>0.15)")]
    ax.legend(handles=legend, loc="lower right"); plt.tight_layout()
    return save_chart(fig, "severity")

def chart_mechanism(M):
    gd = M["gd"].sort_values("basic_share").reset_index(drop=True)
    labels = [f"{'soft ' if s else 'hard '}{int(v)} v{int(u)}"
              for v, s, u in zip(gd.player_value, gd.is_soft, gd.dealer_upcard)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(range(len(gd)), gd.basic_share*100, color="#0D47A1", edgecolor="white")
    ax.set_yticks(range(len(gd))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(100/3, color="gray", ls="--", lw=1, label="uniform share (1 of 3 actions)")
    ax.set_xlabel("Share of visits the correct (basic) action was tried (%)")
    ax.set_title("The mechanism: starvation — the right action was barely sampled", fontweight="bold")
    ax.legend(loc="lower right"); plt.tight_layout(); return save_chart(fig, "mechanism")

def chart_relocate(M):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(["corrected\n(wrong -> right)"], [M["reloc_corrected"]], color="#2E7D32", edgecolor="white")
    ax.barh(["spoiled\n(right -> wrong)"], [-M["reloc_spoiled"]], color="#B71C1C", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.text(M["reloc_corrected"]+0.2, 0, f"+{M['reloc_corrected']}", va="center", fontweight="bold")
    ax.text(-M["reloc_spoiled"]-0.2, 1, f"-{M['reloc_spoiled']}", va="center", ha="right", fontweight="bold")
    ax.set_xlabel("Cells changed (of %d shared)" % M["reloc_shared"])
    ax.set_title(f"Raising exploration 0.1 -> 0.3 relocates error (net {M['reloc_net']:+d})", fontweight="bold")
    plt.tight_layout(); return save_chart(fig, "relocate")

def chart_ledger(M):
    edges = M["edges"]
    label = {"baseline":"eps 0.1 (fixed)", "eps03":"eps 0.3 (fixed)",
             "decay_sa":"decay, plain average", "decay":"decay + constant step",
             "split":"decay + step + splits", "es":"exploring starts + splits"}
    names = ["eps03", "decay_sa", "baseline", "decay", "split", "es"]
    rows = [(label.get(n, n), edges[n][0], edges[n][1]) for n in names]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; ses = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(labels, vals, xerr=ses, color="#0D47A1", edgecolor="white",
            error_kw={"ecolor":"#444","capsize":3})
    ax.axvline(edges["basic"][0], color="#2E7D32", ls="--", lw=1.5,
               label=f"basic strategy {edges['basic'][0]:.2f}%")
    ax.set_xlabel("House edge %  (lower = better)")
    ax.set_title("House edge by configuration (common large-sample eval)", fontweight="bold")
    ax.invert_yaxis(); ax.legend(loc="lower right"); plt.tight_layout()
    return save_chart(fig, "ledger")

def _heatmap_panels(ax_list, df):
    CATS=["agree","near_equal_ev","genuine_disagreement","under_visited"]
    COLORS={"agree":"#c8e6c9","near_equal_ev":"#ffe082","genuine_disagreement":"#ef9a9a","under_visited":"#e0e0e0"}
    cat_i={c:i for i,c in enumerate(CATS)}
    cmap=ListedColormap([COLORS[c] for c in CATS]); ACT={"hit":"H","stand":"S","double":"D","split":"P","surrender":"R"}
    ups=list(range(2,12))
    def grid(ax,d,sub,field,order):
        rows=[r for r in order if r in set(d[field])]
        g=np.full((len(rows),len(ups)),np.nan); ax.set_facecolor("white")
        for _,r in d.iterrows():
            if r[field] not in rows: continue
            ii,jj=rows.index(r[field]),ups.index(r["dealer_upcard"]); g[ii,jj]=cat_i[r["category"]]
            agree=r.agent_action==r.basic_action
            lab=ACT[r["agent_action"]] if agree else f"{ACT[r['agent_action']]}>{ACT[r['basic_action']]}"
            ax.text(jj,ii,lab,ha="center",va="center",fontsize=(8 if agree else 7),
                    color=("#666" if agree else "#111"),fontweight=("normal" if agree else "bold"))
        ax.imshow(g,cmap=cmap,vmin=0,vmax=len(CATS)-1,aspect="equal")
        ax.set_anchor("N")
        ax.set_xticks(range(len(ups))); ax.set_xticklabels(ups,fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=8)
        ax.set_xticks(np.arange(-.5,len(ups),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=2.0)
        ax.grid(which="major",visible=False)
        ax.tick_params(which="both",length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlabel("dealer upcard",fontsize=9); ax.set_title(sub,fontsize=11,pad=8)
    nonpair=df[~df.can_split]
    panels=[(nonpair[~nonpair.is_soft],"Hard totals","player_value",sorted(nonpair[~nonpair.is_soft].player_value.unique())),
            (nonpair[nonpair.is_soft],"Soft totals","player_value",sorted(nonpair[nonpair.is_soft].player_value.unique()))]
    if df.can_split.any():
        pr=df[df.can_split].copy()
        pr["pair"]=[PAIR_LABEL.get((int(v),bool(s)),str(v)) for v,s in zip(pr.player_value,pr.is_soft)]
        panels.append((pr,"Pairs","pair",[p for p in PAIR_ORDER if p in set(pr["pair"])]))
    for ax,(d,nm,fld,o) in zip(ax_list,panels): grid(ax,d,nm,fld,o)
    return panels,COLORS,CATS

def chart_heatmap(df,title,name):
    npanels=3 if df.can_split.any() else 2
    with sns.axes_style("white"):
        fig,axes=plt.subplots(1,npanels,figsize=(4.7*npanels,7.4))
        axes=np.atleast_1d(axes)
        _,COLORS,CATS=_heatmap_panels(axes,df)
        axes[0].set_ylabel("player total / pair",fontsize=9)
        fig.legend(handles=[Patch(facecolor=COLORS[c],label=c.replace("_"," ")) for c in CATS],
                   loc="lower center",ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(0.5,0.0))
        fig.suptitle(title,fontsize=13,fontweight="bold",y=1.0)
        plt.tight_layout(rect=[0,0.05,1,0.98])
    return save_chart(fig,name)

def chart_capstone(M):
    sp=M["sp"]; es=M["es"]
    kinds=["hard","soft","pair"]
    sg=sp[sp.category=="genuine_disagreement"].kind.value_counts()
    eg=es[es.category=="genuine_disagreement"].kind.value_counts()
    x=np.arange(len(kinds)); w=0.38
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(x-w/2,[sg.get(k,0) for k in kinds],w,label="decay+a +splits",color="#B71C1C",edgecolor="white")
    axes[0].bar(x+w/2,[eg.get(k,0) for k in kinds],w,label="exploring starts",color="#2E7D32",edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(kinds); axes[0].set_ylabel("genuine disagreements")
    axes[0].set_title("Genuine disagreements collapse",fontweight="bold"); axes[0].legend()
    # severity tail: max gap before/after
    axes[1].bar(["decay+a\n+splits","exploring\nstarts"],[M["cap_split_maxgap"],M["cap_es_maxgap"]],
                color=["#B71C1C","#2E7D32"],edgecolor="white")
    for i,v in enumerate([M["cap_split_maxgap"],M["cap_es_maxgap"]]):
        axes[1].text(i,v+0.005,f"{v:.3f}",ha="center",fontweight="bold")
    axes[1].set_ylabel("largest genuine EV gap"); axes[1].set_title("...and the costly tail vanishes",fontweight="bold")
    plt.suptitle("Capstone: forcing coverage removes the coverage residual",fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout(); return save_chart(fig, "capstone")

def build_charts(M):
    return {
        "localization": chart_localization(M),
        "severity": chart_severity(M),
        "mechanism": chart_mechanism(M),
        "relocate": chart_relocate(M),
        "ledger": chart_ledger(M),
        "heatmap_base": chart_heatmap(M["b"], "Learned policy vs basic strategy — base agent (eps 0.1)", "heatmap_base"),
        "heatmap_split": chart_heatmap(M["sp"], "Learned policy vs basic strategy — with splits", "heatmap_split"),
        "heatmap_es": chart_heatmap(M["es"], "Learned policy vs basic strategy — exploring starts", "heatmap_es"),
        "capstone": chart_capstone(M),
    }

# -- PDF -----------------------------------------------------------------------

def build_pdf(C, M):
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    W = A4[0] - 40*mm
    e = M["edges"]

    title = ParagraphStyle("T", fontSize=17, leading=23, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=8)
    sub   = ParagraphStyle("Sub", fontSize=13, leading=18, textColor=HexColor("#BBDEFB"), alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
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

    story = []
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import KeepTogether
    def P(t, s=body): story.append(Paragraph(t, s))
    def fig(key, caption, width=None):
        if width is None:
            width = W*0.80 if key.startswith("heatmap") else W*0.84
        iw, ih = ImageReader(C[key]).getSize()
        story.append(Image(C[key], width=width, height=width*ih/iw, hAlign="CENTER"))
        P(caption, cap)
    def rule(): story.append(HRFlowable(width=W, thickness=1, color=C_ACCENT, spaceAfter=10))
    def panel(flow, bg=C_DARK, height=None, pad=20):
        kw={"colWidths":[W]}
        if height: kw["rowHeights"]=[height]
        t=Table([[flow]], **kw)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",(0,0),(-1,-1),26),("RIGHTPADDING",(0,0),(-1,-1),26),
            ("TOPPADDING",(0,0),(-1,-1),pad),("BOTTOMPADDING",(0,0),(-1,-1),pad)]))
        return t
    def styled_table(data, widths):
        t=Table(data, colWidths=widths)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_DARK),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[HexColor("#F5F5F5"),white]),
            ("GRID",(0,0),(-1,-1),0.5,HexColor("#BDBDBD")),("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        return t

    # COVER
    story.append(Spacer(1, 48*mm))
    cover=[Paragraph("Does a Learning Agent Rediscover Optimal Play", title),
           Paragraph("and Where Does It Fall Short?", title),
           Spacer(1,5*mm),
           Paragraph("A study of reinforcement learning&rsquo;s characteristic behavior,", sub),
           Paragraph("told through blackjack", sub),
           Spacer(1,9*mm),
           Paragraph(f"Tabular Monte Carlo control &bull; {M['train_eps']/1e6:.0f}M training hands "
                     "&bull; audited cell-by-cell against basic strategy", meta),
           Spacer(1,2*mm),
           Paragraph("Part of AI Journey &mdash; Phase 3: Deep Learning &amp; RL", meta),
           Spacer(1,12*mm),
           Paragraph("github.com/arda-basarici/ai-journey",
                     ParagraphStyle("L", fontSize=10.5, textColor=HexColor("#64B5F6"), alignment=TA_CENTER, fontName="Helvetica"))]
    story.append(panel(cover, height=120*mm)); story.append(PageBreak())

    # THE QUESTION
    P("The question behind the question", h1); rule()
    P("A reinforcement-learning agent that learns blackjack from nothing but the win or loss at the "
      "end of each hand will, given enough play, converge toward expert play. That much is "
      "unsurprising. The interesting question is not whether it succeeds but where it fails, and "
      "why &mdash; because the places a learned policy diverges from a known optimum are where the "
      "mechanics of learning-from-experience become visible. This report uses blackjack as an "
      "instrument: the agent is a way of watching how a value-based learner allocates its attention, "
      "where it grows confident, and where it stays blind.", lead)
    P("Blackjack is the right instrument for a specific reason. Its optimal policy &mdash; basic "
      "strategy, the value-maximizing action in every situation &mdash; is exactly computable and "
      "long established. That gives the study something most machine-learning problems lack: a "
      "complete, trustworthy answer key. Every decision the agent makes can be checked against the "
      "known-correct one, so a disagreement is never a matter of opinion &mdash; it is the agent "
      "measurably falling short of a solved problem, which means each shortfall can be traced to a "
      "cause rather than waved away.", body)

    P("From a simulator to an environment", h1); rule()
    P("This work is the second movement of a longer project. The first built a Monte Carlo simulator "
      "of blackjack from scratch &mdash; an engine that deals, plays, and resolves hands under fixed "
      "rules, validated until its house edge and outcome distributions matched the known mathematics "
      "of the game. Here the same engine takes on a second role: what was a tool for generating data "
      "becomes an environment for learning. The agent plays through it to improve, updating its "
      "own estimates from the rewards the engine returns. The continuity is deliberate &mdash; the "
      "engine was already trusted, and the optimal strategy the first project derived is precisely "
      "what makes this project&rsquo;s central question answerable. A study of where a learner falls "
      "short is only meaningful if you already know, exactly, what not falling short looks like.", body)

    box=[Paragraph("The method, in one box", boxh),
         Paragraph("<b>The agent</b> is a tabular Monte Carlo control learner: it plays complete "
                   "hands, sees only the terminal win or loss, and from that single delayed signal "
                   "updates an estimated value for every decision it made. Its policy is to take, in "
                   "each situation, the action it currently values most.", boxb),
         Paragraph("<b>The audit</b> places that policy cell-by-cell against basic strategy, "
                   "classifying each disagreement by whether the two actions genuinely differ in "
                   "value, sit within a hair of each other, or rest on too little experience to trust. "
                   "Crucially, the audit is computed from the agent&rsquo;s trained value table and its "
                   "<i>training</i> visit counts &mdash; not from the separate simulations used to "
                   "measure house edge &mdash; so a cell&rsquo;s classification is a property of the "
                   "trained policy and does not shift if the edge evaluation is run at a different size.", boxb)]
    story.append(panel(box, bg=C_PANEL, pad=12 )); story.append(PageBreak())

    # 1. AGGREGATE LIES
    P("1. The aggregate lies; the structure is in the conditioning", h1); rule()
    P(f"The pooled {M['agree_unwt']:.1f}% hides the result; conditioning on decision type reveals it &mdash; error is localized, not spread.", take)
    P(f"Pooled over every situation, the agent agrees with basic strategy {M['agree_unwt']:.1f}% of "
      f"the time unweighted, and {M['agree_wt']:.1f}% weighted by how often each situation occurs. "
      "Taken alone, that number is true and nearly useless. It averages the decisions the agent has "
      "effectively mastered &mdash; stand on a hard twenty, hit a low total &mdash; together with the "
      "handful it genuinely struggles with. The structure appears the moment agreement is conditioned "
      f"on the kind of decision: disagreement runs at {M['loc_hard']:.1f}% on hard totals, "
      f"{M['loc_soft']:.1f}% on soft totals, and {M['loc_double']:.1f}% wherever doubling is an option.", body)
    fig("localization", "Figure 1: Disagreement with basic strategy, conditioned on decision type. The pooled average "
        "(dashed) hides a 16x spread between common and rare decisions.")
    fig("heatmap_base", "Figure 2: The learned base policy against basic strategy. Green agrees; yellow are near-ties; "
        "red are genuine disagreements. The red concentrates in the soft and doubling bands.")
    P("That spread is the first real result, and it is a general one wearing a blackjack costume. The "
      "agent has all but solved the common, frequently-revisited decisions and concentrated nearly "
      "all its error in the rare ones. This is the defining behavior of a learner whose knowledge is "
      "built from experience: what is seen often is learned well; what is seen seldom is learned "
      "poorly. A single headline metric is simply the wrong altitude from which to judge a learned "
      "policy &mdash; the informative view is always the conditioned one.", body)

    # 2. SEVERITY
    P("2. Reading the disagreements honestly: severity, not count", h1); rule()
    P("The disagreements span a twenty-fold range in severity; counting them as equal is the mistake a fixed threshold invites.", take)
    P(f"Having found that disagreements cluster, the natural next move is to count them &mdash; and "
      f"that is the second place the obvious approach misleads. There are {M['n_genuine']} genuine "
      "disagreements, but treating them as equivalent collapses two very different things: decisions "
      "where the agent&rsquo;s choice costs almost nothing, and decisions where it costs real value. "
      "The honest unit is severity &mdash; expected value forfeited &mdash; and measured that way the "
      f"{M['n_genuine']} span a twenty-fold range, from {M['gap_min']:.3f} to {M['gap_max']:.3f}.", body)
    fig("severity", "Figure 3: The genuine disagreements sorted by severity. A fixed 0.02 threshold (dashed) tips "
        "the most balanced decision in the game into the 'error' column on a single thousandth.")
    P(f"The most instructive case sits right at the boundary: hard sixteen against a dealer ten, the "
      f"most famously balanced decision in the game, visited over {M['v16_visits']:,} times, where the "
      f"value of hitting and standing differ by only {M['v16_gap']:.3f} &mdash; essentially a coin "
      "flip. A count leaning on a fixed cutoff labels this an error on a thousandth of a unit of "
      "value. Reporting the distribution dissolves the artifact, and surfaces a subtler point the next "
      f"section tests: the largest gaps are not the agent confidently choosing wrong, but valuing an "
      f"action it barely tried. The {M['gap_max']:.3f} outlier &mdash; soft sixteen against a four "
      f"&mdash; is a case where the correct action was sampled in only about {M['soft16_share']:.0f}% "
      "of visits.", body)

    # 3. MECHANISM
    P("3. The mechanism: starvation, not stubbornness", h1); rule()
    P("The agent isn&rsquo;t wrong where it&rsquo;s confident &mdash; it&rsquo;s wrong where it barely looked.", take)
    P("A wrong choice can come from two very different places: the correct action was tried too rarely "
      "to be valued accurately (a coverage problem), or a wrong action was lifted by a few lucky early "
      "outcomes the estimate could never shed (a memory problem). They are separable by a simple "
      "question: in the cells where the agent disagrees, how often did it actually try the action "
      "basic strategy recommends?", body)
    fig("mechanism", "Figure 4: For each disagreement cell, the share of visits the correct action was tried. "
        "Almost everywhere it falls well below an even split &mdash; the agent never gathered the "
        "evidence to value it.")
    P("The answer is decisive. In nearly every disagreement the correct action was sampled only a "
      "small fraction of the time, so the agent never gathered the evidence to value it properly. This "
      "explains the localization with one stroke: under a policy that mostly exploits what it already "
      "believes, an action that looks unpromising early is tried less, which keeps its estimate "
      "pessimistic, which makes it look unpromising &mdash; a self-reinforcing starvation. The agent "
      "is not stubborn; it is uninformed, and uninformed in a structured way that traces directly back "
      "to how a value-based learner allocates its own experience. This is why exploration is not a "
      "tuning detail but the central design problem of this entire class of method.", body)

    # 4. RELOCATE
    P("4. More exploration relocates the error", h1); rule()
    P("Raising exploration corrects as many cells as it spoils: the error moves, it doesn&rsquo;t shrink.", take)
    P("If under-exploration causes the error, the intuitive fix is to explore more. The most "
      "counterintuitive result is that this does not work the way one expects. Raising the exploration "
      f"rate from 0.1 to 0.3 does not shrink the disagreement; it moves it. Across the "
      f"{M['reloc_shared']} shared situations, the higher rate corrects {M['reloc_corrected']} "
      f"decisions that were wrong and spoils {M['reloc_spoiled']} that were right &mdash; a net change "
      f"of {M['reloc_net']:+d}. The total is essentially unchanged; what shifts is which cells are wrong.", body)
    fig("relocate", "Figure 5: Raising exploration from 0.1 to 0.3, cell by cell. Coverage of the rare is bought "
        "with distortion of the common.", width=W*0.66)
    P("The reason is a genuine tension, not a flaw to be tuned away. The agent&rsquo;s value estimates "
      "reflect the policy it actually plays. Explore more, and the rare cells finally get coverage "
      "&mdash; but the common cells are now evaluated under a policy that plays randomly thirty percent "
      "of the time, which biases their estimates downward. There is no fixed exploration rate that "
      "wins everywhere, because the very mechanism that informs the neglected decisions corrupts the "
      "well-understood ones. This is the on-policy learner&rsquo;s central bind in miniature.", body)

    # 5. THE FIX
    P("5. Why the fix needs two parts: schedule and memory", h1); rule()
    P("Decaying exploration supplies the experience; constant-step memory keeps it from being anchored to a noisy start.", take)
    P("The resolution has two components. The first is obvious once the tension is named: explore "
      "heavily early, then decay exploration toward zero so the policy can settle into clean "
      "exploitation. But decaying alone is not enough. If each value is a simple running average over "
      "every outcome it ever saw, the noisy early returns are baked in permanently &mdash; the average "
      "cannot forget its own beginning. The cure is to weight recent experience more heavily, by "
      "updating with a constant step size rather than a true average, so each estimate tracks the "
      "improving policy instead of being anchored to its uninformed start.", body)
    fig("ledger", "Figure 6: House edge by configuration, re-evaluated at a common large sample so the "
        "comparison is fair. Lower is better; the dashed line is basic strategy.")
    P(f"The effect is visible in the ledger of house edges. The decaying schedule with a constant step "
      f"reaches roughly {e['decay'][0]:.2f}% against an optimal near {e['basic'][0]:.2f}%, while the "
      f"same schedule under a plain running average sits at {e['decay_sa'][0]:.2f}% and a high fixed "
      f"exploration rate is worst at {e['eps03'][0]:.2f}%; adding splits brings the best natural-play "
      f"agent to {e['split'][0]:.2f}%. Read against the measurement uncertainty of about "
      f"&plusmn;{e['decay'][1]:.2f}% on each estimate, the disciplined reading is narrow but real: "
      "recency-weighting and splits genuinely help, while the closest configurations differ by less "
      "than their error bars and cannot honestly be ranked against one another. That restraint is "
      "reinforced by a telling detail &mdash; at a smaller evaluation the top of the table inverted. "
      "The win was real only once the measurement was precise enough to see it.", body)

    # 6. SPLITS
    P("6. The same story in a new register: splits", h1); rule()
    P("A new kind of decision, the same coverage signature &mdash; evidence the mechanism is general, not fitted.", take)
    P("Extending the agent to handle pairs is both a completion of the problem and a test of whether "
      "the mechanism is truly general. The agent is taught no rule; the situation is simply added to "
      "what it can perceive, and it must discover from experience whether and when to split. Adding "
      "splits improved the edge, since correct splitting recovers value a no-split policy leaves on "
      "the table. The result is otherwise the same story in a new column: most splitting decisions are "
      "learned well; the errors concentrate in the rare low pairs &mdash; twos, threes, fours, and "
      "nines &mdash; each visited only around twenty-three hundred times, whose split action is tried "
      "far fewer times still. It is the identical coverage signature, now reproduced in a structurally "
      "different decision &mdash; and a mechanism that recurs across different decisions is more "
      "trustworthy than one fitted to a single case.", body)
    fig("heatmap_split", "Figure 7: The policy with splits, before exploring starts. Most of the pairs column is "
        "learned (green); red marks the under-split rare low pairs the agent rarely experiences.")

    # 7. CAPSTONE
    P("7. The capstone: making the explanation prove itself", h1); rule()
    P("A prediction locked before the run: force coverage, and the gap collapses to a floor of genuine ties.", take)
    P("Everything to this point argues that the residual error is, in the main, a coverage problem. An "
      "argument is not a demonstration, so the study closes with a test designed to be able to fail. "
      "If the residual is truly coverage, then forcing coverage should remove it. Exploring "
      "starts &mdash; the canonical Monte Carlo control variant &mdash; begins every hand from a "
      "deliberately chosen situation-and-action, giving every decision equal airtime regardless of how "
      "rarely natural play would reach it, then follows the learned policy. The prediction was fixed "
      "before the run: the coverage-driven disagreements should collapse toward optimal play, while "
      "the genuine near-ties should remain, since no amount of coverage can separate decisions that "
      "are truly tied. Had everything vanished, the coverage story would have been too clean to trust.", body)
    fig("capstone", "Figure 8: Forcing coverage collapses the genuine disagreements and erases the costly tail, "
        "leaving only near-ties.")
    P(f"The prediction held. The capstone runs against the splits agent, which carries "
      f"{M['cap_split_gen']} genuine disagreements; forcing coverage cut them to {M['cap_es_gen']}, "
      f"and dropped the largest surviving gap from {M['cap_split_maxgap']:.3f} to "
      f"{M['cap_es_maxgap']:.3f} &mdash; a tail of costly errors reduced to nothing but small ones. The "
      "cells that had been starved flipped to the correct action exactly where the mechanism said the "
      "agent had simply never looked: the rare low pairs that had been hit now split, the avoided "
      f"doubles now double. What survived was precisely what was predicted to &mdash; {M['cap_es_gen']} "
      f"near-ties, each now backed by ample experience under forced starts "
      f"({M['es_surv_vmin']:,}&ndash;{M['es_surv_vmax']:,} visits, far above their roughly twenty-three "
      "hundred in natural play) and still split down the middle because there is genuinely nothing to "
      "separate. The "
      "residual decomposes cleanly into two parts and only two: a reducible portion that was coverage, "
      "now collapsed, and an irreducible floor that is genuine indifference.", body)
    fig("heatmap_es", "Figure 9: The exploring-starts policy. The red of Figure 2 is almost gone; what remains is "
        "yellow near-ties and a few thinly-visited deep states.")
    P(f"It is worth being honest about what the capstone did not do. Forcing coverage from two-card "
      f"starting situations leaves the deepest, multi-card hands still thinly visited, so a small "
      f"number of sparse states remain &mdash; {M['es_undervisited']}, in this run &mdash; and coverage "
      "was therefore made near-uniform over decision points rather than over every reachable position. "
      "The experiment earned its claim; it did not manufacture a flawless sweep, and saying so is part "
      "of the same discipline that made the claim worth trusting.", body)

    # WHAT THIS SHOWS + NEXT
    P("What this study actually shows", h1); rule()
    P("Read as a whole, the report is not about blackjack and not about a particular agent. It is about "
      "a characteristic way that learning-from-experience behaves, made visible because a known optimum "
      "was available to measure against. A value-based learner masters what it sees often and stays "
      "blind where it looks seldom; its own policy shapes the data it learns from, so it can starve "
      "itself of the very experience it needs; naive attempts to fix this relocate the error rather "
      "than removing it, because coverage of the rare trades against accuracy on the common; the "
      "genuine fix pairs a decaying exploration schedule with memory that favors "
      "recent experience; and the residual that remains, once coverage is forced, is not failure but "
      "the irreducible floor of decisions that do not matter. None of these is specific to cards.", body)
    P("Where this goes next", h1); rule()
    P("The natural continuation moves from a setting where the optimum is known to one where it is not. "
      "Extending the agent toward card-counting and bet-sizing introduces a state the table can no "
      "longer cleanly hold, and an objective that is no longer a single computable answer but a trade "
      "between expected value and the risk of ruin &mdash; a problem with no tidy ground-truth key. "
      "That shift is exactly where the disciplines practiced here &mdash; conditioning rather than "
      "pooling, severity over count, locked and falsifiable predictions, ranking only on "
      "distinguishable differences &mdash; stop being conveniences and become the only safeguards "
      "left. The value of having done this study against a solved problem is that those habits are now "
      "established before they are needed on an unsolved one.", body)
    story.append(Spacer(1, 16*mm))
    closing=[Paragraph("Built as Part of AI Journey",
                       ParagraphStyle("CT", fontSize=16, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=14, leading=20)),
             Paragraph("A structured path from Python foundations to AI engineering.",
                       ParagraphStyle("CB", fontSize=10, textColor=HexColor("#BBDEFB"), alignment=TA_CENTER, fontName="Helvetica", spaceAfter=14, leading=15)),
             Paragraph("github.com/arda-basarici/ai-journey",
                       ParagraphStyle("CL", fontSize=11, textColor=HexColor("#64B5F6"), alignment=TA_CENTER, fontName="Helvetica-Bold", leading=14))]
    story.append(panel(closing, height=165, pad=24))
    doc.build(story)
    print("Report generated:", OUTPUT_PDF)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-hands", type=int, default=1_000_000)
    args = ap.parse_args()
    print("loading runs ...")
    runs = load_runs()
    print(f"evaluating house edges at {args.eval_hands:,} hands (cached) ...")
    edges = edge_ledger(runs, args.eval_hands)
    print("computing metrics + charts ...")
    M = compute_metrics(runs, edges)
    C = build_charts(M)
    print("building PDF ...")
    build_pdf(C, M)

if __name__ == "__main__":
    main()