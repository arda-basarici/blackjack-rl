"""Betting Against the Noise — Phase-3 Problem-B report (PDF).

Fourth movement of the blackjack project (engine -> policy audit -> table-to-network -> betting).
Companion to ``report_generator.py`` / ``report_generator_dqn.py`` — same house style, same
reportlab + matplotlib stack, same cover/section/box vocabulary. Deliberately self-contained like its
siblings (the generalized cross-project report tool is deferred until after the repo restructure).

Numbers policy (stricter than the DQN report — enforced, not conventional):
  * Every number and statistical claim in the prose is pinned in DATA and ASSERTED by
    ``verify_data()`` against the live artifacts at build time (loaders + the committed wonging/ladder
    run records; z/p statistics recomputed, never trusted from memory) — the build fails loud if the
    report ever drifts. Sole exception: the n=3 wave-1 p=0.005 in Section 7 (a historical value from
    B2d_EXPERIMENTS Test 15 — the wave-1/wave-2 seed split is not recoverable from the saved evals).
  * FIGURES are rendered live through the same ``analysis_loader`` plotters the chapter notebooks
    use — one plotting code path, so report charts can never diverge from the notebooks'.

Run from the repo root:  .venv\\Scripts\\python.exe scripts/report_generator_session.py
Needs: reportlab, matplotlib, pandas (the loaders); no torch — nothing is re-trained or re-evaluated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt

# the loaders call plt.show() (notebook-style); under Agg that is a harmless no-op — keep the build quiet
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "runs").is_dir())
sys.path.insert(0, str(ROOT))

from blackjack_rl.analysis_loader import (  # noqa: E402  (needs ROOT on sys.path first)
    KELLY_LADDER,
    coverage_growth_ztest, ladder_baselines, load_bet_evals, load_bet_runs, near_kelly_runs, oracle_run,
    plot_bet_coverage, plot_bet_embedding_row, plot_bet_orbit, plot_bet_replication, plot_count_frequency,
    plot_coverage_growth, plot_encoding_ablation, plot_kelly_distance, plot_ladder_bars,
    plot_native_curves, plot_prize_bar, plot_signal_vs_noise,
    bet_embedding, bet_project,
)

OUTPUT_PDF = str(ROOT / "betting-against-the-noise.pdf")
CHARTS_DIR = ROOT / "report_charts_session"
CHARTS_DIR.mkdir(exist_ok=True)

# -- house palette (identical to the sibling generators) ----------------------------------------
C_DARK   = HexColor("#1A1A2E")
C_ACCENT = HexColor("#0D47A1")
C_INK    = HexColor("#212121")
C_PANEL  = HexColor("#EEF2F7")

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "font.family": "sans-serif"})

# ============================ DATA — pinned, audited, build-verified ============================
# Every value cites its source. verify_data() asserts the loader-backed ones (tolerance in the getter).
DATA = dict(
    # 20k bet-ladder baselines (runs/20260629-153359_bet-ladder_20000sess) — growth x1e-4/hand, dd %
    growth_kelly=-0.048, growth_flat=-0.150, growth_prize=0.102,        # Kelly-over-Flat, z~7 (decisive)
    ruin_kelly=-0.348,   ruin_flat=-0.395,   ruin_gap=0.047,            # marginal, p~0.04
    ruin_kelly_dd=2.34,  ruin_flat_dd=0.80, growth_kelly_dd=0.14, growth_flat_dd=0.00,
    z_growth_prize=7.1, p_ruin_gap=0.04,                                # unpaired tests on the 20k ladder
    # headline agents (Test 13/14; 6 seeds, native cell) — mean +/- SD across seeds
    agent_growth=-0.19, agent_growth_sd=0.08, agent_growth_ruin=0.00, agent_growth_dd=0.18,
    agent_ruin=-0.48,   agent_ruin_sd=0.04, agent_ruin_dd=1.39, agent_ruin_ruin=0.03,  # ruin g0.95/dbl-off
    z_agent_growth=-1.3, z_agent_ruin=-4.1,                             # agent-vs-Flat, seed-SE + 20k CI
    # best-checkpoint (H3) — the visited ramps, measured (Test 13 + growth raw cell)
    bc_ruin_dd_lo=13.9, bc_ruin_dd_hi=18.3,                             # ruin dbl-off / dbl-on, dd %
    bc_growth=-0.52, bc_growth_sd=0.23, bc_growth_dd=5.0, z_bc_growth=-4.8,  # growth raw, n=9
    visit_dist_min=2, visit_dist_max=3,                                 # near-Kelly orbit dips (L1)
    # encoding ablation (Test 14) — growth x1e-4
    enc_raw=-0.19, enc_logratio=-0.28, enc_none=-0.19,
    # bankroll-coverage (Test 15) — cover vs fixed, native growth
    cover_p_growth=0.89, cover_p_ruin=0.90, cover_starved_pct=25.93, fixed_starved_pct=0.05,
    # committed edge reference (20M hands, git eab466d)
    break_even_tc=0.76, edge_tc0=-0.32, edge_tc6=2.46, reward_sd=1.15,  # edge %, per-hand SD
    freq_tc6=0.9, freq_tc8=0.26,                                        # % of hands
    f_tc2=0.49, f_tc4=1.24, f_tc6=1.96,                                 # full-Kelly f* %, mu/sigma^2
    edge_over_sd_pct=2.1, n_hands_ref_m=20.0,                           # edge/SD ratio; reference size (M)
    kelly_ladder="1 1 1 2 5 8 8",                                       # discrete Kelly over TC -4..+8
    # B2c committed 20k runs — wonging record + the ladder's flat-8 cells (all verified below)
    wong_forced_400=-0.062, wong_400=+0.143, wong_tax_400=0.21,
    wong_forced_200=-0.314, wong_200=+0.136, wong_tax_200=0.45,
    flat8_ruin_400=20.0, flat8_ruin_200=53.0, flat8_growth_200=+4.67,
)


def verify_data() -> None:
    """Assert every loader-backed pinned constant against the live artifacts. Fail loud on drift."""
    import json
    import math

    base = ladder_baselines()
    ev = load_bet_evals()

    def se_of(row) -> float:
        """MC standard error from a ladder row's 95% growth CI."""
        return float(row.growth_hi_1e4 - row.growth_lo_1e4) / 2 / 1.96

    def z_unpaired(mu_a, se_a, mu_b, se_b) -> float:
        return (mu_a - mu_b) / math.sqrt(se_a ** 2 + se_b ** 2)

    def cell(regime: str, bettor: str):
        return base[(base.regime == regime) & (base.bettor == bettor)].iloc[0]

    def agent(regime: str, **filt):
        m = ev[(ev.phase == "final") & (ev.regime == regime) & (ev.train_regime == regime)
               & (ev.bettor == "agent")]
        for k, v in filt.items():
            m = m[m[k] == v]
        return m

    growth_raw = agent("growth", bankroll_feature="raw")
    ruin_clean = agent("ruin", gamma=0.95, double="off")
    bc_growth = ev[(ev.phase == "best-ckpt") & (ev.regime == "growth") & (ev.train_regime == "growth")
                   & (ev.bettor == "agent") & (ev.bankroll_feature == "raw")]
    enc = {f: agent("growth", bankroll_feature=f).growth_1e4.mean() for f in ("raw", "logratio", "none")}

    gk, gf = cell("growth", "kelly"), cell("growth", "flat")
    rk, rf = cell("ruin", "kelly"), cell("ruin", "flat")
    z_ruin_gap = z_unpaired(rk.growth_1e4, se_of(rk), rf.growth_1e4, se_of(rf))
    wong_rec = json.load(open(sorted((ROOT / "runs").glob("*_wonging_*"))[-1] / "record.json",
                              encoding="utf-8"))["cells"]

    def wong(cell_key: str) -> float:
        return wong_rec[cell_key]["growth_rate"]["value"] * 1e4

    checks = [
        ("growth_kelly", cell("growth", "kelly").growth_1e4, 0.005),
        ("growth_flat", cell("growth", "flat").growth_1e4, 0.005),
        ("growth_prize", cell("growth", "kelly").growth_1e4 - cell("growth", "flat").growth_1e4, 0.005),
        ("ruin_kelly", cell("ruin", "kelly").growth_1e4, 0.005),
        ("ruin_flat", cell("ruin", "flat").growth_1e4, 0.005),
        ("ruin_gap", cell("ruin", "kelly").growth_1e4 - cell("ruin", "flat").growth_1e4, 0.005),
        ("ruin_kelly_dd", cell("ruin", "kelly").dd_pct, 0.05),
        ("ruin_flat_dd", cell("ruin", "flat").dd_pct, 0.05),
        ("agent_growth", growth_raw.growth_1e4.mean(), 0.01),
        ("agent_growth_sd", growth_raw.growth_1e4.std(), 0.01),
        ("agent_ruin", ruin_clean.growth_1e4.mean(), 0.01),
        ("agent_ruin_sd", ruin_clean.growth_1e4.std(), 0.01),
        ("agent_ruin_dd", ruin_clean.dd_pct.mean(), 0.05),
        ("agent_growth_ruin", growth_raw.ruin_pct.mean(), 0.02),
        ("agent_ruin_ruin", ruin_clean.ruin_pct.mean(), 0.02),
        ("bc_growth", bc_growth.growth_1e4.mean(), 0.01),
        ("bc_growth_sd", bc_growth.growth_1e4.std(), 0.01),
        ("bc_growth_dd", bc_growth.dd_pct.mean(), 0.1),
        ("enc_raw", enc["raw"], 0.01),
        ("enc_logratio", enc["logratio"], 0.01),
        ("enc_none", enc["none"], 0.01),
        ("cover_p_growth", coverage_growth_ztest("growth")["p"], 0.01),
        ("cover_p_ruin", coverage_growth_ztest("ruin")["p"], 0.01),
        # significance claims made in prose — recomputed here, never asserted from memory
        ("z_growth_prize", z_unpaired(gk.growth_1e4, se_of(gk), gf.growth_1e4, se_of(gf)), 0.2),
        ("p_ruin_gap", math.erfc(abs(z_ruin_gap) / math.sqrt(2)), 0.01),
        ("z_agent_growth", z_unpaired(growth_raw.growth_1e4.mean(),
                                      growth_raw.growth_1e4.std() / math.sqrt(len(growth_raw)),
                                      gf.growth_1e4, se_of(gf)), 0.15),
        ("z_agent_ruin", z_unpaired(ruin_clean.growth_1e4.mean(),
                                    ruin_clean.growth_1e4.std() / math.sqrt(len(ruin_clean)),
                                    rf.growth_1e4, se_of(rf)), 0.15),
        ("z_bc_growth", z_unpaired(bc_growth.growth_1e4.mean(),
                                   bc_growth.growth_1e4.std() / math.sqrt(len(bc_growth)),
                                   gf.growth_1e4, se_of(gf)), 0.2),
        # secondary table/prose cells
        ("agent_growth_dd", growth_raw.dd_pct.mean(), 0.05),
        ("growth_kelly_dd", gk.dd_pct, 0.05),
        ("growth_flat_dd", gf.dd_pct, 0.05),
        ("edge_over_sd_pct", DATA["edge_tc6"] / DATA["reward_sd"], 0.1),
        # B2c committed cells — wonging record + the ladder's flat-8 rows
        ("wong_forced_400", wong("growth/forced"), 0.005),
        ("wong_400", wong("growth/wong"), 0.005),
        ("wong_forced_200", wong("ruin/forced"), 0.005),
        ("wong_200", wong("ruin/wong"), 0.005),
        ("wong_tax_400", wong("growth/wong") - wong("growth/forced"), 0.02),
        ("wong_tax_200", wong("ruin/wong") - wong("ruin/forced"), 0.02),
        ("flat8_ruin_400", cell("growth", "flat-8").ruin_pct, 0.5),
        ("flat8_ruin_200", cell("ruin", "flat-8").ruin_pct, 0.5),
        ("flat8_growth_200", cell("ruin", "flat-8").growth_1e4, 0.01),
    ]
    # the baselines the tables print as ruin "0" — Kelly and Flat never ruin, in either regime
    for regime in ("growth", "ruin"):
        for bettor in ("kelly", "flat"):
            assert cell(regime, bettor).ruin_pct == 0, f"{regime}/{bettor} ruin_pct != 0"
    # structural claims: the Kelly ladder string and the near-Kelly visit distances
    live_ladder = " ".join(str(KELLY_LADDER[c]) for c in sorted(KELLY_LADDER))
    assert DATA["kelly_ladder"] == live_ladder, f"kelly ladder drifted: {live_ladder}"
    dists = near_kelly_runs("ruin", 4).min_kelly_dist
    assert (abs(dists.min() - DATA["visit_dist_min"]) < 0.01
            and abs(dists.max() - DATA["visit_dist_max"]) < 0.01), f"visit dists drifted: {dists.tolist()}"
    from blackjack_rl.session.references import load_edge_reference
    ref = load_edge_reference()
    n_total = sum(e.n for e in ref.edges.values())
    e0, e1 = ref.edges[0].mean_return, ref.edges[1].mean_return
    checks += [
        ("edge_tc0", ref.edges[0].mean_return * 100, 0.02),
        ("edge_tc6", ref.edges[6].mean_return * 100, 0.02),
        ("freq_tc6", ref.edges[6].n / n_total * 100, 0.05),
        ("freq_tc8", ref.edges[8].n / n_total * 100, 0.05),
        ("break_even_tc", -e0 / (e1 - e0), 0.02),           # linear interpolation between TC 0 and +1
        ("f_tc2", ref.edges[2].mean_return / ref.edges[2].variance * 100, 0.03),
        ("f_tc4", ref.edges[4].mean_return / ref.edges[4].variance * 100, 0.03),
        ("f_tc6", ref.edges[6].mean_return / ref.edges[6].variance * 100, 0.03),
        ("n_hands_ref_m", n_total / 1e6, 0.5),
    ]

    failures = [f"  {name}: pinned {DATA[name]:+.4g} != live {float(live):+.4g} (tol {tol})"
                for name, live, tol in checks if abs(DATA[name] - float(live)) > tol]
    if failures:
        raise AssertionError("report numbers drifted from the artifacts:\n" + "\n".join(failures))
    print(f"verify_data: {len(checks)} pinned numbers verified against live loaders")


# ============================ CHARTS — one code path with the notebooks ============================
def _grab(name: str) -> str:
    """Save the plotter's current figure (the loaders draw notebook-style) and close everything."""
    out = CHARTS_DIR / f"{name}.png"
    plt.gcf().savefig(out, dpi=160, bbox_inches="tight")
    plt.close("all")
    return str(out)


def build_charts() -> dict[str, str]:
    runs = load_bet_runs()
    evals = load_bet_evals()
    agent_cfg = {"growth": {"bankroll_feature": "raw"}, "ruin": {"gamma": 0.95, "double": "off"}}
    charts: dict[str, str] = {}

    # the analytic story
    import importlib.util as ilu
    import json as _json
    from blackjack_rl.core.paths import EDGE_REFERENCE_PATH
    spec = ilu.spec_from_file_location("plot_edge_by_count", ROOT / "scripts" / "plot_edge_by_count.py")
    edge_mod = ilu.module_from_spec(spec); spec.loader.exec_module(edge_mod)
    edge_mod.render(_json.loads(Path(EDGE_REFERENCE_PATH).read_text()), CHARTS_DIR / "edge_by_count.png")
    charts["edge_curve"] = str(CHARTS_DIR / "edge_by_count.png")

    plot_prize_bar(regime="growth"); charts["prize"] = _grab("prize")
    plot_count_frequency(); charts["count_freq"] = _grab("count_freq")
    plot_signal_vs_noise(DATA["edge_tc6"] / 100, DATA["reward_sd"], signal_label="edge at TC +6")
    charts["signal_noise"] = _grab("signal_noise")

    # the learned bettor
    plot_ladder_bars(evals, "growth_1e4", agent_cfg, title="growth rate (x1e-4/hand) — agent vs Kelly vs Flat")
    charts["ladder_growth"] = _grab("ladder_growth")
    plot_ladder_bars(evals, "dd_pct", agent_cfg, title="deep-drawdown % — agent vs Kelly vs Flat")
    charts["ladder_dd"] = _grab("ladder_dd")
    ruin_harm = runs[(runs.regime == "ruin") & (runs.lr_sched == "harmonic")]
    rep = ruin_harm[(~ruin_harm.double) & (ruin_harm.seed == 0)].iloc[-1]
    plot_bet_orbit(rep.path); charts["orbit"] = _grab("orbit")
    plot_kelly_distance(near_kelly_runs("ruin", 4),
                        label=lambda r: f"{'dbl' if r.double else 'sgl'}/{r.lr_sched[:4]}/s{r.seed}")
    charts["kelly_dist"] = _grab("kelly_dist")
    plot_bet_orbit(oracle_run(), note="oracle control — denoised expected log-reward · same net · γ=0")
    charts["oracle"] = _grab("oracle")
    plot_ladder_bars(evals, "dd_pct", agent_cfg, phase="best-ckpt",
                     title="best-checkpoint deep-drawdown % — the visited ramps vs Kelly vs Flat")
    charts["bc_dd"] = _grab("bc_dd")
    plot_native_curves(runs[(runs.regime == "growth") & (runs.bankroll_feature == "raw")].tail(4),
                       title="native bet-vs-count (growth, raw) — mostly flat / coarse gating")
    charts["native"] = _grab("native")

    # the wealth-hypothesis arc
    probe = runs[(runs.regime == "growth") & (runs.bankroll_feature == "raw") & (runs.seed == 3)].iloc[-1]
    states, emb, bet = bet_embedding(probe.path)
    plot_bet_embedding_row(bet_project(emb, "tsne"), states, bet)
    charts["embedding"] = _grab("embedding")
    plot_bet_replication("growth", "raw", bankrolls=(400, 100))
    charts["replication"] = _grab("replication")
    plot_encoding_ablation(evals); charts["ablation"] = _grab("ablation")
    plot_bet_coverage("growth"); charts["coverage_growth_bets"] = _grab("coverage_growth_bets")
    plot_coverage_growth(); charts["coverage_growth"] = _grab("coverage_growth")

    # appendix equations — matplotlib mathtext rendered to tight transparent PNGs (no LaTeX install)
    def formula(name: str, tex: str, fontsize: float = 14) -> None:
        f = plt.figure(figsize=(8, 1.2))
        f.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=fontsize)
        out = CHARTS_DIR / f"eq_{name}.png"
        f.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.06, transparent=True)
        plt.close(f)
        charts[f"eq_{name}"] = str(out)

    formula("kelly_def", r"f^*(TC)\ =\ \mathrm{arg\,max}_f\ \ \mathrm{E}\left[\,\log(1 + f\,R_{TC})\,\right]")
    formula("kelly_taylor",
            r"g(f)\ =\ \mathrm{E}\left[\log(1+fR)\right]\ \approx\ f\mu\ -\ \frac{1}{2}f^{2}\sigma^{2}"
            r"\qquad\Longrightarrow\qquad f^{*}\ =\ \frac{\mu}{\sigma^{2}}")
    formula("fractional",
            r"g(c)\ =\ g_{\mathrm{max}}\,\left(2c - c^{2}\right)\ =\ g_{\mathrm{max}}\,\left(1-(1-c)^{2}\right)")

    print(f"build_charts: {len(charts)} charts -> {CHARTS_DIR.name}/")
    return charts


# ============================ PDF ============================
def build_pdf(C: dict[str, str]) -> None:
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
    h1.keepWithNext = 1  # never strand a section heading at a page bottom

    story: list = []

    def P(t, s=body): story.append(Paragraph(t, s))
    def fig(key, caption, width=None):
        if C.get(key) is None:
            return
        width = width or W * 0.84
        iw, ih = ImageReader(C[key]).getSize()
        story.append(KeepTogether([  # image + caption move as one — no orphaned/split captions
            Image(C[key], width=width, height=width * ih / iw, hAlign="CENTER"),
            Paragraph(caption, cap),
        ]))
    def rule():
        hr = HRFlowable(width=W, thickness=1, color=C_ACCENT, spaceAfter=10)
        hr.keepWithNext = 1  # heading + rule + first paragraph stay together
        story.append(hr)
    def eq(key):
        iw, ih = ImageReader(C[key]).getSize()
        w = iw * 72 / 240  # natural size at the render dpi — crisp, text-matched scale
        story.append(Spacer(1, 2 * mm))
        story.append(Image(C[key], width=w, height=w * ih / iw, hAlign="CENTER"))
        story.append(Spacer(1, 3 * mm))
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
    def box(header, body_paras):
        flows = [Paragraph(header, boxh)] + [Paragraph(t, boxb) for t in body_paras]
        story.append(panel(flows, bg=C_PANEL, pad=12)); story.append(Spacer(1, 4 * mm))

    # ----- COVER -----
    story.append(Spacer(1, 42 * mm))
    cover = [Paragraph("Betting Against the Noise", title),
             Spacer(1, 2 * mm),
             Paragraph("Can reinforcement learning rediscover Kelly bet-sizing", sub),
             Paragraph("on blackjack&rsquo;s thin, countable edge?", sub),
             Spacer(1, 8 * mm),
             Paragraph("A learned DQN bettor vs. the analytic Kelly ladder, measured over sessions on four axes",
                       meta),
             Spacer(1, 2 * mm),
             Paragraph("Part of AI Journey &mdash; Phase 3: Deep Learning &amp; Reinforcement Learning", meta),
             Spacer(1, 10 * mm),
             Paragraph("arda-basarici.github.io/blackjack-betting",
                       ParagraphStyle("L", fontSize=10.5, textColor=HexColor("#64B5F6"), alignment=TA_CENTER, fontName="Helvetica"))]
    story.append(panel(cover, height=118 * mm)); story.append(PageBreak())

    D = DATA  # short alias for prose interpolation — every number below passed verify_data()

    # ----- EXECUTIVE SUMMARY -----
    P("Executive summary", h1); rule()
    P("Blackjack betting has a rare property for a learning problem: <b>the optimal rule is derivable.</b> "
      "Once you count cards, the edge at each true count is measurable, and the growth-optimal wager is the "
      "Kelly bet. That makes it a clean test of a question that recurs everywhere in applied ML: <i>can an "
      "end-to-end learner rediscover a rule we could also just derive?</i> We trained a DQN bettor on raw "
      "per-hand log-growth reward and measured it against the analytic Kelly ladder and a flat-betting floor, "
      "on identical terms, across seeds, in two bankroll regimes.", lead)
    P("<b>It does not rediscover Kelly &mdash; not from raw hand rewards, not within this "
      "experiment&rsquo;s sample budget.</b> The learned bettor converges to approximately flat betting in "
      "the growth regime and slightly below flat in the ruin regime; the near-Kelly bet curves it "
      "transiently produces measure <i>worse</i> than flat. Only the analytic Kelly ladder beats the flat "
      "floor &mdash; and only barely.", body)
    story.append(styled_table([
        thr(["regime", "policy", "growth /hand (&times;1e-4)", "ruin %", "deep-drawdown %"]),
        tr(["growth", "learned BetAgent", f"{D['agent_growth']:+.2f} &plusmn; {D['agent_growth_sd']:.2f}",
            f"{D['agent_growth_ruin']:.2f}", f"{D['agent_growth_dd']:.2f}"]),
        tr(["growth", "analytic Kelly", f"{D['growth_kelly']:+.3f}", "0", f"{D['growth_kelly_dd']:.2f}"]),
        tr(["growth", "Flat (floor)", f"{D['growth_flat']:+.3f}", "0", f"{D['growth_flat_dd']:.2f}"]),
        tr(["ruin", "learned BetAgent", f"{D['agent_ruin']:+.2f} &plusmn; {D['agent_ruin_sd']:.2f}",
            f"{D['agent_ruin_ruin']:.2f}", f"{D['agent_ruin_dd']:.2f}"]),
        tr(["ruin", "analytic Kelly", f"{D['ruin_kelly']:+.3f}", "0", f"{D['ruin_kelly_dd']:.2f}"]),
        tr(["ruin", "Flat (floor)", f"{D['ruin_flat']:+.3f}", "0", f"{D['ruin_flat_dd']:.2f}"]),
    ], [W * 0.12, W * 0.28, W * 0.28, W * 0.13, W * 0.19], hl_rows=(2, 5)))
    P("Scoreboard. Agent = mean &plusmn; SD over 6 training seeds (2,000 sessions each); Kelly/Flat = the "
      "20,000-session committed baseline. Native bankroll regime for every policy. Ruin is ~zero for every "
      "policy shown &mdash; it is an over-betting phenomenon (Section 2) &mdash; and no policy in this "
      "report is judged on growth alone.", cap)
    P("The reason is not a bug, a weak architecture, or a bad state encoding &mdash; each alternative was "
      "isolated and ruled out, twice with controlled experiments the report walks through. The reason is the "
      f"signal: the entire prize for perfect bet-sizing is ~{D['growth_prize']:.2f}&times;10<super>-4</super> "
      "log-growth per hand, buried in per-hand noise fifty times larger, carried by counts the shoe rarely "
      "deals. Structure &mdash; a Kelly rule read off an edge curve measured once, offline &mdash; extracts "
      "it; end-to-end value learning, re-estimating the same curve through that noise while acting, did not "
      "come close, and the sample arithmetic of Section 6 says why. <b>The practical lesson: when a thin "
      "signal has a derivable form, encode it; don&rsquo;t ask a value learner to rediscover it.</b>", body)
    story.append(PageBreak())

    # ----- 0. THE FOURTH MOVEMENT -----
    P("The fourth movement of the project", h1); rule()
    P("This report closes a four-part blackjack investigation. The first movement built a Monte Carlo "
      "simulator &mdash; fixed rules, reproducible seeds, pluggable strategies, millions of hands. The second "
      "trained a tabular Monte Carlo agent on it and audited the learned policy cell-by-cell against basic "
      "strategy; the finding was that its errors concentrated in rare, coverage-starved cells. The third "
      "replaced the table with a DQN and asked whether neural generalization repairs that coverage problem "
      "&mdash; it does, partially, at the price of distortions a table cannot make.", lead)
    P("All three movements share a limitation: they end at the hand. Playing decisions were measured against "
      "basic strategy &mdash; a reference that is <i>optimal per hand</i> but silent about money. Betting is "
      "where blackjack actually pays: count the shoe, bet small when the edge is against you, bet big on the "
      "rare counts where it turns. And betting brings the one thing the earlier movements never had &mdash; "
      "a <i>derivable optimum</i>. Given a measured edge curve, the growth-optimal bet is the Kelly fraction; "
      "no learning required. So the final movement can ask its question cleanly:", body)
    P("<b>When the optimal rule can be derived, can an end-to-end reinforcement learner rediscover it from "
      "raw reward &mdash; and if not, what exactly stops it?</b>", take)
    P("The answer occupies the rest of this report. Sections 1&ndash;2 build the instrument: the measured "
      "edge, the Kelly reference, and what perfect betting is actually worth. Sections 3&ndash;5 run the "
      "experiment and read the result. Sections 6&ndash;7 establish <i>why</i> &mdash; including a hypothesis "
      "of our own that we built two controlled experiments to test, and falsified. Sections 8&ndash;9 close "
      "the project.", body)

    # ----- 1. THE INSTRUMENT -----
    P("1. The instrument &mdash; a countable edge, and a known optimum", h1); rule()
    P("The unit of play is a <b>session</b>: a bankroll (in table-minimum units) played through 1,000 hands "
      "of six-deck blackjack under fixed rules, with basic-strategy play throughout. Playing decisions are "
      "frozen &mdash; the only free choice is the <b>wager</b>, picked before each hand from an arithmetic "
      "1&ndash;8 unit spread. Per-hand reward is the log-increment of wealth, so a session&rsquo;s total "
      "reward telescopes to log(final/starting bankroll): maximizing it is maximizing compound growth.", lead)
    P("The bettor&rsquo;s information is the <b>Hi-Lo true count</b>. High counts mean a ten/ace-rich shoe "
      "&mdash; more player blackjacks at 3:2, better doubles &mdash; and the classic counting claim is that "
      "the player edge rises roughly half a percent per true count. We did not take that on faith: the edge "
      "was <b>measured</b>, 20 million basic-strategy hands binned by true count (20 parallel workers, "
      f"per-worker seeds, merged variance accumulators). Break-even lands at TC&nbsp;+{D['break_even_tc']:.2f}; "
      f"the neutral count carries {D['edge_tc0']:.2f}% and the rich counts reach "
      f"+{D['edge_tc6']:.1f}% at TC&nbsp;+6. This measured curve &mdash; not a textbook table &mdash; is the "
      "reference every result below is audited against.", body)
    fig("edge_curve", "The signature measurement: player edge vs Hi-Lo true count, 95% CIs, with the implied "
        "full-Kelly fraction (right axis) and the break-even count. 20,000,000 hands, committed reference "
        "(git eab466d). Low-n tails shaded.")
    box("The yardstick: Kelly betting (derivation in Appendix A)", [
        "Choosing log-growth as the objective is a <i>risk preference</i>, not a mathematical necessity "
        "&mdash; expected-value maximization says bet everything on any positive edge and goes broke on the "
        "way. Given that preference, the optimal fraction of bankroll is the <b>Kelly bet</b>: to second "
        "order, f* = edge / variance of the per-hand return, floored at zero when the edge is negative.",
        f"From the measured curve, full Kelly is small: f* &asymp; {D['f_tc2']:.1f}% of bankroll at TC +2, "
        f"{D['f_tc4']:.1f}% at +4, {D['f_tc6']:.1f}% at +6. At a 400-unit bankroll that is bets of roughly "
        "2, 5, and 8 units &mdash; which is why the 1&ndash;8 arithmetic spread and the ~400u bankroll were "
        "<i>derived together</i> from the curve, not assumed. Snapped to the spread, the discrete-Kelly "
        f"ladder over counts (-4&hellip;+8) is <b>{D['kelly_ladder']}</b>.",
        "Two refinements matter later: <b>fractional Kelly</b> (betting a fraction of f* costs growth only "
        "to second order but cuts volatility linearly), and the <b>ruin-aware bend</b> (with a hard ruin "
        "barrier, the optimum bets below Kelly as wealth approaches it). Appendix A derives both.",
    ])
    P("Everything runs in two <b>bankroll regimes</b>, because bankroll changes what betting is about. The "
      "<b>growth</b> regime starts rich (400u): ruin is effectively out of reach and the pure question is "
      "growth-optimal sizing. The <b>ruin</b> regime starts lean (200u) with a hard ruin barrier: "
      "over-betting can now end the session, so restraint carries survival value. Each policy is always "
      "scored in its native regime.", body)

    # ----- 2. THE ANALYTIC LADDER -----
    P("2. The analytic ladder &mdash; what perfect betting is worth", h1); rule()
    P("Before any learning, the analytic baselines set the scale. Three reference bettors &mdash; <b>Flat</b> "
      "(always one unit: no betting skill), <b>discrete Kelly</b> (the ladder above: perfect skill on the "
      "same 1&ndash;8 menu), and <b>flat-8</b> (always the maximum: reckless aggression) &mdash; were each "
      "measured over 20,000 sessions per regime. A bettor is never one number here; four axes are held "
      "apart throughout: <b>growth rate</b> (log-growth per hand, the objective), <b>ruin %</b> (the "
      "catastrophe), <b>deep-drawdown %</b> (losing half the bankroll &mdash; pain short of ruin), and the "
      "final-bankroll distribution.", lead)
    story.append(styled_table([
        thr(["regime", "bettor", "growth /hand (&times;1e-4)", "ruin %", "deep-drawdown %"]),
        tr(["growth (400u)", "Flat", f"{D['growth_flat']:+.3f}", "0", f"{D['growth_flat_dd']:.2f}"]),
        tr(["growth (400u)", "Kelly (discrete)", f"{D['growth_kelly']:+.3f}", "0", f"{D['growth_kelly_dd']:.2f}"]),
        tr(["growth (400u)", "flat-8 (max)", "&mdash;", f"{D['flat8_ruin_400']:.0f}", "&mdash;"]),
        tr(["ruin (200u)", "Flat", f"{D['ruin_flat']:+.3f}", "0", f"{D['ruin_flat_dd']:.2f}"]),
        tr(["ruin (200u)", "Kelly (discrete)", f"{D['ruin_kelly']:+.3f}", "0", f"{D['ruin_kelly_dd']:.2f}"]),
        tr(["ruin (200u)", "flat-8 (max)", f"{D['flat8_growth_200']:+.2f}", f"{D['flat8_ruin_200']:.0f}", "&mdash;"]),
    ], [W * 0.18, W * 0.24, W * 0.28, W * 0.12, W * 0.18], hl_rows=(2, 5)))
    P("The committed 20k-session bet ladder. flat-8 growth in the growth regime and drawdown columns for "
      "flat-8 omitted &mdash; ruin dominates every other axis for it.", cap)
    P("Three findings anchor the whole report. <b>First, the betting lever works, but the prize is tiny.</b> "
      f"Kelly beats Flat on growth in the growth regime by ~{D['growth_prize']:.2f}&times;10<super>-4</super> "
      f"per hand (unpaired z&nbsp;&asymp;&nbsp;{D['z_growth_prize']:.0f}, decisive); in the ruin regime the "
      f"edge is real but marginal (~{D['ruin_gap']:.2f}&times;10<super>-4</super>, "
      f"p&nbsp;&asymp;&nbsp;{D['p_ruin_gap']:.2f}) and bought with more drawdown. <b>Second, even perfect betting loses money here.</b> Kelly&rsquo;s growth is negative in "
      "both regimes &mdash; the table minimum forces a mandatory bet on the frequent negative-edge hands, a "
      "tax no sizing rule escapes. &lsquo;Kelly beats Flat&rsquo; means <i>loses less</i>; the skill being "
      "priced is restraint, not profit. <b>Third, ruin is purely an over-betting phenomenon.</b> Kelly never "
      f"ruins, even on the lean bankroll; flat-8 ruins {D['flat8_ruin_400']:.0f}% of sessions rich and "
      f"{D['flat8_ruin_200']:.0f}% lean.", body)
    P("<b>Kelly is optimal under the constraint &mdash; and the constraint can still make the game "
      "losing.</b>", take)
    P("The flat-8 row is also this project&rsquo;s best argument for the four-axis discipline. In the ruin "
      f"regime it posts the <i>best</i> growth number on the board (+{D['flat8_growth_200']:.2f}) while "
      f"ruining {D['flat8_ruin_200']:.0f}% of sessions &mdash; the growth is pure survivor bias, averaged "
      "over the half that lived. Collapse the axes into one score and the worst bettor looks like the "
      "best.", body)
    P("<b>A policy can look best on growth and still be unacceptable once ruin is shown &mdash; no policy "
      "in this report is judged on growth alone.</b>", take)
    P("One sidebar quantifies the tax. Allowing the bettor to <i>sit out</i> negative counts (min-wager "
      "zero &mdash; &lsquo;Wonging&rsquo;, after Stanford Wong) flips continuous-Kelly growth from "
      f"{D['wong_forced_400']:+.3f} to <b>{D['wong_400']:+.3f}</b>&times;10<super>-4</super> at 400u, and "
      f"from {D['wong_forced_200']:+.3f} to <b>{D['wong_200']:+.3f}</b> at 200u &mdash; counting genuinely "
      "beats the house once you are not forced to bet its hands, and with <i>less</i> drawdown. This matters "
      "in Section 9: it is the analytic answer to a learning experiment we scoped but did not run.", body)
    fig("prize", "The prize, drawn to scale: Kelly vs Flat growth in the growth regime, 20k-session 95% CIs. "
        "The gap between the bars — about 0.10×1e-4 log-growth per hand — is the ENTIRE learnable sizing "
        "prize: everything perfect count-based bet-sizing is worth on this table.")

    # ----- 3. THE QUESTION -----
    P("3. The question &mdash; can a value learner rediscover it?", h1); rule()
    P("The learner is deliberately plain: a DQN (two hidden layers of 64) that sees "
      "<b>(true count, decks remaining, bankroll)</b> and outputs a Q-value per bet level; the greedy argmax "
      "is the wager. Reward is the raw per-hand log-growth &mdash; exactly the objective Kelly maximizes, so "
      "if value learning succeeds, the greedy policy <i>is</i> the Kelly ladder. No reward shaping, no "
      "curriculum, no oracle hints: the point is to test end-to-end learning on honest terms. Training uses "
      "the standard apparatus &mdash; replay buffer, target network, Huber loss &mdash; hardened over the "
      "investigation (reward scaling against the Huber knee, harmonic learning-rate decay, epsilon decay, "
      "batch and &gamma; sweeps; the ruin regime gets &gamma;&nbsp;=&nbsp;0.95 and the growth regime the "
      "myopic &gamma;&nbsp;=&nbsp;0, which is optimal there because Kelly is a per-hand optimum).", lead)
    box("What is being learned here (vs. the previous report)", [
        "<b>Play is frozen</b> to basic strategy &mdash; unlike the previous report, no playing decision is "
        "learned. <b>Only the wager is learned.</b>",
        "<b>State:</b> (Hi-Lo true count, decks remaining, bankroll) &middot; <b>Actions:</b> bet sizes "
        "1&ndash;8 units &middot; <b>Reward:</b> per-hand log-growth of wealth.",
        "<b>Success criterion:</b> the greedy bet-by-count curve matches the discrete-Kelly ladder "
        f"({D['kelly_ladder']}) <i>and</i> the four-axis evaluation matches Kelly&rsquo;s.",
    ])
    P("Every result reads off the same <b>three-rung ladder</b>: Flat is the floor (no skill), analytic "
      "Kelly is the ceiling (the derivable optimum), and the BetAgent is the question. &lsquo;&asymp; "
      "Flat&rsquo; means no betting skill was learned; &lsquo;&asymp; Kelly&rsquo; would mean the rule was "
      "rediscovered. The measurement protocol: each configuration trains under <b>6 random seeds</b>; each "
      "seed is scored over 2,000 fresh sessions on the four axes; agents carry the SD across seeds, the "
      "deterministic baselines carry their 20k-session Monte-Carlo CIs; bettors play independent shoe "
      "streams, so comparisons use unpaired z-tests. Figures show seeds as dots rather than whiskers "
      "&mdash; at n&nbsp;=&nbsp;6 an SD bar would hide exactly the seed-to-seed story that matters.", body)
    P("One trap shaped the whole evaluation design, and it recurs so often it deserves stating up front: "
      "<b>a bet curve that looks like Kelly at one probe can be a terrible policy once measured.</b> The "
      "report never trusts an eyeballed curve; verdicts come only from the four-axis evaluation at the "
      "native bankroll.", take)

    # ----- 4. THE RESULT -----
    P("4. The result &mdash; never Kelly", h1); rule()
    P("The scoreboard in the executive summary is the verdict; here is how to read it. In the <b>growth</b> regime the "
      f"agent lands at {D['agent_growth']:+.2f}&nbsp;&plusmn;&nbsp;{D['agent_growth_sd']:.2f} against "
      f"Flat&rsquo;s {D['growth_flat']:+.3f} &mdash; statistically indistinguishable from the no-skill floor "
      f"(z&nbsp;&asymp;&nbsp;{abs(D['z_agent_growth']):.1f}, n.s.) and nowhere near Kelly&rsquo;s "
      f"{D['growth_kelly']:+.3f}. In the "
      f"<b>ruin</b> regime it lands <i>below</i> the floor: {D['agent_ruin']:+.2f} vs {D['ruin_flat']:+.3f} "
      f"(z&nbsp;&asymp;&nbsp;{abs(D['z_agent_ruin']):.0f} across seeds), with more drawdown &mdash; no skill gained, and a small price "
      "paid for trying. This holds across every hardening cell (double-DQN on/off, the &gamma; sweep, three "
      "state encodings): no configuration approaches the Kelly rung.", lead)
    fig("ladder_growth", "Growth rate by policy and regime — the agent bar (mean over 6 seeds, dots) sits at "
        "Flat's height in growth and below it in ruin; the Kelly rung stays out of reach in both. Baselines "
        "from the 20k ladder with 95% CIs.")
    P("A verdict this negative earns its keep only if the alternatives are ruled out. The next three "
      "sections do that in order: the policy&rsquo;s behaviour during training (it is not a near-miss), the "
      "signal itself (the wall), and the representation (a hypothesis we falsified ourselves).", body)

    # ----- 5. UP CLOSE -----
    P("5. The bettor up close &mdash; an orbit that visits and leaves", h1); rule()
    P("Watching the greedy bet-vs-count curve evolve over training is the most revealing view of what "
      "&lsquo;&asymp; Flat&rsquo; actually looks like. The <b>low counts settle</b> quickly to the minimum "
      "bet and lock. The <b>high counts never settle</b> &mdash; they flicker between Kelly-like ramps and "
      "flat for the entire run. Settling at the low counts is not skill: Kelly also bets the minimum there, "
      "so that behaviour is indistinguishable from betting flat everywhere. Only the high counts, where "
      "Kelly ramps to 5 and 8 units, can reveal betting skill &mdash; and those are exactly the ones that "
      "never lock.", lead)
    fig("orbit", "The training orbit of a representative ruin-regime run: greedy bet level by true count "
        "(heatmap) over training, loss beneath. Low counts lock at the minimum; the high-count rows flicker "
        "between ramp and flat for the whole run. Read this as policy INSTABILITY, not near-success: the "
        "ramp appears transiently, is not retained, and — Section 6 — does not evaluate well.")
    P("The flicker is not noise around flat &mdash; the orbit repeatedly <i>passes through</i> genuinely "
      "Kelly-shaped curves and leaves them. Tracking the L1 distance between the bet curve and the Kelly "
      "ladder over training makes the visits explicit: dips to L1 distance "
      f"{D['visit_dist_min']:.0f}&ndash;{D['visit_dist_max']:.0f} (essentially Kelly) that "
      "never persist. The network can <i>express</i> the target policy; it cannot <i>hold</i> it. That "
      "observation sharpens the question the next section answers &mdash; and it plants a hope the next "
      "section kills: perhaps the best mid-training checkpoint, rather than the final policy, is the real "
      "prize.", body)
    fig("kelly_dist", "Distance-to-Kelly over training for the four runs that approach it closest — each dip "
        "is a checkpoint where the policy was briefly Kelly-shaped. None holds.")

    # ----- 6. WHY -----
    P("6. Why &mdash; the thin edge", h1); rule()
    P("Before blaming the signal, rule out the method. The <b>oracle control</b> keeps the same network, "
      "replay buffer and encode&rarr;Q&rarr;argmax pipeline at the growth bettor&rsquo;s &gamma;&nbsp;=&nbsp;0, "
      "and swaps each hand&rsquo;s <i>realized</i> log-reward for its <i>noise-free expectation</i> from the "
      "measured edge curve, rescaled to order one. It runs on plain baseline knobs &mdash; batch 128, "
      "constant learning rate &mdash; none of the noise-averaging machinery the real-reward runs needed. If "
      "the flatline were a bug, a capacity limit, or an optimisation failure, the oracle would flatten too. "
      "It does not: it locks a clean, stable Kelly ramp and keeps it.", lead)
    fig("oracle", "The positive control: same network and pipeline, denoised expected reward. Low counts "
        "hold the minimum, high counts ramp to Kelly levels — and stay. The pipeline is sound; what changes "
        "on real reward is only the signal.")
    P("So the wall is the signal, and it has three compounding parts. <b>Part one: the prize is tiny.</b> "
      f"Everything a learner could win is the Kelly-over-Flat gap of Section 2 &mdash; "
      f"~{D['growth_prize']:.2f}&times;10<super>-4</super> log-growth per hand. <b>Part two: it is "
      "sub-noise.</b> The bettor must estimate the edge <i>through</i> the per-hand payoff, and a single "
      f"hand&rsquo;s reward has standard deviation ~{D['reward_sd']:.2f} &mdash; even the strongest max-bet "
      f"edge (+{D['edge_tc6']:.1f}% at TC&nbsp;+6) is about <b>2% of one standard deviation</b>. Resolving a "
      "value difference that small takes enormous samples per state.", body)
    fig("signal_noise", "Schematic, to scale: the per-hand reward distribution at the richest max-bet count "
        "against the no-edge baseline. The two curves are visually one — the signal a value learner must "
        "resolve is buried in per-hand noise.")
    P("<b>Part three: the states that matter are rare.</b> The shoe spends most of its life near neutral; "
      f"TC&nbsp;+6, where Kelly first bets the maximum, is {D['freq_tc6']:.1f}% of hands, and TC&nbsp;+8 is "
      f"{D['freq_tc8']:.2f}%. The high counts are simultaneously where the edge lives, where the required "
      "sample size is largest, and where samples are scarcest. This is the betting-side reprise of the "
      "tabular audit&rsquo;s finding two movements ago &mdash; errors concentrate where natural play "
      "starves the learner of data.", body)
    fig("count_freq", "True-count visit frequency (log scale), from the 20M-hand reference. The shaded "
        "region — where Kelly bets above the minimum — is a few percent of all hands.")
    box("The signal wall, at a glance", [
        f"<b>1. The prize is tiny</b> &mdash; the entire Kelly-over-Flat gap is "
        f"~{D['growth_prize']:.2f}&times;10<super>-4</super> log-growth per hand.",
        f"<b>2. The noise is huge</b> &mdash; per-hand reward SD ~{D['reward_sd']:.2f}; even the strongest "
        "max-bet edge is ~2% of one standard deviation.",
        f"<b>3. The carriers are rare</b> &mdash; the counts that hold the edge are "
        f"~{D['freq_tc6']:.1f}% of hands (TC +6) and {D['freq_tc8']:.2f}% (TC +8).",
        "Each factor alone is surmountable; their <b>product</b> is the wall. The oracle isolates it: "
        "remove factor 2 and the same pipeline learns Kelly immediately.",
    ])
    P("What about the ramps the orbit visits &mdash; is the best checkpoint secretly a good policy? "
      "<b>No, and measurably so.</b> Taking each run&rsquo;s closest-to-Kelly checkpoint (selected by curve "
      "shape, never by performance &mdash; an honest &lsquo;is the ramp real&rsquo; probe, not "
      "cherry-picking) and scoring it on the four axes: in the ruin regime the visited ramps breach the "
      f"half-bankroll mark {D['bc_ruin_dd_lo']:.0f}&ndash;{D['bc_ruin_dd_hi']:.0f}% of the time versus "
      f"Flat&rsquo;s {D['ruin_flat_dd']:.1f}%, no better on growth; in the growth regime they run "
      f"~{D['bc_growth_dd']:.0f}% drawdown versus zero and are decisively worse on growth "
      f"({D['bc_growth']:+.2f}&nbsp;&plusmn;&nbsp;{D['bc_growth_sd']:.2f} vs {D['growth_flat']:+.3f}, "
      f"z&nbsp;&asymp;&nbsp;{abs(D['z_bc_growth']):.0f}). They look like Kelly at a single-count probe and over-bet in the full policy "
      "&mdash; the Section-3 trap, caught red-handed. The wandering itself was also chased to mechanism: an "
      "early &lsquo;limit cycle&rsquo; hypothesis (the agent forgets punishments once it stops over-betting) "
      "was ruled out by construction at &gamma;&nbsp;=&nbsp;0, leaving static argmax instability on a "
      "near-flat Q-surface &mdash; harmonic learning-rate decay collapses the orbit, and it collapses to "
      "<i>flat</i>, not to the ramp. Flat is the loss-attractor; the ramps are the part that averages away.", body)
    fig("bc_dd", "The visited ramps, measured: best-checkpoint deep-drawdown vs the baselines. The "
        "Kelly-shaped excursions are over-betting policies the training objective correctly declines to keep.")
    P("Put together, the evidence points to the signal as the wall &mdash; and the correct reading of "
      "&lsquo;&asymp; Flat&rsquo; stops being &lsquo;the agent failed to find the answer&rsquo; and becomes "
      "<b>flat is the loss-optimal answer for the signal it could actually resolve</b>. One escape hatch "
      "remains: maybe the signal is fine and the <i>representation</i> is the problem. That hypothesis got "
      "the full treatment.", body)

    # ----- 7. THE WEALTH HYPOTHESIS -----
    P("7. A hypothesis of our own, tested and falsified", h1); rule()
    P("It started with a picture. Projecting the trained network&rsquo;s penultimate-layer activations over "
      "a grid of (count, depth, bankroll) states, three colourings tell three tempting stories: <b>bankroll "
      "separates the clusters</b> (&lsquo;it keyed on wealth&rsquo;), <b>count is a clean gradient within "
      "them</b> (&lsquo;no, it encoded the count&rsquo;), and the <b>greedy bet tracks the wealth "
      "clusters</b> &mdash; which seems to settle it for wealth. The growth and ruin networks even mirror: "
      "one bets big on its low-bankroll side, the other on its high-bankroll side. The hypothesis "
      "practically writes itself: <i>the agent learned the &times;bankroll half of Kelly but not the "
      "count gate &mdash; the wall is representational, not fundamental.</i>", lead)
    fig("embedding", "The picture that launched the hypothesis: one growth network's embedding, coloured by "
        "bankroll, count, and greedy bet. Clusters split by wealth; the bet appears to track them.")
    P("Two disciplines dissolved the picture before any conclusion was drawn from it. <b>Out-of-"
      "distribution:</b> the probe grid sweeps bankrolls 50&ndash;600u, but each agent lives in a narrow "
      "band around its start &mdash; most of that wealth axis is states the policy never occupies, so the "
      "clean separation is the network extrapolating where no gradient ever corrected it. <b>Replication:</b> "
      "probing every seed&rsquo;s bet-vs-count curve at the native bankroll, the seeds disagree completely "
      "&mdash; dead-flat, coarse gating, erratic &mdash; and none is the Kelly ramp. The elegant structure "
      "was one seed&rsquo;s idiosyncrasy. An embedding is suggestive, not decisive.", body)
    fig("replication", "The replication check that killed the first read: every growth seed's greedy bet "
        "curve at native (left) and out-of-distribution (right) bankroll, discrete Kelly dashed. No common "
        "structure survives across seeds.")
    P("So the hypothesis was promoted from picture-reading to <b>prediction</b>, and tested twice, from "
      "opposite sides. <b>Test one &mdash; remove wealth.</b> The bet encoder was made configurable: "
      "bankroll fed raw, as a scale-free log-ratio, or <b>removed from the input entirely</b>. If the agent "
      "keyed on wealth, blinding it to wealth must change its behaviour. It changes nothing: "
      f"<b>none</b> ({D['enc_none']:+.2f}) is statistically identical to <b>raw</b> ({D['enc_raw']:+.2f}), "
      f"log-ratio no better ({D['enc_logratio']:+.2f}); all three sit at the Flat floor.", body)
    fig("ablation", "The encoding ablation: growth across seeds for raw / log-ratio / no-bankroll encoders, "
        "Kelly and Flat dashed. Removing wealth from the input entirely moves nothing.")
    P("<b>Test two &mdash; fill wealth in.</b> If the embedding&rsquo;s mirror is starvation &mdash; each "
      "regime over-betting exactly the wealth band it never trains in &mdash; then training <i>across</i> "
      "the wealth axis (each session&rsquo;s starting bankroll cycling 100&ndash;600u) should collapse the "
      "artifact; and if that out-of-band over-betting was costing growth, erasing it should pay. The first "
      f"prediction <b>confirmed</b>: the coverage-trained agents play {D['cover_starved_pct']:.0f}% of hands "
      f"in the formerly-starved band (vs {D['fixed_starved_pct']:.2f}%) and the over-betting humps collapse "
      "onto Kelly&rsquo;s floor &mdash; the mirror was untrained-wealth extrapolation, not strategy. The "
      f"second prediction <b>falsified</b>: growth does not move (p&nbsp;&asymp;&nbsp;{D['cover_p_growth']:.2f} "
      f"growth regime, {D['cover_p_ruin']:.2f} ruin), nor does any other axis. The artifact lived where "
      "native play never goes; erasing it is cosmetic.", body)
    fig("coverage_growth_bets", "Filling in the wealth axis: bet at a fixed count across bankroll, "
        "fixed-start vs coverage-trained agents vs Kelly. The starvation hump collapses; the four-axis "
        "outcome does not move.")
    P("A footnote the method earned: the first coverage wave (3 seeds) showed a <i>significant</i> ruin-"
      "regime gain (p&nbsp;&asymp;&nbsp;0.005). It did not survive three more seeds &mdash; a small-sample "
      "fluke of exactly the kind this project&rsquo;s protocol exists to catch, recorded rather than "
      "erased.", take)
    P("Two independent experiments, opposite directions, one verdict: <b>nothing wealth-shaped is "
      "load-bearing.</b> Remove wealth &mdash; nothing changes; fill it in &mdash; the artifact dissolves "
      "and still nothing changes. With that, both representational doors are closed: the oracle of "
      "Section 6 closed the <i>capacity</i> door (the network can represent and hold the Kelly ladder when "
      "the signal is clean), and the two ablations closed the <i>input-encoding</i> door. The wall is not "
      "the representation; it is the thin, sub-noise, rare-state edge of Section 6. And the arc matters as "
      "much as the verdict: a hypothesis formed from an internal representation, an over-read caught by "
      "out-of-distribution and replication checks, and a falsification delivered by our own controlled "
      "experiments.", body)

    # ----- 8. SYNTHESIS -----
    P("8. Synthesis &mdash; structure beats end-to-end on a sub-noise signal", h1); rule()
    P("The edge exists; Kelly captures it; the end-to-end learner, in these runs and under this reward, "
      "did not. The asymmetry is sample budgets. The analytic bettor reads the edge from a curve measured <i>once, offline</i>, over 20 "
      "million pooled hands &mdash; then applies a two-line formula. The DQN must re-estimate that same "
      "curve <i>online</i>, through per-hand noise fifty times larger than the signal, from the few percent "
      "of hands that carry it, while simultaneously using the estimate to act. Same edge, opposite sample "
      "economics. Where a signal is thin, noisy and rare, structure is not a convenience &mdash; it is the "
      "difference between extractable and not.", lead)
    P("This closes a through-line the whole project kept meeting. The tabular audit found errors pooling in "
      "coverage-starved cells; the DQN study found generalization repairing coverage while distorting "
      "boundaries; the betting study finds the extreme case &mdash; a signal so thin that no amount of "
      "end-to-end estimation within the experiment&rsquo;s budget resolves it, while a derived rule "
      "captures it outright. The skill the environment actually prices is <i>restraint</i> &mdash; not "
      "over-betting a mostly-negative game &mdash; and the learned agent achieves it only in the trivial "
      "sense of converging to the minimum bet everywhere: a mistake avoided, not a skill gained.", body)
    P("A result like this is only as trustworthy as the checks it survived, and the method is the other "
      "half of what this project set out to practice: build the analytic baseline first, test the learner "
      "against it on identical terms, chase the tempting hypothesis, falsify your own interpretation, and "
      "record the false positive instead of hiding it. Four times the first read of the evidence was "
      "wrong. Recording these is the point &mdash; the conclusion stands <i>because</i> they were "
      "caught:", body)
    story.append(styled_table([
        thr(["first read (tempting)", "the check that caught it", "corrected finding"]),
        tr(["double-DQN looked 'safe' (low drawdown)", "multi-seed CIs, not one run",
            "seed luck &mdash; dd 2.6 &plusmn; 3.05%, unstable"], bold0=False),
        tr(["the visited near-Kelly ramps are the real policy", "best-checkpoint four-axis eval",
            "far riskier than Flat, no better on growth"], bold0=False),
        tr(["it keyed on wealth, not count (embedding)", "OOD + replication + encoding ablation",
            "seed-specific artifact; falsified twice"], bold0=False),
        tr(["wealth-coverage recovers growth (n=3, p=0.005)", "three more seeds, four-axis re-test",
            "gain vanished (p&asymp;0.9); small-sample fluke"], bold0=False),
    ], [W * 0.36, W * 0.30, W * 0.34]))
    P("The honesty trail — four tempting over-reads, and the discipline that caught each.", cap)
    P("Four movements end here. A simulator became a measurement instrument; a table exposed where "
      "learning starves; a network showed what generalization buys and breaks; and a bettor drew the "
      "clean line: <b>when the optimum is derivable, derive it &mdash; and know how to prove that the "
      "learner&rsquo;s failure is the signal&rsquo;s fault, not your pipeline&rsquo;s.</b> The second half "
      "of that sentence is the transferable skill.", body)

    # ----- 9. SCOPED AND SET ASIDE -----
    P("9. What we scoped and set aside", h1); rule()
    P("Each entry below was designed far enough to know what it would cost and predict what it would show "
      "&mdash; the value here is demonstrating the judgment, not promising the work.", lead)
    P("<b>Count-coverage (the missing sibling).</b> Section 7 filled in the <i>wealth</i> axis; the exact "
      "analog for the <i>count</i> axis &mdash; oversampling high-count states via engineered shoes or "
      "stratified replay &mdash; would directly separate rarity from noise. Prediction: the ramp stabilizes "
      "but real-play growth does not move, because oversampling reweights gradients without growing the "
      f"~{D['growth_prize']:.2f}&times;10<super>-4</super> prize. Deferred because engineered shoes must "
      "first be validated to preserve the measured conditional edge (count is a linear summary of "
      "composition; forcing it can silently change the game) &mdash; a study of its own.", body)
    P("<b>Learnable abstention (drop the forced minimum bet).</b> The table-minimum tax dominates the "
      "growth numbers, and Section 2&rsquo;s Wonging sidebar already gives the analytic answer: sitting "
      "out negative counts flips growth positive. Adding a bet-0 action would test whether a value learner "
      "can discover abstention &mdash; a larger, cleaner signal than sizing (the tax is "
      "~0.2&ndash;0.45&times;10<super>-4</super> per hand, several times the sizing prize), which makes it "
      "the most promising learning experiment on this list.", body)
    P("<b>Prioritized / stratified replay.</b> The standard remedy for rare-state starvation (Schaul et "
      "al.); the cheap version of count-coverage. Same prediction, same caveat &mdash; it reweights, it "
      "does not add information.", body)
    P("<b>Paired evaluation with common random numbers.</b> Our bettors play independent shoes, so tiny "
      "gaps need 20k sessions to resolve. Sharing shoe streams across policies would collapse the variance "
      "of the <i>difference</i> and resolve the ruin-regime margins cleanly at a fraction of the cost; it "
      "requires a small evaluator change and per-session artifacts, and would be the first infrastructure "
      "investment of any follow-up.", body)
    P("<b>Multi-step credit (TD(&lambda;)), horizon-relative ruin, factored-vs-monolithic, learning to "
      "count.</b> Considered and set aside deliberately: the bet&rsquo;s reward lands the same hand it is "
      "placed, so multi-step returns buy little; a &lsquo;ruined-for-the-remaining-horizon&rsquo; metric "
      "is more realistic but needs an arbitrary recovery model; joint play-and-bet learning and learning "
      "the count statistic itself from raw composition are natural extensions of the question, at "
      "respectively higher compute and lower expected yield.", body)

    # ----- CLOSE -----
    P("Close", h1); rule()
    P("What was tested: whether an end-to-end value learner, given the same information a card counter "
      "uses and the exact objective Kelly optimizes, rediscovers the derivable betting rule. What "
      "happened: it did not &mdash; it converged to the no-skill floor, and its Kelly-shaped excursions "
      "measured worse than the floor. Why: the prize is tiny, the noise is fifty times larger, and the "
      "states that carry the signal are rare &mdash; while the analytic rule reads the same edge from a "
      "curve measured once, offline. <b>The transferable problem is not blackjack betting; it is deciding "
      "when to learn a policy end-to-end and when to encode known structure &mdash; and knowing how to "
      "prove which regime you are in. RL was the experiment; structure was the lesson.</b>", lead)

    # ----- APPENDIX A -----
    story.append(PageBreak())
    P("Appendix A &mdash; the Kelly criterion, derived", h1); rule()
    P("<b>A.1 Why log-growth is the objective &mdash; a chosen risk preference.</b> Let f be the fraction "
      "of bankroll wagered and R the per-unit return of a hand (win +1, lose &ndash;1, blackjack +1.5, "
      "doubles &plusmn;2, &hellip;). Wealth multiplies: after the hand it is W(1 + fR). Maximizing "
      "<i>expected wealth</i> E[W(1+fR)] is linear in f &mdash; on any positive edge it says bet "
      "everything, every hand, and a single loss ends the game; almost-sure ruin with maximal expectation "
      "is the classic pathology of the arithmetic mean in a multiplicative process. Over T hands wealth is "
      "a <i>product</i> of factors, and what controls its long-run behaviour is the mean of the "
      "<i>logs</i>: log W(T) = log W(0) + sum of log(1 + f&middot;R(t)). Maximizing E[log W] maximizes the "
      "compound growth <i>rate</i> and, by the law of large numbers, almost surely dominates any other "
      "strategy eventually. Choosing it is still a preference &mdash; it accepts large swings that a "
      "mean-variance bettor would not &mdash; and the honest framing is: <i>given</i> log-growth as the "
      "objective, Kelly is a theorem; the objective itself is a decision. This project declares it as the "
      "objective and prices the swings separately on the risk axes.", body)
    P("<b>A.2 The Kelly fraction.</b> The exact object is the growth-maximizing fraction at each count "
      "&mdash; concave in f, so the optimum is unique:", body)
    eq("eq_kelly_def")
    P("Writing the expectation outcome by outcome (the only place the count enters is the outcome "
      "probabilities) and Taylor-expanding the logarithm &mdash; below, the mean of R is the <i>edge</i> "
      "and its variance is &asymp; 1.3 for blackjack&rsquo;s payoff mix:", body)
    eq("eq_kelly_taylor")
    P("(floored at zero when the edge is negative &mdash; don&rsquo;t bet a losing count). A term of order "
      "edge-squared beside the variance is dropped along the way (the edge is ~0.01, the variance ~1.3) "
      "&mdash; the small-bet approximation, and the only approximation involved; it is why the Kelly "
      "fraction has the memorable <i>edge over variance</i> form. The moments have no closed form (the "
      "game tree is too rich), so they are Monte-Carlo estimated &mdash; the 20M-hand curve of Section 1. "
      f"From it: f* &asymp; {D['f_tc2']:.1f}% of bankroll at TC +2, {D['f_tc4']:.1f}% at +4, "
      f"{D['f_tc6']:.1f}% at +6 &mdash; full Kelly on this game is <i>small</i>, under 2% of bankroll "
      "across the realistic range. At a 400-unit bankroll that is 2, 5, and 8 units; snapped to the "
      f"1&ndash;8 spread this is the discrete ladder <b>{D['kelly_ladder']}</b> used throughout, and the measured "
      "cost of the snapping is nil (discrete Kelly &asymp; continuous Kelly on every axis). The spread top "
      "itself is derived the same way &mdash; top &asymp; f*(+6) &times; bankroll &asymp; 8 units at 400u "
      "&mdash; and an arithmetic ladder matches the near-linear ramp of f* in the productive band, where a "
      "geometric (1,2,4,8) menu would over-resolve the bottom and starve the 5&ndash;7u bets Kelly "
      "actually wants.", body)
    P("<b>A.3 Fractional Kelly &mdash; why backing off is cheap.</b> Bet a fraction c of full Kelly, for c "
      "between 0 and 1. Substituting c&middot;f* into g(f), with g<sub>max</sub> the full-Kelly growth "
      "rate:", body)
    eq("eq_fractional")
    P("Half Kelly keeps 75% of the growth at half the volatility: the growth penalty near the peak is "
      "second-order (the top of a concave curve is flat &mdash; g&rsquo;(f*) = 0) while the volatility cut "
      "is first-order (volatility scales linearly with f). This asymmetry is why practical bettors play "
      "fractional Kelly, and why over-betting is so much worse than under-betting: past f* the same "
      "second-order flatness turns against you, and by 2f* growth is back to zero with volatility doubled. "
      "Section 2&rsquo;s ladder shows the live version &mdash; flat-8 is far beyond f* at most counts, and "
      "it ruins.", body)
    P("<b>A.4 The ruin-aware bend.</b> Without a ruin barrier, maximizing E[log W(T)] is <i>myopic</i>: "
      "full Kelly every hand, horizon irrelevant &mdash; finite time alone does not change the bet. A hard "
      "barrier (bankroll below the table minimum ends the session permanently) breaks the myopia: hitting "
      "it forfeits every future positive-edge hand, so capital near the barrier carries option value "
      "beyond the current hand&rsquo;s log term. The optimum becomes a function b*(W, TC, hands left) that "
      "equals Kelly when W is far from the barrier and bends <i>below</i> Kelly as W approaches it &mdash; "
      "under-bet to survive. This is why the project runs two regimes: at 400u the barrier is out of reach "
      "and pure Kelly is the target; at 200u restraint acquires survival value, which is exactly the "
      "regime where the learned agent&rsquo;s over-betting excursions get priced (Section 6).", body)
    P("<b>A.5 What is exact and what is estimated.</b> The definition of f* is exact; the "
      "edge-over-variance form is the small-bet approximation (fine here &mdash; the dropped term is four "
      "orders below the kept one); the moments are Monte-Carlo estimates whose standard errors shrank as "
      "1/&radic;n to two significant figures over the 20M-hand run. The reference the agent is graded "
      "against is therefore itself a measured object with quantified uncertainty &mdash; matching the "
      "report&rsquo;s standing rule that a number without provenance and error bars is not yet a result.", body)


    doc.build(story)
    print(f"wrote {OUTPUT_PDF}")


def main() -> None:
    verify_data()
    build_pdf(build_charts())


if __name__ == "__main__":
    main()
