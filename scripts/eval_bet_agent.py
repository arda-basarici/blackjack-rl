"""Four-axis evaluation of the learned ``BetAgent`` vs the analytic baselines (B2d-3 deliverable).

Loads a trained agent (``load_bet_agent``) and scores it on the full ``metrics`` suite — never collapsed
— against discrete-``KellyBet`` and ``FlatBet``, in **both** regimes, on identical terms via
``cell_eval``:

    {growth, ruin} config  x  {agent, kelly-disc, flat} bettor   (6 cells)

Outcome axis (growth-rate ± CI, final-bankroll quantiles) and risk axis (ruin Wilson, drawdown) are
reported side by side, never merged. Plus the bet-vs-count curve overlaid on Kelly (a diagnostic). The
agent is the same loaded policy in both regimes (trained on its own regime; evaluating it in the other
is an honest cross-check, not a claim of optimality there).

Thread note: this is a torch policy fanned across CPU workers, so OMP/MKL threads are pinned to 1 here
(workers inherit it) to avoid n_workers x threads oversubscription — inference on this tiny net is
single-thread-fine.

    .venv\\Scripts\\python.exe scripts/eval_bet_agent.py <run_dir> [n_sessions] [n_workers] [regime]

``n_workers`` caps the parallel MC pool (each worker is a torch process — it's the peak-memory knob;
lower it when running several evals or on a RAM-tight box). ``regime`` ('growth'|'ruin', omit for both)
limits which configs to score. Kelly/Flat are deterministic, so their metrics are cached (keyed by
n_sessions/n_workers/drawdown/curve/regimes) and reused — repeat agent evals score only the agent cells.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # pin BEFORE torch import; spawned eval workers inherit it
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.session.bet_agent import FlatBet, KellyBet, greedy_bet_curve
from blackjack_rl.session.cell_eval import Cell, evaluate_cells
from blackjack_rl.session.env import GROWTH_BANKROLL, RUIN_BANKROLL, growth_config, ruin_config
from blackjack_rl.session.persistence import load_bet_agent, load_bet_checkpoint
from blackjack_rl.session.references import load_edge_reference

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
N_WORKERS = 12
DRAWDOWN_LEVEL = 0.5
CONFIGS = ("growth", "ruin")
BETTORS = ("agent", "kelly", "flat")


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    msg = f"[{datetime.now():%H:%M:%S}] {line}"
    print(msg, flush=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _row(label: str, values: list[str]) -> None:
    _log(f"{label:<26}" + "".join(f"{v:>16}" for v in values))


def _build_cells(agent, curve: dict[int, float], regimes: tuple[str, ...]) -> list[Cell]:
    """The {regime} x {agent, kelly, flat} cells. **Agent cells first** so an agent-only eval (when the
    baselines are cached) reuses the *same* seed range — ``evaluate_cells`` seeds by cell position, so
    fronting the agent keeps its results reproducible regardless of whether baselines are co-evaluated."""
    config_map = {"growth": growth_config(), "ruin": ruin_config()}
    bettors = {"agent": agent, "kelly": KellyBet(curve, discretize=True), "flat": FlatBet(1.0)}
    return [Cell(f"{c}/{b}", config_map[c], BasicStrategy(), bettors[b]) for b in BETTORS for c in regimes]


# --- baseline cache: Kelly/Flat are deterministic policies, identical every eval -> compute once, reuse.
# Pickled (preserves the metric dicts' float quantile keys, which JSON would stringify). Git-ignored logs/.
_CACHE_PATH = LOGS_DIR / "baseline_eval_cache.pkl"


def _baseline_key(n_sessions: int, n_workers: int, drawdown_level: float, curve: dict, regimes: tuple) -> str:
    """Cache key over everything that determines the baseline metrics (incl. the seed-allocation inputs)."""
    payload = json.dumps({
        "n": n_sessions, "w": n_workers, "dd": drawdown_level, "regimes": list(regimes),
        "kelly": sorted((int(k), round(float(v), 6)) for k, v in curve.items()),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _load_baseline_cache(key: str) -> dict | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        return pickle.loads(_CACHE_PATH.read_bytes()).get(key)
    except Exception:  # corrupt/partial cache — recompute rather than crash
        return None


def _save_baseline_cache(key: str, baseline_metrics: dict) -> None:
    cache = {}
    if _CACHE_PATH.exists():
        try:
            cache = pickle.loads(_CACHE_PATH.read_bytes())
        except Exception:
            cache = {}
    cache[key] = baseline_metrics
    _CACHE_PATH.write_bytes(pickle.dumps(cache))


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _save_eval_result(run_path, label, ckpt_session, n_actual, n_workers, regimes, metrics) -> Path:
    """Persist the eval as a structured, provenance-stamped artifact beside the agent (regenerable, but
    saved so a notebook reads metrics without re-running). `metrics` carries the float-keyed quantile dicts
    `json` stringifies on write — the loader/notebook reads them back as strings."""
    result = {
        "kind": "bet_eval",
        "agent": label,
        "checkpoint_session": ckpt_session,  # None = final model
        "n_sessions": n_actual,
        "n_workers": n_workers,
        "base_seed": 0,
        "drawdown_level": DRAWDOWN_LEVEL,
        "regimes": list(regimes),
        "git_hash": _git_hash(),
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "metrics": metrics,
    }
    tag = f"ckpt{ckpt_session:05d}" if ckpt_session is not None else "final"
    out_path = Path(run_path) / f"eval_{tag}_{n_actual}sess.json"
    out_path.write_text(
        json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    return out_path


def _log_table(config_name: str, bankroll0: float, m: dict) -> None:
    def cell(b: str, key: str) -> dict:
        return m[f"{config_name}/{b}"][key]

    n = m[f"{config_name}/agent"]["n_sessions"]
    _log(f"=== {config_name.upper()} (start {bankroll0:.0f}u, {n:,} sessions) ===")
    _row("metric", list(BETTORS))
    _row("growth/hand (x1e-4)", [f"{cell(b, 'growth_rate')['value'] * 1e4:+.3f}" for b in BETTORS])
    _row("  95% CI", [
        f"[{cell(b, 'growth_rate')['low'] * 1e4:+.2f},{cell(b, 'growth_rate')['high'] * 1e4:+.2f}]"
        for b in BETTORS
    ])
    _row("ruin %", [f"{cell(b, 'ruin')['estimate'] * 100:.2f}" for b in BETTORS])
    _row(f"P(dd>={DRAWDOWN_LEVEL:.0%}) %",
         [f"{cell(b, f'drawdown_breach_{DRAWDOWN_LEVEL}')['estimate'] * 100:.2f}" for b in BETTORS])
    _row("final bankroll p10/50/90", [
        f"{cell(b, 'bankroll')['quantiles'][0.1]:.0f}/{cell(b, 'bankroll')['quantiles'][0.5]:.0f}/"
        f"{cell(b, 'bankroll')['quantiles'][0.9]:.0f}" for b in BETTORS
    ])


def _log_bet_curve(agent, curve: dict[int, float]) -> None:
    """Diagnostic: the learned greedy bet vs discrete-Kelly across the probe counts (growth bankroll)."""
    agent_bets = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=GROWTH_BANKROLL, decks_remaining=3.0)
    kelly = KellyBet(curve, discretize=True)
    kelly_bets = {c: kelly.bet(true_count=float(c), decks_remaining=3.0, bankroll=GROWTH_BANKROLL)
                  for c in PROBE_COUNTS}
    _log("=== bet-vs-count (greedy, @400u) ===")
    _row("true count", [f"{c:+d}" for c in PROBE_COUNTS])
    _row("agent", [f"{agent_bets[c]:g}" for c in PROBE_COUNTS])
    _row("kelly", [f"{kelly_bets[c]:g}" for c in PROBE_COUNTS])


def main() -> None:
    run_dir = sys.argv[1]
    n_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else N_WORKERS
    regime = sys.argv[4] if len(sys.argv) > 4 else None  # 'growth' | 'ruin' | omit for both
    if regime is not None and regime not in CONFIGS:
        raise SystemExit(f"regime must be one of {CONFIGS} or omitted, got {regime!r}")
    regimes = (regime,) if regime else CONFIGS
    torch.set_num_threads(1)

    if "@" in run_dir:  # run_dir@session -> a mid-training checkpoint (H3 diagnostic); else the final model
        run_path, _, sess = run_dir.rpartition("@")
        ckpt_session = int(sess)
        agent = load_bet_checkpoint(run_path, ckpt_session)
        label = f"{Path(run_path).name}@{ckpt_session}"
    else:
        run_path, ckpt_session = run_dir, None
        agent = load_bet_agent(run_path)
        label = Path(run_path).name
    ref = load_edge_reference()
    cells = _build_cells(agent, ref.kelly_curve, regimes)
    agent_cells = [c for c in cells if c.label.endswith("/agent")]

    def on_progress(done: int, total: int) -> None:
        if done == total or done % n_workers == 0:
            _log(f"  {done}/{total} tasks")

    key = _baseline_key(n_sessions, n_workers, DRAWDOWN_LEVEL, ref.kelly_curve, regimes)
    cached = _load_baseline_cache(key)
    start = datetime.now()
    if cached is not None:  # reuse the deterministic Kelly/Flat metrics; evaluate the agent only
        _log(f"eval_bet_agent: baseline cache HIT — agent-only ({len(agent_cells)} cells), "
             f"~{n_sessions:,} sessions/cell, {n_workers} workers, agent={label}")
        fresh, n_actual = evaluate_cells(
            agent_cells, n_sessions=n_sessions, n_workers=n_workers, base_seed=0,
            drawdown_level=DRAWDOWN_LEVEL, on_progress=on_progress,
        )
        metrics = {**cached, **fresh}
    else:
        _log(f"eval_bet_agent: baseline cache MISS — {len(cells)} cells (caching baselines), "
             f"~{n_sessions:,} sessions/cell, {n_workers} workers, agent={label}")
        metrics, n_actual = evaluate_cells(
            cells, n_sessions=n_sessions, n_workers=n_workers, base_seed=0,
            drawdown_level=DRAWDOWN_LEVEL, on_progress=on_progress,
        )
        _save_baseline_cache(key, {k: v for k, v in metrics.items() if not k.endswith("/agent")})

    elapsed = (datetime.now() - start).total_seconds()
    _log(f"--- results ({elapsed:.0f}s, {n_actual:,} sessions/cell) ---")
    _log_bet_curve(agent, ref.kelly_curve)
    bankroll0 = {"growth": GROWTH_BANKROLL, "ruin": RUIN_BANKROLL}
    for r in regimes:
        _log_table(r, bankroll0[r], metrics)
    saved = _save_eval_result(run_path, label, ckpt_session, n_actual, n_workers, regimes, metrics)
    _log(f"saved eval -> {saved}")


if __name__ == "__main__":
    main()
