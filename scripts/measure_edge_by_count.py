"""High-n edge-by-count measurement — produces the canonical committed reference (B2a/B2c).

Fans the flat-bet basic-strategy edge-by-count measurement (``session.references``) across CPU cores
and merges the per-worker Welford partials losslessly (Chan's parallel variance, ``CountAccumulator.
merge``). The single-stream ``edge_by_count`` is too noisy in the extreme buckets to anchor the bet
spread; this drives ~20M hands to tighten the mid-curve and give every bucket a CI.

**Reproducibility-model change (vs B1's single stream).** ``run_sessions`` seeds the global RNG once
per call, so one process = one stream. Here each worker is a *separate process* seeded ``base_seed +
worker_id`` → ``n_workers`` independent streams, merged. The result is regenerable from
``(base_seed, n_workers, hands_per_worker, max_hands_per_session, sim-config)`` — all recorded in the
artifact. Same inputs → same curve (the merge is order-independent up to float rounding).

Run from the repo root (needs the Phase-2 simulator on the path, like any run):

    .venv\\Scripts\\python.exe scripts/measure_edge_by_count.py

Progress streams to ``logs/live.log`` (tail it: ``Get-Content .\\logs\\live.log -Wait``). The result
is written to the **committed** reference ``core.paths.EDGE_REFERENCE_PATH`` (regenerated in place,
with full provenance: run_id / timestamp / git_hash) — the single source the Kelly baseline and the
signature figure both read (DESIGN D17), not a git-ignored ``runs/`` artifact.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from multiprocessing import Pool

from blackjack_rl.core.paths import EDGE_REFERENCE_PATH, LOGS_DIR
from blackjack_rl.core.persistence import git_hash
from blackjack_rl.session.env import problem_b_config
from blackjack_rl.session.references import (
    CountAccumulator,
    accumulate_edges,
    kelly_bet_curve,
)

TOTAL_HANDS = 20_000_000
N_WORKERS = 20
BASE_SEED = 0
MAX_HANDS_PER_SESSION = 1000
FLAT_BET_ANCHOR_PCT = -0.45  # player edge under flat-bet basic strategy (≈ the 0.45% house edge)


def _worker(task: tuple[int, int]) -> CountAccumulator:
    """Run one worker's hand budget off its own seed and return the raw (mergeable) accumulator.
    Top-level so it is picklable under Windows ``spawn`` (the child re-imports this module)."""
    worker_id, n_hands = task
    return accumulate_edges(
        n_hands=n_hands, seed=BASE_SEED + worker_id, max_hands_per_session=MAX_HANDS_PER_SESSION
    )


# Windows consoles default to cp1252, which can't encode the unicode in our log lines (±, Δ, —);
# force utf-8 so a cosmetic print can never crash the run before the artifact is saved.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass


def _log(line: str) -> None:
    """Append a timestamped line to logs/live.log AND echo to stdout (work-in-the-open)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[{stamp}] {line}"
    print(msg, flush=True)
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _write_reference(record: dict, path) -> None:
    """Write the canonical edge-by-count reference to ``path``, stamping provenance (run_id /
    timestamp / git_hash) as ``save_run`` does — but to a **stable committed file**, regenerated in
    place (overwrite), with git as the safety net. The reference is data the package ships (B2c, D17),
    not a never-overwrite ``runs/`` artifact, so the bet baseline and the figure read one source."""
    now = datetime.now()
    full = {
        "run_id": f"edge-by-count_seed{BASE_SEED}_{now.strftime('%Y%m%d-%H%M%S')}",
        "timestamp": now.isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")


def _build_record(merged: CountAccumulator, hands_per_worker: int, elapsed_s: float) -> dict:
    edges = merged.edges()
    pooled = merged.pooled_mean  # exact, over ALL buckets (incl. n<2) — the anchor quantity
    kelly = kelly_bet_curve(edges)
    cfg = problem_b_config()
    return {
        "kind": "edge_by_count",
        "config": {
            "total_hands": TOTAL_HANDS,
            "n_workers": N_WORKERS,
            "base_seed": BASE_SEED,
            "seed": BASE_SEED,  # so save_run's seed field is populated
            "worker_seeds": [BASE_SEED + i for i in range(N_WORKERS)],
            "hands_per_worker": hands_per_worker,
            "max_hands_per_session": MAX_HANDS_PER_SESSION,
            "sim_config": {
                "rules": "6-deck S17 3:2 (vegas_strip)",
                "num_decks": cfg.num_decks,
                "penetration": cfg.penetration,
                "counting_system": "HiLo",
                "min_bet": cfg.min_bet,
            },
        },
        "n_total": merged.n_total,
        "elapsed_s": round(elapsed_s, 1),
        "anchor_check": {
            "pooled_mean_return": pooled,
            "pooled_pct": pooled * 100.0,
            "flat_bet_anchor_pct": FLAT_BET_ANCHOR_PCT,
            "deviation_pct": pooled * 100.0 - FLAT_BET_ANCHOR_PCT,
        },
        "edges": [
            {
                "true_count": e.true_count,
                "mean_return": e.mean_return,
                "variance": e.variance,
                "std_error": e.std_error,
                "n": e.n,
            }
            for e in edges.values()
        ],
        "kelly_curve": {str(tc): f for tc, f in kelly.items()},
    }


def main() -> None:
    per_worker = -(-TOTAL_HANDS // N_WORKERS)  # ceil; the last worker may slightly overshoot the total
    tasks = [(i, per_worker) for i in range(N_WORKERS)]
    _log(
        f"edge-by-count: {TOTAL_HANDS:,} hands over {N_WORKERS} workers "
        f"(~{per_worker:,}/worker), base_seed={BASE_SEED}"
    )

    start = datetime.now()
    merged = CountAccumulator()
    done = 0
    with Pool(N_WORKERS) as pool:
        for partial in pool.imap_unordered(_worker, tasks):
            merged = merged.merge(partial)
            done += 1
            el = (datetime.now() - start).total_seconds()
            _log(
                f"  worker {done}/{N_WORKERS} merged — {merged.n_total:,} hands so far, "
                f"{len(merged.buckets)} buckets, {el:.0f}s"
            )
    elapsed = (datetime.now() - start).total_seconds()

    record = _build_record(merged, per_worker, elapsed)
    ac = record["anchor_check"]
    _log(
        f"anchor check: pooled edge {ac['pooled_pct']:+.3f}%  vs flat anchor "
        f"{ac['flat_bet_anchor_pct']:+.2f}%  (Δ {ac['deviation_pct']:+.3f}%)"
    )
    _log("edge by true count (mean ± SE, n):")
    for e in record["edges"]:
        _log(
            f"  TC {e['true_count']:+d}:  {e['mean_return'] * 100:+6.3f}% "
            f"± {e['std_error'] * 100:.3f}%   (n={e['n']:,})"
        )

    _write_reference(record, EDGE_REFERENCE_PATH)
    _log(f"saved -> {EDGE_REFERENCE_PATH}  ({elapsed:.0f}s, {merged.n_total:,} hands)")
    print(json.dumps(ac, indent=2))


if __name__ == "__main__":
    main()
