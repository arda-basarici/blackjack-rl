"""Four-axis eval of each run's BEST near-Kelly checkpoint — the H3 diagnostic.

For every run dir, find the logged checkpoint whose greedy bet curve is closest (L1) to the discrete-Kelly
ladder, then four-axis eval THAT checkpoint via ``eval_bet_agent`` (``run_dir@session``). The baselines are
cached, so each run is ~1 minute.

DIAGNOSTIC framing (NOT model selection / NOT a deliverable claim): the question is *"when a run visited
its best ramp, was that ramp actually a better policy than flat — higher growth, tolerable ruin — or a
noise excursion?"* Selecting the checkpoint by closeness-to-Kelly would be leakage for a *claim* ("RL
reaches Kelly"); it is fine for *interrogating* the best excursion the orbit produced. Treat the output as
"is the ramp real?", not "here is our Kelly bettor."

    .venv\\Scripts\\python.exe scripts/eval_best_checkpoints.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
KELLY: dict[int, int] = {-4: 1, -2: 1, 0: 1, 2: 2, 4: 5, 6: 8, 8: 8}


def best_ramp_checkpoint(run_dir: str) -> tuple[int, int]:
    """(session, L1-distance-to-Kelly) of the run's closest-to-Kelly logged checkpoint.

    The learning curve is logged at the same cadence the weights are snapshotted, so the returned session
    has a loadable ``checkpoints/ckpt_<session>.pt``."""
    curve = json.load(open(Path(run_dir) / "record.json", encoding="utf-8"))["metrics"]["learning_curve"]

    def distance(checkpoint: dict) -> int:
        bets = checkpoint["bet_by_count"]
        return sum(abs(bets.get(str(c), bets.get(c)) - KELLY[c]) for c in COUNTS)

    best = min(curve, key=distance)
    return best["session"], distance(best)


def main() -> None:
    run_dirs = sys.argv[1:]
    if not run_dirs:
        raise SystemExit("usage: eval_best_checkpoints.py <run_dir> [<run_dir> ...]")
    for run_dir in run_dirs:
        session, dist = best_ramp_checkpoint(run_dir)
        print(f"\n>>> {Path(run_dir).name}  best ramp @{session} (Kelly L1 dist {dist})", flush=True)
        subprocess.run(
            [sys.executable, "scripts/eval_bet_agent.py", f"{run_dir}@{session}", "2000", "8"], check=False
        )


if __name__ == "__main__":
    main()
