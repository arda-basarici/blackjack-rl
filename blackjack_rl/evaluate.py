"""CLI: re-evaluate a saved run's policy over more hands, without retraining.

    python -m blackjack_rl.evaluate --run runs/<dir> --hands 1000000 --seed 0

The trained policy lives in the saved record's qtable, so this loads it and measures the
greedy house edge (and basic strategy on the same shuffles) at whatever sample size you want.
The agreement / category breakdown is a fixed property of the policy (it doesn't depend on the
number of eval hands), so it is read straight from the saved record rather than recomputed.
"""
from __future__ import annotations

import argparse
import time

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.evaluation.metrics import GreedyPolicy, evaluate_policy
from blackjack_rl.experiment import load_agent
from blackjack_rl.persistence import load_record
from blackjack_rl.util import format_duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a saved run's policy without retraining."
    )
    parser.add_argument("--run", required=True, help="run directory or record.json path")
    parser.add_argument("--hands", type=int, default=1_000_000, help="eval hands")
    parser.add_argument("--seed", type=int, default=0, help="eval RNG seed")
    args = parser.parse_args()

    record = load_record(args.run)
    agent = load_agent(record)
    print(f"loaded policy from {args.run}")
    print(f"re-evaluating edge over {args.hands:,} hands (seed {args.seed}) ...")

    t0 = time.perf_counter()
    agent_edge = evaluate_policy(GreedyPolicy(agent), n_hands=args.hands, seed=args.seed)
    basic_edge = evaluate_policy(BasicStrategy(), n_hands=args.hands, seed=args.seed)
    elapsed = time.perf_counter() - t0

    gap = agent_edge.edge - basic_edge.edge
    print(f"agent edge: {agent_edge.edge * 100:.3f}% +/- {agent_edge.std_error * 100:.3f}")
    print(f"basic edge: {basic_edge.edge * 100:.3f}% +/- {basic_edge.std_error * 100:.3f}")
    print(f"gap (agent - basic): {gap * 100:.3f}%")

    # Fidelity is fixed by the policy, not the eval size — show what was recorded at train time.
    diff = record.get("diff")
    if diff:
        print(
            f"\npolicy fidelity (from saved record; min_visits {diff['min_visits']}, "
            f"ev_tol {diff['ev_tol']}):"
        )
        print(
            f"  agreement: {diff['agreement_weighted']:.1%} visit-weighted, "
            f"{diff['agreement_unweighted']:.1%} unweighted"
        )
        print(f"  categories: {diff['category_counts']}")
    print(f"\ndone in {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
