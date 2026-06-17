"""CLI: train a tabular MC blackjack agent, evaluate it, and save the run.

    python -m blackjack_rl --episodes 5000000 --seed 42
    python -m blackjack_rl --episodes 5000000 --epsilon-schedule linear \
        --epsilon-start 0.3 --epsilon-end 0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from blackjack_rl.config import ExperimentConfig
from blackjack_rl.experiment import run_experiment
from blackjack_rl.schedules import KINDS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tabular Monte Carlo blackjack agent and save the run."
    )
    parser.add_argument("--episodes", type=int, default=5_000_000, help="training hands")
    parser.add_argument("--seed", type=int, default=42, help="training RNG seed")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate (constant schedule)")
    parser.add_argument("--epsilon-schedule", choices=KINDS, default="constant", help="exploration schedule")
    parser.add_argument("--epsilon-start", type=float, default=0.3, help="start rate (decaying schedule)")
    parser.add_argument("--epsilon-end", type=float, default=0.0, help="end rate (decaying schedule)")
    parser.add_argument("--eval-hands", type=int, default=200_000, help="hands for edge eval")
    parser.add_argument("--eval-seed", type=int, default=0, help="eval RNG seed")
    parser.add_argument("--min-visits", type=int, default=1000, help="policy-diff visit threshold")
    parser.add_argument("--ev-tol", type=float, default=0.02, help="policy-diff EV-gap tolerance")
    parser.add_argument(
        "--progress-every", type=int, default=500_000, help="progress every N episodes (0 = silent)"
    )
    parser.add_argument("--runs-dir", type=str, default=None, help="output dir (default: ./runs)")
    args = parser.parse_args()

    config = ExperimentConfig(
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
    )
    result = run_experiment(
        config,
        eval_hands=args.eval_hands,
        eval_seed=args.eval_seed,
        min_visits=args.min_visits,
        ev_tol=args.ev_tol,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        progress_every=args.progress_every or None,
        verbose=True,
    )

    print(f"\nrun saved to: {result.run_dir}")
    print(f"agent edge: {result.agent_edge.edge * 100:.3f}% +/- {result.agent_edge.std_error * 100:.3f}")
    print(f"basic edge: {result.basic_edge.edge * 100:.3f}% +/- {result.basic_edge.std_error * 100:.3f}")
    print(
        f"agreement:  {result.diff.agreement_weighted:.1%} visit-weighted, "
        f"{result.diff.agreement_unweighted:.1%} unweighted"
    )
    print(f"categories: {result.diff.category_counts}")


if __name__ == "__main__":
    main()
