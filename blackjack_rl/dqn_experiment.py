"""Run orchestration + CLI for the DQN — train, evaluate, diff vs basic strategy, and save.

The deep-Q parallel to ``experiment.run_experiment`` (tabular). Train a ``DQNAgent``, measure its
greedy house edge and basic strategy's (the anchor), diff the learned policy against basic strategy
by *interrogating* the network cell by cell (``network_diff``), assemble a record, and save it
(never overwriting). No Q-table is stored — the network has none — so the saved ``diff.cells`` are
the materialized policy; a same-seed rerun reproduces the agent (A7, no checkpoint/resume yet).

    python -m blackjack_rl.dqn_experiment --episodes 2000000 \
        --epsilon-schedule linear --epsilon-start 0.3 --epsilon-end 0.0 --eval-hands 1000000
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.evaluation.metrics import EdgeResult, GreedyPolicy, evaluate_policy
from blackjack_rl.evaluation.network_diff import diff_network
from blackjack_rl.evaluation.policy_diff import DiffReport
from blackjack_rl.persistence import save_run
from blackjack_rl.schedules import KINDS
from blackjack_rl.training.deep_q import train_dqn
from blackjack_rl.training.exploring_starts_dqn import train_dqn_es
from blackjack_rl.util import format_duration

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class _Tee:
    """Write a stream to several sinks at once (terminal + log file), so a run's progress is both
    visible live and saved for later comparison."""

    def __init__(self, *sinks) -> None:
        self._sinks = sinks

    def write(self, data: str) -> int:
        for s in self._sinks:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._sinks:
            s.flush()


@dataclass(frozen=True)
class DQNRunResult:
    """The outcome of one DQN run: where it was saved (if any) and the headline numbers."""

    run_dir: Path | None
    agent_edge: EdgeResult
    basic_edge: EdgeResult
    diff: DiffReport
    agent: DQNAgent


def run_dqn(
    config: DQNConfig,
    eval_hands: int = 200_000,
    eval_seed: int = 0,
    ev_tol: float = 0.02,
    runs_dir: Path | None = None,
    progress_every: int | None = None,
    verbose: bool = False,
    save: bool = True,
) -> DQNRunResult:
    """Train, evaluate (agent + basic), diff vs basic strategy, and (optionally) persist one run."""

    def log(message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    if config.exploring_starts:
        explore = "exploring-starts (greedy follow, no epsilon)"
    elif config.epsilon_schedule == "constant":
        explore = f"epsilon {config.epsilon}"
    else:
        explore = f"epsilon {config.epsilon_schedule} {config.epsilon_start}->{config.epsilon_end}"
    lr_desc = (
        f"lr {config.lr}" if config.lr_schedule == "constant"
        else f"lr {config.lr_schedule} {config.lr}->{config.lr_end}"
    )
    log(
        f"[{started:%Y-%m-%d %H:%M:%S}] training {config.num_episodes:,} episodes "
        f"(seed {config.seed}, {explore}, {lr_desc}) ..."
    )
    learning_curve: list[dict] = []
    trainer = train_dqn_es if config.exploring_starts else train_dqn
    agent = trainer(config, progress_every=progress_every, on_checkpoint=learning_curve.append)
    train_seconds = time.perf_counter() - t0
    log(f"  training done in {format_duration(train_seconds)}")

    eval_start = time.perf_counter()
    log(f"evaluating agent over {eval_hands:,} hands ...")
    agent_edge = evaluate_policy(GreedyPolicy(agent), n_hands=eval_hands, seed=eval_seed)
    log(f"evaluating basic strategy over {eval_hands:,} hands ...")
    basic_edge = evaluate_policy(BasicStrategy(), n_hands=eval_hands, seed=eval_seed)
    log("diffing learned policy vs basic strategy (cell-by-cell network interrogation) ...")
    report = diff_network(agent, ev_tol=ev_tol)
    eval_seconds = time.perf_counter() - eval_start

    finished = datetime.now().astimezone()
    total_seconds = time.perf_counter() - t0
    log(
        f"[{finished:%Y-%m-%d %H:%M:%S}] eval + diff done in "
        f"{format_duration(eval_seconds)} (total {format_duration(total_seconds)})"
    )

    run_dir: Path | None = None
    if save:
        record = {
            "method": "dqn",
            "config": asdict(config),
            "eval": {"hands": eval_hands, "seed": eval_seed},
            "timing": {
                "started_at": started.isoformat(timespec="seconds"),
                "finished_at": finished.isoformat(timespec="seconds"),
                "train_seconds": round(train_seconds, 1),
                "eval_seconds": round(eval_seconds, 1),
                "total_seconds": round(total_seconds, 1),
            },
            "metrics": {"agent": asdict(agent_edge), "basic": asdict(basic_edge)},
            "diff": {
                "ev_tol": ev_tol,
                "agreement_unweighted": report.agreement_unweighted,
                "agreement_weighted": report.agreement_weighted,
                "category_counts": report.category_counts,
                "cells": [asdict(cell) for cell in report.cells],
            },
            "sample_counts": [
                {"player_value": k[0], "is_soft": k[1], "dealer_upcard": k[2],
                 "action": k[3], "count": v}
                for k, v in getattr(agent, "sample_counts", {}).items()
            ],
            "learning_curve": learning_curve,
        }
        target = runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR
        run_dir = save_run(target, record)
        log(f"saved run to {run_dir}")

    return DQNRunResult(
        run_dir=run_dir, agent_edge=agent_edge, basic_edge=basic_edge, diff=report, agent=agent
    )


def _print_summary(result: DQNRunResult) -> None:
    """Headline numbers plus the genuine disagreements and a couple of famous probe cells."""
    diff = result.diff
    if result.run_dir is not None:
        print(f"\nrun saved to: {result.run_dir}")
    print(f"agent edge: {result.agent_edge.edge * 100:.3f}% +/- {result.agent_edge.std_error * 100:.3f}")
    print(f"basic edge: {result.basic_edge.edge * 100:.3f}% +/- {result.basic_edge.std_error * 100:.3f}")
    print(
        f"agreement:  {diff.agreement_weighted:.1%} weighted, "
        f"{diff.agreement_unweighted:.1%} unweighted"
    )
    print(f"categories: {diff.category_counts}")

    genuine = sorted(
        (c for c in diff.cells if c.category == "genuine_disagreement"),
        key=lambda c: abs(c.agent_q - c.basic_q),
        reverse=True,
    )
    if genuine:
        print(f"\ngenuine disagreements ({len(genuine)}), worst first:")
        for c in genuine[:15]:
            kind = "soft" if c.is_soft else "hard"
            gap = abs(c.agent_q - c.basic_q)
            print(
                f"  {kind} {c.player_value} v{c.dealer_upcard}: "
                f"net={c.agent_action} basic={c.basic_action} ev_gap={gap:.3f}"
            )

    # cross-reference a couple of cells the tabular report singled out
    by_cell = {(c.player_value, c.is_soft, c.dealer_upcard): c for c in diff.cells}
    print("\nprobe cells (report's notable ones):")
    for label, key in (("soft 16 v4 (report outlier)", (16, True, 4)), ("hard 16 v10 (near-tie)", (16, False, 10))):
        c = by_cell.get(key)
        if c is not None:
            print(f"  {label}: net={c.agent_action} basic={c.basic_action} -> {c.category}")


def _parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN blackjack agent (Problem A), validate vs basic strategy, and save."
    )
    parser.add_argument("--episodes", type=int, default=2_000_000, help="training hands")
    parser.add_argument("--seed", type=int, default=42, help="training RNG seed (random + torch)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate (constant schedule)")
    parser.add_argument("--epsilon-schedule", choices=KINDS, default="linear", help="exploration schedule")
    parser.add_argument("--epsilon-start", type=float, default=0.3, help="start rate (decaying schedule)")
    parser.add_argument("--epsilon-end", type=float, default=0.0, help="end rate (decaying schedule)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (start rate if lr decays)")
    parser.add_argument("--lr-schedule", choices=KINDS, default="constant", help="learning-rate schedule (decay lets the estimate converge; constant = original)")
    parser.add_argument("--lr-end", type=float, default=1e-5, help="end learning rate for a decaying schedule (>0 for harmonic/exponential)")
    parser.add_argument("--gamma", type=float, default=1.0, help="TD discount (1.0 for Problem A)")
    parser.add_argument("--hidden", type=str, default="64,64", help="hidden layer sizes, comma-separated")
    parser.add_argument("--batch-size", type=int, default=128, help="replay minibatch size")
    parser.add_argument("--buffer", type=int, default=50_000, help="replay buffer capacity")
    parser.add_argument("--warmup", type=int, default=1_000, help="transitions before first update")
    parser.add_argument("--updates-per-step", type=int, default=1, help="gradient steps per training event")
    parser.add_argument("--train-every", type=int, default=4, help="train only every N decisions (replay ratio; 1 = every step)")
    parser.add_argument("--target-sync", type=int, default=1_000, help="hard-sync target every N steps")
    parser.add_argument("--target-tau", type=float, default=0.0, help="soft/Polyak target update each step (0 = hard sync)")
    parser.add_argument("--double-dqn", action="store_true", help="use Double-DQN targets (curb overestimation)")
    parser.add_argument("--encoding", choices=("scalar", "onehot"), default="scalar", help="input encoding for total+upcard")
    parser.add_argument("--exploring-starts", action="store_true", help="force (state,action) coverage (the DQN capstone)")
    parser.add_argument("--log-q-grid", action="store_true", help="log full per-cell Q each checkpoint (for trajectory plots)")
    parser.add_argument("--swa", action="store_true", help="Stochastic Weight Averaging over the back half (averages out the oscillation)")
    parser.add_argument("--with-splits", action="store_true", help="enable split action + pair state")
    parser.add_argument("--eval-hands", type=int, default=200_000, help="hands for edge eval")
    parser.add_argument("--eval-seed", type=int, default=0, help="eval RNG seed")
    parser.add_argument("--ev-tol", type=float, default=0.02, help="policy-diff EV-gap tolerance")
    parser.add_argument("--progress-every", type=int, default=100_000, help="progress every N episodes (0 = silent)")
    parser.add_argument("--runs-dir", type=str, default=None, help="output dir (default: ./runs)")
    parser.add_argument("--no-save", action="store_true", help="do not persist the run")
    parser.add_argument("--no-log", action="store_true", help="do not tee output to a logs/ file")
    args = parser.parse_args()

    config = DQNConfig(
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        epsilon_schedule=args.epsilon_schedule,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_end=args.lr_end,
        gamma=args.gamma,
        hidden=_parse_hidden(args.hidden),
        batch_size=args.batch_size,
        buffer_capacity=args.buffer,
        warmup=args.warmup,
        updates_per_step=args.updates_per_step,
        train_every=args.train_every,
        target_sync_every=args.target_sync,
        target_tau=args.target_tau,
        double_dqn=args.double_dqn,
        encoding=args.encoding,
        exploring_starts=args.exploring_starts,
        log_q_grid=args.log_q_grid,
        swa=args.swa,
        with_splits=args.with_splits,
        seed=args.seed,
    )
    log_file = None
    if not args.no_log:
        DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DEFAULT_LOGS_DIR / f"dqn_{datetime.now():%Y%m%d-%H%M%S}_seed{args.seed}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(
            f"(logging to {log_path})  episodes={args.episodes} double_dqn={args.double_dqn} "
            f"eps={args.epsilon_schedule} {args.epsilon_start}->{args.epsilon_end} lr={args.lr} "
            f"train_every={args.train_every}"
        )
    try:
        result = run_dqn(
            config,
            eval_hands=args.eval_hands,
            eval_seed=args.eval_seed,
            ev_tol=args.ev_tol,
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            progress_every=args.progress_every or None,
            verbose=True,
            save=not args.no_save,
        )
        _print_summary(result)
    finally:
        if log_file is not None:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            log_file.close()


if __name__ == "__main__":
    main()
