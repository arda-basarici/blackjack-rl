"""Run orchestration — train, evaluate, diff, and persist one reproducible run.

Ties the Stage 2 pieces together: train a TabularAgent, measure its greedy house edge and
basic strategy's (the anchor), diff the learned policy cell by cell, assemble the record, and
save it (never overwriting). This is where the run record is finally assembled — the piece
deferred from persistence (D8). See DESIGN.md Stage 2.

``load_agent`` is the inverse: rebuild a trained policy from a saved record's qtable, so a run
can be re-evaluated (more hands, other seeds) without retraining.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.evaluation.metrics import EdgeResult, GreedyPolicy, evaluate_policy
from blackjack_rl.evaluation.policy_diff import DiffReport, diff_policy
from blackjack_rl.persistence import save_run
from blackjack_rl.training.monte_carlo import train
from blackjack_rl.util import format_duration

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass(frozen=True)
class RunResult:
    """The outcome of one run: where it was saved and the headline numbers."""

    run_dir: Path
    agent_edge: EdgeResult
    basic_edge: EdgeResult
    diff: DiffReport


def _qtable_records(agent: TabularAgent) -> list[dict[str, object]]:
    """Flatten the agent's Q-table and visit counts into JSON-friendly records."""
    records: list[dict[str, object]] = []
    for (state_key, action), q in agent.q.items():
        player_value, is_soft, dealer_upcard = state_key
        records.append(
            {
                "player_value": player_value,
                "is_soft": is_soft,
                "dealer_upcard": dealer_upcard,
                "action": action,
                "q": q,
                "n": agent.n.get((state_key, action), 0),
            }
        )
    return records


def load_agent(record: dict[str, Any], epsilon: float = 0.0) -> TabularAgent:
    """Rebuild a TabularAgent from a saved run's ``qtable`` — no retraining.

    The policy is fully defined by its Q-table, so a saved run can be re-evaluated instantly
    (more hands, other seeds, other rule configs). Inverse of ``_qtable_records``.
    """
    agent = TabularAgent(epsilon=epsilon)
    for r in record["qtable"]:
        state_key = (int(r["player_value"]), bool(r["is_soft"]), int(r["dealer_upcard"]))
        action = r["action"]
        agent.q[(state_key, action)] = float(r["q"])
        agent.n[(state_key, action)] = int(r["n"])
    return agent


def run_experiment(
    config: ExperimentConfig,
    eval_hands: int = 200_000,
    eval_seed: int = 0,
    min_visits: int = 1000,
    ev_tol: float = 0.02,
    runs_dir: Path | None = None,
    progress_every: int | None = None,
    verbose: bool = False,
) -> RunResult:
    """Train, evaluate (agent + basic), diff, and persist one run.

    When ``verbose`` is set, prints timestamped phase markers and elapsed times to stderr.
    Timing is always recorded in the saved record regardless of ``verbose``.
    """

    def log(message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    log(
        f"[{started:%Y-%m-%d %H:%M:%S}] training {config.num_episodes:,} episodes "
        f"(seed {config.seed}, epsilon {config.epsilon}) ..."
    )
    agent = train(config, progress_every=progress_every)
    train_seconds = time.perf_counter() - t0
    log(f"  training done in {format_duration(train_seconds)}")

    eval_start = time.perf_counter()
    log(f"evaluating agent over {eval_hands:,} hands ...")
    agent_edge = evaluate_policy(GreedyPolicy(agent), n_hands=eval_hands, seed=eval_seed)
    log(f"evaluating basic strategy over {eval_hands:,} hands ...")
    basic_edge = evaluate_policy(BasicStrategy(), n_hands=eval_hands, seed=eval_seed)
    log("diffing learned policy vs basic strategy ...")
    report = diff_policy(agent, min_visits=min_visits, ev_tol=ev_tol)
    eval_seconds = time.perf_counter() - eval_start

    finished = datetime.now().astimezone()
    total_seconds = time.perf_counter() - t0
    log(
        f"[{finished:%Y-%m-%d %H:%M:%S}] eval + diff done in "
        f"{format_duration(eval_seconds)} (total {format_duration(total_seconds)})"
    )

    record = {
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
            "min_visits": min_visits,
            "ev_tol": ev_tol,
            "agreement_unweighted": report.agreement_unweighted,
            "agreement_weighted": report.agreement_weighted,
            "category_counts": report.category_counts,
            "cells": [asdict(cell) for cell in report.cells],
        },
        "qtable": _qtable_records(agent),
    }
    target = runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR
    run_dir = save_run(target, record)
    log(f"saved run to {run_dir}")
    return RunResult(run_dir=run_dir, agent_edge=agent_edge, basic_edge=basic_edge, diff=report)
