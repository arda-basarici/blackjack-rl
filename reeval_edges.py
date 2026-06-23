"""Re-evaluate trained policies' house edge over many hands — fixes the 200k single-seed noise.

The per-run edge saved in each record is ONE eval_seed=0, 200k-hand sample (per-hand reward SD
~1.15 => SE ~0.26%/hand, ~+/-0.5% at 95%). That is far too noisy to rank near-optimal policies, and
seed 0 happens to draw low. This script RELOADS each saved policy (no retraining) and re-evaluates
its edge over EVAL_HANDS hands, where the standard error falls to ~0.08%/hand — tight enough to place
each policy against the ~0.54% (full) / ~1.11% (no-split) basic-strategy optimum.

Run from the repo root (needs torch + the Phase-2 simulator on the path, same as a normal run):

    python reeval_edges.py

Reduce EVAL_HANDS or trim POLICIES if you want it faster; ~1.5M hands is a few minutes per policy.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.env import problem_a_config
from blackjack_rl.evaluation.metrics import GreedyPolicy, evaluate_policy

RUNS = Path("runs")
EVAL_HANDS = 1_500_000        # SE ~0.08%/hand at this budget (vs ~0.26% at 200k)
SEEDS = [0]                   # one seed is enough at 1.5M; add e.g. [0, 1] for a cross-check

# (label, run_dir, kind, with_surrender_at_eval). kind: "dqn" | "tabular" | "basic".
# with_surrender must match how the run was evaluated (problem_a_config(with_surrender=...)).
POLICIES = [
    # naive DQN (20260619-224926) is intentionally omitted: it predates weight-saving (no model.pt),
    # so it can't be reloaded — and its ~2% edge is already clearly above optimum (>2 SE even at 200k),
    # so it needs no re-measurement. Keep its 200k value, banded, in the scoreboard.
    ("best DQN (trimmed)",       "20260622-015812_seed42_afbf4c5", "dqn",     False),
    ("tabular (trimmed)",        "20260617-152831_seed42_c298e9c", "tabular", False),
    ("tabular (split)",          "20260618-001215_seed42_a642d60", "tabular", False),
    ("DQN (complete [16,16])",   "20260622-144112_seed42_050857d", "dqn",     True),
    ("DQN (complete [64,64])",   "20260623-173931_seed42_050857d", "dqn",     True),
    ("basic (full optimum)",     None,                             "basic",   False),
    ("basic (complete, +surr)",  None,                             "basic",   True),
]


def load_tabular(run_dir: str) -> TabularAgent:
    """Rebuild a TabularAgent from the qtable stored in its record (faithful: its greedy_action
    argmaxes over the LEGAL actions using these stored Q-values, mask included)."""
    rec = json.loads((RUNS / run_dir / "record.json").read_text(encoding="utf-8"))
    with_splits = bool(rec["config"].get("with_splits", False))
    agent = TabularAgent(epsilon=0.0, with_splits=with_splits)
    for e in rec["qtable"]:                              # keys match blackjack_rl.state.encode_state
        base = (e["player_value"], e["is_soft"], e["dealer_upcard"])
        key = (*base, e["can_split"]) if with_splits else base
        agent.q[(key, e["action"])] = e["q"]
    return agent


def make_policy(kind: str, run_dir: str | None):
    if kind == "basic":
        return BasicStrategy()
    if kind == "dqn":
        from blackjack_rl.evaluation.embedding import load_agent  # lazy: only DQN needs torch
        return GreedyPolicy(load_agent(RUNS / run_dir))   # real network => faithful legal-action mask
    if kind == "tabular":
        return GreedyPolicy(load_tabular(run_dir))
    raise ValueError(kind)


def main() -> None:
    print(f"re-evaluating {len(POLICIES)} policies over {EVAL_HANDS:,} hands, seeds {SEEDS}")
    print(f"{'policy':28s} {'edge %/hand':>12s} {'+/- SE':>8s}   per-seed")
    print("-" * 70)
    for label, run_dir, kind, surr in POLICIES:
        cfg = problem_a_config(with_surrender=surr)
        try:
            pol = make_policy(kind, run_dir)
            res = [evaluate_policy(pol, n_hands=EVAL_HANDS, seed=sd, config=cfg) for sd in SEEDS]
            edges = [r.edge * 100 for r in res]
            mean = st.mean(edges)
            se = (st.pstdev(edges) / len(edges) ** 0.5) if len(edges) > 1 else res[0].std_error * 100
            print(f"{label:28s} {mean:12.3f} {se:8.3f}   {[round(e, 3) for e in edges]}")
        except Exception as ex:  # one policy failing must not lose the others
            print(f"{label:28s} {'FAILED':>12s}           {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
