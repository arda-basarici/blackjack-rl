"""Re-evaluate the report's headline policies' house edge over many hands — fixes the 200k noise.

The per-run edge saved in each record is ONE eval_seed=0, 200k-hand sample (per-hand reward SD
~1.15 => SE ~0.26%/hand), too noisy to rank near-optimal policies, and seed 0 drew low. This RELOADS
each headline policy (no retraining) and re-evaluates over EVAL_HANDS hands, where SE falls to
~0.05%/hand at 5M. Only the policies whose edge is *shown as a claim* (the Ch1/Ch6/Ch7 scoreboards)
are here; other tables present edge as a trend or already-banded, so they don't need tightening.

Run from the repo root (needs torch + the Phase-2 simulator on the path, same as a normal run):

    python scripts/reeval_edges.py

Tune EVAL_HANDS for precision/runtime (SE ~ 1.15/sqrt(N)): 2M ~0.08%, 3M ~0.07%, 5M ~0.05%.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.tabular.agent import TabularAgent
from blackjack_rl.core.env import problem_a_config
from blackjack_rl.evaluation.metrics import GreedyPolicy, evaluate_policy

RUNS = Path("runs")
EVAL_HANDS = 5_000_000        # SE ~0.05%/hand; set 2M/3M for a faster pass
SEED = 0                      # one seed is plenty at these budgets (the seed-0 bias was a small-N artifact)

# (label, run_dir | None, kind, with_surrender_at_eval). kind: "dqn" | "tabular" | "basic".
# naive (20260619-224926) is intentionally omitted: it predates weight-saving (no model.pt), can't be
# reloaded — and its ~2% edge is clearly above optimum at any band, so it needs no re-measurement.
POLICIES = [
    ("basic (no-surrender optimum)",  None,                             "basic",   False),
    ("basic (with-surrender optimum)", None,                            "basic",   True),
    ("best DQN (trimmed)",            "20260622-015812_seed42_afbf4c5", "dqn",     False),
    ("tabular (trimmed)",             "20260617-152831_seed42_c298e9c", "tabular", False),
    ("tabular (split)",               "20260618-001215_seed42_a642d60", "tabular", False),
    ("DQN (complete [16,16])",        "20260622-144112_seed42_050857d", "dqn",     True),
    ("DQN (complete [64,64])",        "20260623-173931_seed42_050857d", "dqn",     True),
]


def load_tabular(run_dir: str) -> TabularAgent:
    """Rebuild a TabularAgent from the qtable in its record (faithful: greedy_action argmaxes over the
    LEGAL actions with these stored Q-values, mask included)."""
    rec = json.loads((RUNS / run_dir / "record.json").read_text(encoding="utf-8"))
    ws = bool(rec["config"].get("with_splits", False))
    agent = TabularAgent(epsilon=0.0, with_splits=ws)
    for e in rec["qtable"]:                              # keys match blackjack_rl.core.state.encode_state
        base = (e["player_value"], e["is_soft"], e["dealer_upcard"])
        agent.q[((*base, e["can_split"]) if ws else base, e["action"])] = e["q"]
    return agent


def load_dqn(run_dir: str):
    """Reload a trained DQN, passing BOTH with_splits AND with_surrender from the config so the net's
    action head matches the checkpoint (the repo's load_agent omits with_surrender -> 4 vs 5 action
    size-mismatch on complete-game runs)."""
    import torch
    from blackjack_rl.dqn.agent import DQNAgent
    cfg = json.loads((RUNS / run_dir / "record.json").read_text(encoding="utf-8"))["config"]
    agent = DQNAgent(epsilon=0.0,
                     with_splits=bool(cfg.get("with_splits", False)),
                     with_surrender=bool(cfg.get("with_surrender", False)),
                     hidden=tuple(cfg.get("hidden", (64, 64))),
                     encoding=cfg.get("encoding", "scalar"))
    agent.q_net.load_state_dict(torch.load(RUNS / run_dir / "model.pt", map_location="cpu"))
    agent.q_net.eval()
    return agent


def make_policy(kind: str, run_dir: str | None):
    if kind == "basic":
        return BasicStrategy()
    if kind == "dqn":
        return GreedyPolicy(load_dqn(run_dir))   # real network => faithful legal-action mask
    return GreedyPolicy(load_tabular(run_dir))


def _save(results) -> None:
    """Persist results after EVERY policy, so a mid-run crash keeps what finished (the lesson, applied)."""
    Path("reeval_results.json").write_text(
        json.dumps({"eval_hands": EVAL_HANDS, "seed": SEED, "results": results}, indent=2), encoding="utf-8")
    with open("reeval_results.txt", "w", encoding="utf-8") as f:
        f.write("re-eval over %d hands, seed %d\n\n" % (EVAL_HANDS, SEED))
        for r in results:
            f.write("%-30s edge = %7.3f%%  +/- %.3f%%\n" % (r["policy"], r["edge_pct"], r["se_pct"])
                    if "edge_pct" in r else "%-30s %s\n" % (r["policy"], r["error"]))


def main() -> None:
    print("re-evaluating %d policies over %s hands (seed %d)\n" % (len(POLICIES), f"{EVAL_HANDS:,}", SEED))
    # RESUME: keep any policy already saved with a value, and skip re-running it. So a re-run only does
    # what's missing/failed — re-running after a crash (or to add the fixed complete-game runs) is cheap.
    cached = {}
    if Path("reeval_results.json").exists():
        for r in json.loads(Path("reeval_results.json").read_text(encoding="utf-8")).get("results", []):
            if r.get("edge_pct") is not None:
                cached[r["policy"]] = r
    results = []
    for label, run_dir, kind, surr in POLICIES:
        if label in cached:
            print("%-30s edge = %7.3f%%  +/- %.3f%%   (cached)" % (label, cached[label]["edge_pct"], cached[label]["se_pct"]))
            results.append(cached[label]); _save(results); continue
        t = time.time()
        try:
            r = evaluate_policy(make_policy(kind, run_dir), n_hands=EVAL_HANDS, seed=SEED,
                                config=problem_a_config(with_surrender=surr))
            print("%-30s edge = %7.3f%%  +/- %.3f%%   (%.0fs)" % (label, r.edge * 100, r.std_error * 100, time.time() - t))
            results.append({"policy": label, "run": run_dir, "edge_pct": round(r.edge * 100, 4),
                            "se_pct": round(r.std_error * 100, 4), "n": r.n})
        except Exception as ex:
            print("%-30s FAILED: %s: %s" % (label, type(ex).__name__, ex))
            results.append({"policy": label, "run": run_dir, "error": "%s: %s" % (type(ex).__name__, ex)})
        _save(results)   # write after each policy
    print("\nsaved -> reeval_results.json / reeval_results.txt")


if __name__ == "__main__":
    main()
