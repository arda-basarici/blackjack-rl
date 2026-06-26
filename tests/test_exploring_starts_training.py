"""Tests for the exploring-starts enumeration, rollout, and training loop (MC-ES control)."""
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import problem_a_config
from blackjack_rl.tabular.agent import TabularAgent
from blackjack_rl.tabular.exploring_starts import (
    enumerate_start_pairs,
    es_rollout,
    train_exploring_starts,
)


# --- enumeration ---------------------------------------------------------------------------------

def test_enumerate_with_splits_counts_and_legality() -> None:
    pairs = enumerate_start_pairs(with_splits=True)
    specs = {s for s, _ in pairs}
    splits = [(s, a) for s, a in pairs if a == "split"]
    assert len(specs) == 330                      # 33 states x 10 upcards
    assert len(pairs) == 1090                     # 690 non-pair*3 + 100 pair*4
    assert len(splits) == 100                     # split forced only on pair states
    assert all(s[2] for s, a in splits)
    assert not any(a == "surrender" for _, a in pairs)
    assert ((16, False, False, 10), "split") not in pairs   # never split a non-pair

def test_enumerate_no_splits_has_no_split_action() -> None:
    pairs = enumerate_start_pairs(with_splits=False)
    assert pairs and not any(a == "split" for _, a in pairs)
    assert len({s for s, _ in pairs}) == 230      # hard 5-19 (15) + soft 13-20 (8) = 23, x10


# --- rollout -------------------------------------------------------------------------------------

def test_es_rollout_forces_first_action_and_state() -> None:
    agent = TabularAgent(epsilon=0.0, step_size=0.001, with_splits=True)
    ep = es_rollout(agent, (16, False, False, 6), "double", problem_a_config())
    assert ep is not None
    key, action, _payout = ep.steps[0]
    assert action == "double"
    assert key == (16, False, 6, False)           # key order: pv, soft, upcard, can_split

def test_es_rollout_split_seeds_a_split_decision() -> None:
    agent = TabularAgent(epsilon=0.0, step_size=0.001, with_splits=True)
    ep = es_rollout(agent, (16, False, True, 6), "split", problem_a_config())  # 8,8 vs 6
    assert ep is not None
    assert ep.steps[0][1] == "split"
    assert ep.steps[0][0] == (16, False, 6, True)


# --- training loop: forced coverage --------------------------------------------------------------

def test_training_covers_every_start_state() -> None:
    cfg = ExperimentConfig(num_episodes=60_000, step_size=0.001, with_splits=True, seed=42)
    agent = train_exploring_starts(cfg)
    visited = {key for (key, _a) in agent.n}
    start_keys = {(pv, soft, up, csp)
                  for (pv, soft, csp, up) in {s for s, _ in enumerate_start_pairs(True)}}
    assert start_keys <= visited                   # every forced start state was reached


def test_run_exploring_starts_saves_record(tmp_path) -> None:
    import json
    from blackjack_rl.tabular.exploring_starts import run_exploring_starts
    cfg = ExperimentConfig(num_episodes=500, epsilon=0.0, step_size=0.001, with_splits=True, seed=42)
    result = run_exploring_starts(cfg, eval_hands=500, eval_seed=0, runs_dir=tmp_path)
    assert result.run_dir.exists()
    rec = json.loads((result.run_dir / "record.json").read_text())
    assert rec["method"] == "exploring_starts"
    assert rec["config"]["with_splits"] is True and rec["config"]["epsilon"] == 0.0
    assert rec["qtable"] and "learning_curve" in rec
