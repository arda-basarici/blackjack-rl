"""Tests for blackjack_rl.tabular.experiment — orchestration mechanics on a tiny config."""
import json

from blackjack_rl.core.config import ExperimentConfig
from blackjack_rl.tabular.experiment import RunResult, run_experiment


def test_run_experiment_writes_full_record(tmp_path) -> None:
    result = run_experiment(
        ExperimentConfig(num_episodes=500, seed=1),
        eval_hands=500,
        eval_seed=2,
        min_visits=10,
        ev_tol=0.02,
        runs_dir=tmp_path,
    )
    assert isinstance(result, RunResult)
    assert result.run_dir.exists()

    record = json.loads((result.run_dir / "record.json").read_text())
    for section in ("config", "eval", "timing", "metrics", "diff", "qtable", "run_id", "git_hash"):
        assert section in record
    for key in ("started_at", "finished_at", "train_seconds", "eval_seconds", "total_seconds"):
        assert key in record["timing"]
    assert record["metrics"]["agent"]["n"] == 500
    assert record["metrics"]["basic"]["n"] == 500
    assert record["diff"]["cells"]
    assert record["qtable"]
    assert result.agent_edge.edge == record["metrics"]["agent"]["edge"]
