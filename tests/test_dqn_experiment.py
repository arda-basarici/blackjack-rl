"""Smoke test for the DQN runner (dqn_experiment.run_dqn): a tiny end-to-end train + eval + diff
with no persistence. Not a quality check — just that the orchestration wires together and returns
a sane result."""
from __future__ import annotations

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.core.config import DQNConfig
from blackjack_rl.dqn.experiment import run_dqn
from blackjack_rl.evaluation.policy_diff import DiffReport


def test_run_dqn_end_to_end_no_save() -> None:
    cfg = DQNConfig(num_episodes=800, warmup=100, batch_size=64, seed=0)
    result = run_dqn(cfg, eval_hands=2000, eval_seed=0, save=False)

    assert result.run_dir is None
    assert isinstance(result.agent, DQNAgent)
    assert isinstance(result.diff, DiffReport)
    assert len(result.diff.cells) == 240  # full grid materialized from the net
    assert result.diff.category_counts.get("under_visited", 0) == 0  # N/A for a network
    # edges are per-hand fractions; just a sanity band (2k hands is noisy)
    assert -0.5 < result.agent_edge.edge < 0.5
    assert -0.5 < result.basic_edge.edge < 0.5
