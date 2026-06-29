"""Round-trip tests for bettor persistence (save_bet_run / load_bet_agent)."""
from __future__ import annotations

import pytest
import torch

from blackjack_rl.session.bet_agent import BetAgent
from blackjack_rl.session.env import growth_config
from blackjack_rl.session.persistence import construction_of, load_bet_agent, save_bet_run
from blackjack_rl.session.train import BetTrainConfig

_STATES = [
    {"true_count": tc, "decks_remaining": 3.0, "bankroll": 400.0} for tc in (-6, -2, 0, 3, 6, 9)
]


def _make_agent() -> BetAgent:
    torch.manual_seed(0)
    return BetAgent(levels=(1, 2, 4, 8), hidden=(32, 16), num_decks=6.0, bankroll_scale=400.0)


def test_save_then_load_reproduces_policy(tmp_path):
    agent = _make_agent()
    cfg = BetTrainConfig(session=growth_config(), n_sessions=5, gamma=0.0)
    run_dir = save_bet_run(tmp_path, agent, cfg, metrics={"final_curve": {"0": 1.0}}, run_id="t")

    assert (run_dir / "record.json").exists()
    assert (run_dir / "model.pt").exists()

    loaded = load_bet_agent(run_dir)
    # construction restored exactly
    assert loaded.levels == (1.0, 2.0, 4.0, 8.0)
    assert loaded.hidden == (32, 16)
    assert loaded.num_decks == 6.0 and loaded.bankroll_scale == 400.0
    # identical greedy bets everywhere -> same weights loaded
    for s in _STATES:
        assert agent.bet(**s) == loaded.bet(**s)


def test_record_carries_provenance_and_metrics(tmp_path):
    import json

    agent = _make_agent()
    cfg = BetTrainConfig(session=growth_config(), n_sessions=5, gamma=0.0, seed=7)
    run_dir = save_bet_run(tmp_path, agent, cfg, metrics={"k": 42}, run_id="prov")
    record = json.loads((run_dir / "record.json").read_text())

    assert record["kind"] == "bet_agent"
    assert "git_hash" in record and "timestamp" in record  # stamped by save_run
    assert record["metrics"]["k"] == 42
    assert record["config"]["seed"] == 7  # full BetTrainConfig persisted
    assert record["construction"] == construction_of(agent)


def test_loaded_agent_is_greedy(tmp_path):
    agent = _make_agent()
    cfg = BetTrainConfig(session=growth_config(), n_sessions=5)
    run_dir = save_bet_run(tmp_path, agent, cfg, metrics={}, run_id="eps")
    assert load_bet_agent(run_dir).epsilon == 0.0


def test_load_rejects_non_bet_run(tmp_path):
    """A run dir from another experiment (same record.json/model.pt shape) fails clearly, not cryptically."""
    import json

    rd = tmp_path / "other"
    rd.mkdir()
    (rd / "record.json").write_text(json.dumps({"kind": "dqn", "construction": {}}))
    with pytest.raises(ValueError, match="not a bet_agent run"):
        load_bet_agent(rd)
