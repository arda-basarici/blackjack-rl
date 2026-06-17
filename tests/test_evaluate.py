"""Tests for the load + re-evaluate path (no retraining)."""
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.experiment import _qtable_records, load_agent
from blackjack_rl.persistence import load_record, save_run


def test_load_agent_roundtrips_qtable() -> None:
    original = TabularAgent()
    original.q[((20, False, 10), "stand")] = 0.7
    original.n[((20, False, 10), "stand")] = 1234
    original.q[((11, False, 6), "double")] = 0.5
    original.n[((11, False, 6), "double")] = 99
    record = {"qtable": _qtable_records(original)}
    loaded = load_agent(record)
    assert loaded.q == original.q
    assert loaded.n == original.n


def test_load_agent_roundtrips_split_qtable() -> None:
    original = TabularAgent(with_splits=True)
    original.q[((16, False, 10, True), "split")] = 0.4
    original.n[((16, False, 10, True), "split")] = 321
    original.q[((11, False, 6, False), "double")] = 0.5
    original.n[((11, False, 6, False), "double")] = 88
    record = {"config": {"with_splits": True}, "qtable": _qtable_records(original)}
    loaded = load_agent(record)
    assert loaded.with_splits is True
    assert loaded.q == original.q
    assert loaded.n == original.n


def test_load_record_reads_saved_run(tmp_path) -> None:
    run_dir = save_run(tmp_path, {"config": {"seed": 1}, "qtable": []})
    record = load_record(run_dir)
    assert record["config"]["seed"] == 1
    record2 = load_record(run_dir / "record.json")
    assert record2["run_id"] == record["run_id"]
