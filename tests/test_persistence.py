"""Tests for run persistence: records round-trip to disk, never overwriting (D8)."""
from __future__ import annotations

import json

from blackjack_rl.persistence import git_hash, save_run


def test_git_hash_is_a_string():
    h = git_hash()
    assert isinstance(h, str) and h != ""


def test_save_run_roundtrips_and_stamps_provenance(tmp_path):
    record = {
        "config": {"num_episodes": 1000, "epsilon": 0.1, "seed": 42},
        "metrics": {"house_edge": 0.0045},
    }
    run_dir = save_run(tmp_path, record, run_id="testrun")
    loaded = json.loads((run_dir / "record.json").read_text())
    assert loaded["run_id"] == "testrun"
    assert "timestamp" in loaded
    assert "git_hash" in loaded
    assert loaded["config"]["seed"] == 42
    assert loaded["metrics"]["house_edge"] == 0.0045


def test_save_run_never_overwrites_even_with_same_run_id(tmp_path):
    record = {"config": {"seed": 1}}
    d1 = save_run(tmp_path, record, run_id="same")
    d2 = save_run(tmp_path, record, run_id="same")
    assert d1 != d2
    assert (d1 / "record.json").exists() and (d2 / "record.json").exists()
    assert d1.name == "same" and d2.name == "same-1"


def test_default_run_id_contains_seed(tmp_path):
    run_dir = save_run(tmp_path, {"config": {"seed": 7}})
    assert "seed7" in run_dir.name
