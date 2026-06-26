"""Tests for the configurable torch thread count (num_threads)."""
from __future__ import annotations

import os

from blackjack_rl.core.config import DQNConfig


def test_default_is_single_thread() -> None:
    assert DQNConfig(num_episodes=10).num_threads == 1
    assert DQNConfig(num_episodes=10).torch_threads() == 1


def test_explicit_and_all_cores_resolution() -> None:
    assert DQNConfig(num_episodes=10, num_threads=4).torch_threads() == 4
    assert DQNConfig(num_episodes=10, num_threads=0).torch_threads() == (os.cpu_count() or 1)


def test_negative_threads_rejected() -> None:
    for bad in (-1, -8):
        try:
            DQNConfig(num_episodes=10, num_threads=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for num_threads={bad}")
