"""Tests for the configurable compute device (cpu / cuda / auto)."""
from __future__ import annotations

import torch

from blackjack_rl.core.config import DQNConfig


def test_default_is_cpu() -> None:
    cfg = DQNConfig(num_episodes=10)
    assert cfg.device == "cpu"
    assert cfg.resolve_device() == "cpu"


def test_auto_resolves_by_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert DQNConfig(num_episodes=10, device="auto").resolve_device() == expected


def test_cuda_without_gpu_errors() -> None:
    if torch.cuda.is_available():
        return  # a GPU is present; nothing to assert
    try:
        DQNConfig(num_episodes=10, device="cuda").resolve_device()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for device='cuda' with no GPU")


def test_bad_device_rejected() -> None:
    for bad in ("gpu", "CUDA", "tpu", ""):
        try:
            DQNConfig(num_episodes=10, device=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for device={bad!r}")


def test_cpu_training_runs_on_cpu_device(tmp_path) -> None:
    """A tiny end-to-end run with device='cpu' trains and evaluates without error."""
    from blackjack_rl.dqn.experiment import run_dqn

    cfg = DQNConfig(num_episodes=200, warmup=10, batch_size=8, buffer_capacity=500,
                    encoding="onehot", device="cpu", seed=0)
    res = run_dqn(cfg, eval_hands=200, runs_dir=tmp_path, progress_every=None, save=True)
    assert res.run_dir is not None and (res.run_dir / "model.pt").exists()
