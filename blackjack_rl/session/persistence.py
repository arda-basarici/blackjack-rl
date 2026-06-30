"""Persist & reload a trained bet model (``BetAgent``).

Mirrors the A/DQN convention: ``core.persistence.save_run`` writes ``record.json`` (config + metrics +
auto-stamped provenance) into a fresh run dir, and the weights go beside it as ``model.pt`` — the same
shape ``dqn.embedding.load_agent`` reads back. Save the weights once, then reuse the agent across every
downstream wiring (four-axis eval, figures, B3 factored play) without retraining.

The ``record.json`` has two distinct blocks:
- ``construction`` — the minimal spec to rebuild the agent *shell* before loading weights (levels set
  the output size + the action-index → wager map; ``num_decks`` / ``bankroll_scale`` are the encoder's
  normalizers). This is what :func:`load_bet_agent` needs.
- ``config`` — the full ``BetTrainConfig`` as provenance (how the run was produced; reproducible by
  construction). Not needed to *load*, only to *trust/regenerate*.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from blackjack_rl.core.persistence import save_run
from blackjack_rl.session.bet_agent import BetAgent
from blackjack_rl.session.train import BetTrainConfig


def construction_of(agent: BetAgent) -> dict[str, Any]:
    """The minimal spec to rebuild ``agent``'s shell before loading its weights."""
    return {
        "levels": list(agent.levels),
        "hidden": list(agent.hidden),
        "num_decks": agent.num_decks,
        "bankroll_scale": agent.bankroll_scale,
    }


def save_bet_run(
    runs_root: Path | str, agent: BetAgent, config: BetTrainConfig, metrics: dict, run_id: str | None = None
) -> Path:
    """Persist a trained bettor: ``record.json`` (construction + full config + metrics + stamped
    provenance) beside ``model.pt`` (weights). Returns the run dir. ``config`` (the ``BetTrainConfig``
    that produced the run) is stored verbatim as provenance — write-only: :func:`load_bet_agent` rebuilds
    from ``construction``, never from ``config`` — via :func:`dataclasses.asdict`."""
    record = {
        "kind": "bet_agent",
        "construction": construction_of(agent),
        "config": asdict(config),
        "metrics": metrics,
    }
    run_dir = save_run(runs_root, record, run_id=run_id)
    torch.save(agent.q_net.state_dict(), run_dir / "model.pt")
    return run_dir


def _load_record(run_dir: Path) -> dict[str, Any]:
    """Read a run's ``record.json`` and validate at the boundary (every run dir looks alike)."""
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    if record.get("kind") != "bet_agent":
        raise ValueError(f"{run_dir} is not a bet_agent run (kind={record.get('kind')!r})")
    return record


def _rebuild_and_load(record: dict[str, Any], weights: Path) -> BetAgent:
    """Rebuild the agent *shell* from the record's ``construction`` block, load ``weights`` onto CPU, and
    return it greedy (``epsilon=0``, eval mode) — the deterministic policy for evaluation."""
    c = record["construction"]
    agent = BetAgent(
        levels=c["levels"],
        hidden=tuple(c["hidden"]),
        epsilon=0.0,
        num_decks=c["num_decks"],
        bankroll_scale=c["bankroll_scale"],
    )
    agent.q_net.load_state_dict(torch.load(weights, map_location="cpu"))
    agent.q_net.eval()
    return agent


def load_bet_agent(run_dir: Path | str) -> BetAgent:
    """Reconstruct the *final* trained ``BetAgent`` from a saved run dir (``record.json`` + ``model.pt``).
    Mirrors ``dqn.embedding.load_agent``."""
    run_dir = Path(run_dir)
    return _rebuild_and_load(_load_record(run_dir), run_dir / "model.pt")


def load_bet_checkpoint(run_dir: Path | str, session: int) -> BetAgent:
    """Reconstruct a *mid-training* ``BetAgent`` from a saved trajectory checkpoint
    (``checkpoints/ckpt_<session>.pt``) — same shell as the final model, the weights at that session.

    Used to interrogate a specific point in the training orbit (e.g. a near-Kelly ramp the final policy
    didn't keep — Test 11 / H3). Fails loudly with the available sessions if the checkpoint is absent."""
    run_dir = Path(run_dir)
    weights = run_dir / "checkpoints" / f"ckpt_{session:05d}.pt"
    if not weights.exists():
        ckpt_dir = run_dir / "checkpoints"
        have = sorted(p.stem.removeprefix("ckpt_") for p in ckpt_dir.glob("ckpt_*.pt")) if ckpt_dir.exists() else []
        raise FileNotFoundError(f"no checkpoint for session {session} in {run_dir} (have: {have})")
    return _rebuild_and_load(_load_record(run_dir), weights)
