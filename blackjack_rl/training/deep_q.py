"""Deep Q-learning trainer (CONCEPTS.md section 17).

This unit ships the second stabilizer — the **target network** — and its helpers. The TD training
loop (capture episodes, reconstruct transitions, sample minibatches, optimize) is the next unit
and will live here too.

Why a target network: a TD target ``y = r + gamma * max_a' Q(s', a')`` computed with the *same*
weights being optimized makes the goal move every gradient step — the network chases its own
estimate and can oscillate or diverge. Computing targets from a *frozen* copy, refreshed only
every C steps, gives the online network a stationary goal to descend toward between refreshes.
"""
from __future__ import annotations

import copy

from blackjack_rl.agents.dqn import QNetwork


def make_target(online: QNetwork) -> QNetwork:
    """Return a frozen copy of ``online`` for computing TD targets: identical architecture and
    weights, set to eval mode, with gradients disabled (it is never optimized — only periodically
    synced via :func:`sync_target`)."""
    target = copy.deepcopy(online)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def sync_target(target: QNetwork, online: QNetwork) -> None:
    """Hard update — copy the online weights into the target. Called every C gradient steps; in
    between, the target stays fixed so the optimization goal is stationary."""
    target.load_state_dict(online.state_dict())
