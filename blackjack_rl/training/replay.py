"""Experience replay — the first stabilizer of deep Q-learning (CONCEPTS.md section 17).

Training on transitions in the order they occur feeds the network highly correlated samples
(consecutive decisions in one hand) and lets it forget older experience. A replay buffer breaks
both: every transition is stored, and training draws *uniform random minibatches*, decorrelating
the data and reusing each transition many times.

A ``Transition`` carries (state, action, reward, next_state, done, next_legal_mask). The
``next_legal_mask`` is load-bearing: the TD target maxes over the *legal* next actions only, so
the buffer must remember which actions were legal in s' — otherwise the net can bootstrap from the
value of an action that isn't even allowed there. States are pre-encoded feature vectors
(model-facing); the trainer (next unit) fills these in from captured episodes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Transition:
    """One (s, a, r, s', done) step plus the legal-action mask of s'.

    ``state`` / ``next_state`` are encoded feature vectors (shape ``[in_dim]``); ``action`` is an
    index into the agent's action set; ``next_legal_mask`` is a bool vector (shape ``[n_actions]``)
    that is True where the action is legal in s'. For terminal steps (``done``) the next_* fields
    are unused (the target is just the reward).
    """

    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    next_legal_mask: torch.Tensor


@dataclass(frozen=True)
class Batch:
    """A stacked minibatch ready for the loss — every field has a leading batch dimension B."""

    states: torch.Tensor            # [B, in_dim]  float
    actions: torch.Tensor           # [B]          long
    rewards: torch.Tensor           # [B]          float
    next_states: torch.Tensor       # [B, in_dim]  float
    dones: torch.Tensor             # [B]          bool
    next_legal_masks: torch.Tensor  # [B, n_actions] bool

    def to(self, device) -> "Batch":
        """Return a copy with every tensor moved to ``device`` (the buffer stays on CPU; only the
        sampled minibatch is moved per gradient step, which is the standard DQN pattern)."""
        return Batch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
            next_legal_masks=self.next_legal_masks.to(device),
        )


class ReplayBuffer:
    """A fixed-capacity ring buffer of ``Transition``s with uniform random sampling.

    Once ``capacity`` is reached the oldest transition is overwritten. ``sample`` returns a stacked
    ``Batch``; the trainer calls ``can_sample`` first and waits until enough transitions have
    accumulated to fill a batch. Sampling uses the global RNG, seeded once per run (the project
    reproducibility convention)."""

    def __init__(self, capacity: int = 50_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf: list[Transition] = []
        self._pos = 0  # next write position (ring)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self.capacity

    def can_sample(self, batch_size: int) -> bool:
        return len(self._buf) >= batch_size

    def sample(self, batch_size: int) -> Batch:
        """A uniform random minibatch (distinct items within the batch)."""
        items = random.sample(self._buf, batch_size)
        return Batch(
            states=torch.stack([t.state for t in items]),
            actions=torch.tensor([t.action for t in items], dtype=torch.long),
            rewards=torch.tensor([t.reward for t in items], dtype=torch.float32),
            next_states=torch.stack([t.next_state for t in items]),
            dones=torch.tensor([t.done for t in items], dtype=torch.bool),
            next_legal_masks=torch.stack([t.next_legal_mask for t in items]),
        )
