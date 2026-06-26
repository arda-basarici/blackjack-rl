"""Training loops & orchestration for Problem B (build stages B2–B4).

- `train_bet`        — the bet model on log-growth, with fixed basic play (B2).
- `run_factored`     — assemble count-aware play (dqn) + bet model into the factored policy (B3).
- `train_monolithic` — the end-to-end baseline (B4).
Persists runs via core.persistence (record + model), like the A/DQN experiments.
"""
from __future__ import annotations


def train_bet(config):  # -> BetAgent
    raise NotImplementedError("B2: log-growth training of the discrete-spread bet model")


def run_factored(config):
    raise NotImplementedError("B3: factored play(EV, count-aware) + bet(Kelly) orchestration")


def train_monolithic(config):  # -> MonolithicAgent
    raise NotImplementedError("B4: end-to-end play+bet on log-growth")
