"""Outcome-axis evaluation — the house edge of a fixed policy, measured through the engine.

Runs a policy for many hands and reports the per-hand house edge with its standard error.
Every policy is measured through this same harness, so the trained agent and BasicStrategy are
directly comparable; BasicStrategy's number is our in-harness anchor (~0.5% per hand, the
per-wagered 0.45% from Phase 2 under the doubling adjustment). See DESIGN.md section 7.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from simulator.config import SimulatorConfig
from simulator.game_state import Action, GameState
from strategies.base import Strategy

from blackjack_rl.core.env import problem_a_config, rollout


class _GreedyAgent(Protocol):
    """An agent that can expose a greedy (no-exploration) action and a name."""

    def greedy_action(self, state: GameState) -> Action: ...

    def name(self) -> str: ...


class GreedyPolicy(Strategy):
    """Presents an agent's greedy *target* policy as a Strategy, so evaluation measures the
    greedy policy without mutating the agent's exploration rate."""

    def __init__(self, agent: _GreedyAgent) -> None:
        self._agent = agent

    def decide(self, state: GameState) -> Action:
        return self._agent.greedy_action(state)

    def name(self) -> str:
        return f"greedy({self._agent.name()})"


@dataclass(frozen=True)
class EdgeResult:
    """House edge over ``n`` hands, per hand, in bet units (flat bet = 1).

    ``edge`` is the player's expected loss as a positive fraction (= -mean_reward);
    ``std_error`` is the standard error of the mean reward.
    """

    edge: float
    std_error: float
    mean_reward: float
    n: int


def evaluate_policy(
    policy: Strategy,
    n_hands: int = 200_000,
    seed: int = 0,
    config: SimulatorConfig | None = None,
) -> EdgeResult:
    """Play ``n_hands`` with ``policy`` and return the per-hand house edge and its std error.

    Seeds the global RNG once for reproducibility. Edge = -mean(reward); the mean and variance
    are accumulated with Welford's algorithm (numerically stable, single pass).
    """
    random.seed(seed)
    env_config = config if config is not None else problem_a_config()

    count = 0
    mean = 0.0
    m2 = 0.0
    for _ in range(n_hands):
        reward = rollout(policy, env_config).reward
        count += 1
        delta = reward - mean
        mean += delta / count
        m2 += delta * (reward - mean)

    variance = m2 / (count - 1) if count > 1 else 0.0
    std_error = (variance / count) ** 0.5 if count > 0 else 0.0
    return EdgeResult(edge=-mean, std_error=std_error, mean_reward=mean, n=count)
