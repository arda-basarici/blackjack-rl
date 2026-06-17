"""Monte Carlo control trainer — where the learning happens.

Plays full episodes with the agent's current epsilon-greedy policy, then does credit
assignment: every (state, action) in a hand receives the hand's terminal reward as its return
(terminal-only reward, no discounting — every-visit Monte Carlo). Because the agent's `decide`
is epsilon-greedy over the *current* Q, policy improvement is implicit. See DESIGN.md D1.

Exploration follows ``config``'s schedule; the value update uses ``config.step_size`` if set.
With ``progress_every`` and ``on_checkpoint``, the trainer also emits a learning curve so
convergence is measured, not guessed (A10).
"""
from __future__ import annotations

import random
import sys
import time
from typing import Callable

from simulator.game_state import Action

from blackjack_rl.agents.tabular import _STAGE2_ACTIONS, TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode, problem_a_config, rollout
from blackjack_rl.schedules import make_epsilon_schedule
from blackjack_rl.state import StateKey
from blackjack_rl.util import format_duration


def _apply_episode(agent: TabularAgent, episode: Episode) -> None:
    """Credit assignment for one episode: every (state, action) receives the terminal reward
    as its return (every-visit Monte Carlo, no discounting)."""
    for state_key, action in episode.steps:
        agent.update(state_key, action, episode.reward)


def _greedy_table(agent: TabularAgent) -> dict[StateKey, Action]:
    """Deterministic greedy action per visited state (argmax over Stage-2 actions, first on
    ties). Deterministic so policy-churn doesn't pick up the random tie-break as fake change."""
    states = {state_key for (state_key, _action) in agent.n}
    return {
        s: max(_STAGE2_ACTIONS, key=lambda a: agent.q.get((s, a), 0.0))
        for s in states
    }


def _min_state_visits(agent: TabularAgent) -> int:
    """Fewest visits to any single state (the coverage bottleneck)."""
    visits: dict[StateKey, int] = {}
    for (state_key, _action), count in agent.n.items():
        visits[state_key] = visits.get(state_key, 0) + count
    return min(visits.values()) if visits else 0


def train(
    config: ExperimentConfig,
    progress_every: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> TabularAgent:
    """Train a ``TabularAgent`` by Monte Carlo control over ``config.num_episodes`` hands.

    Seeds the global RNG once so the run is reproducible, sets the exploration rate from the
    config's schedule each episode, and returns the trained agent. Every ``progress_every``
    episodes it prints progress and (if ``on_checkpoint`` is given) emits a learning-curve
    point: policy churn (greedy cells changed since the last checkpoint), min state visits,
    states covered, and current epsilon.
    """
    random.seed(config.seed)
    agent = TabularAgent(epsilon=config.epsilon, step_size=config.step_size)
    env_config = problem_a_config()
    epsilon_at = make_epsilon_schedule(
        config.epsilon_schedule,
        constant=config.epsilon,
        start=config.epsilon_start,
        end=config.epsilon_end,
        num_episodes=config.num_episodes,
    )
    total = config.num_episodes
    start = time.perf_counter()
    prev_greedy: dict[StateKey, Action] = {}

    for i in range(total):
        agent.epsilon = epsilon_at(i)
        _apply_episode(agent, rollout(agent, env_config))
        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0.0
            eta = (total - done) / rate if rate else 0.0
            greedy = _greedy_table(agent)
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  "
                f"elapsed {format_duration(elapsed)}  {rate:,.0f} hands/s  "
                f"eta {format_duration(eta)}  eps {agent.epsilon:.3f}  states {len(greedy)}",
                file=sys.stderr,
            )
            if on_checkpoint is not None:
                churn = sum(1 for s, a in greedy.items() if prev_greedy.get(s) != a)
                on_checkpoint(
                    {
                        "episode": done,
                        "epsilon": round(agent.epsilon, 4),
                        "policy_churn": churn,
                        "min_state_visits": _min_state_visits(agent),
                        "states": len(greedy),
                    }
                )
                prev_greedy = greedy
    return agent
