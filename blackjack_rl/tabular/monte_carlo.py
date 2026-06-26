"""Monte Carlo control trainer — where the learning happens.

Plays full episodes with the agent's current epsilon-greedy policy, then does credit
assignment by applying each decision's own return (the env/engine attributes per sub-hand; the
split decision gets the net — A11/b). Terminal-only reward, no discounting (every-visit MC).
Because the agent's `decide` is epsilon-greedy over the *current* Q, policy improvement is
implicit. See DESIGN.md D1.

Exploration follows ``config``'s schedule; the value update uses ``config.step_size`` if set;
``config.with_splits`` enables the split action + pair-aware state. With ``progress_every`` and
``on_checkpoint`` it emits a learning curve so convergence is measured, not guessed (A10).
"""
from __future__ import annotations

import random
import sys
import time
from typing import Callable

from simulator.game_state import Action

from blackjack_rl.tabular.agent import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode, problem_a_config, rollout
from blackjack_rl.schedules import make_epsilon_schedule
from blackjack_rl.state import StateKey
from blackjack_rl.util import format_duration


def _apply_episode(agent: TabularAgent, episode: Episode) -> None:
    """Credit assignment: apply each decision's *own* return. The engine attributes per
    sub-hand (split decision = net), so for single-chain hands every step's return is the hand
    total — no-split behaviour is unchanged."""
    for state_key, action, step_return in episode.steps:
        agent.update(state_key, action, step_return)


def _greedy_table(agent: TabularAgent) -> dict[StateKey, Action]:
    """Deterministic greedy action per visited state — argmax over the actions actually tried
    there, ties broken by action name (so the random tie-break isn't read as churn). Works for
    no-split and split keys alike without special-casing legality."""
    best: dict[StateKey, tuple[float, Action]] = {}
    for (key, action), q in agent.q.items():
        candidate = (q, action)
        if key not in best or candidate > best[key]:
            best[key] = candidate
    return {key: action for key, (_q, action) in best.items()}


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
    schedule each episode, and returns the trained agent. Every ``progress_every`` episodes it
    prints progress and (if ``on_checkpoint`` is given) emits a learning-curve point.
    """
    random.seed(config.seed)
    agent = TabularAgent(
        epsilon=config.epsilon, step_size=config.step_size, with_splits=config.with_splits
    )
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
        _apply_episode(agent, rollout(agent, env_config, config.with_splits))
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
