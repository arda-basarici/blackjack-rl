"""Monte Carlo control trainer — where the learning happens.

Plays full episodes with the agent's current epsilon-greedy policy, then does credit
assignment: every (state, action) in a hand receives the hand's terminal reward as its return
(terminal-only reward, no discounting — every-visit Monte Carlo). Because the agent's `decide`
is epsilon-greedy over the *current* Q, policy improvement is implicit. See DESIGN.md D1.

Exploration follows ``config``'s schedule; the value update uses ``config.step_size`` (constant
alpha, recency-weighted) if set, else a sample average. See A8.
"""
from __future__ import annotations

import random
import sys
import time

from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode, problem_a_config, rollout
from blackjack_rl.schedules import make_epsilon_schedule
from blackjack_rl.util import format_duration


def _apply_episode(agent: TabularAgent, episode: Episode) -> None:
    """Credit assignment for one episode: every (state, action) receives the terminal reward
    as its return (every-visit Monte Carlo, no discounting)."""
    for state_key, action in episode.steps:
        agent.update(state_key, action, episode.reward)


def train(config: ExperimentConfig, progress_every: int | None = None) -> TabularAgent:
    """Train a ``TabularAgent`` by Monte Carlo control over ``config.num_episodes`` hands.

    Seeds the global RNG once so the run is reproducible, sets the exploration rate from the
    config's schedule each episode, and returns the trained agent. If ``progress_every`` is
    set, prints progress (fraction, elapsed, rate, ETA, current epsilon, states discovered).
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

    for i in range(total):
        agent.epsilon = epsilon_at(i)
        _apply_episode(agent, rollout(agent, env_config))
        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0.0
            eta = (total - done) / rate if rate else 0.0
            states = len({state_key for (state_key, _action) in agent.n})
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  "
                f"elapsed {format_duration(elapsed)}  {rate:,.0f} hands/s  "
                f"eta {format_duration(eta)}  eps {agent.epsilon:.3f}  states {states}",
                file=sys.stderr,
            )
    return agent
