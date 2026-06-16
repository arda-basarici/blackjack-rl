"""Monte Carlo control trainer — where the learning happens.

Plays full episodes with the agent's current epsilon-greedy policy, then does credit
assignment: every (state, action) in a hand receives the hand's terminal reward as its return
(terminal-only reward, no discounting — every-visit Monte Carlo). Because the agent's `decide`
is epsilon-greedy over the *current* Q, policy improvement is implicit: as Q updates, the
policy it induces improves. See DESIGN.md D1 / Stage 2.

`train` is pure training (seeding + the loop); evaluation and run persistence are separate
(Stage 2's evaluation sub-unit).
"""
from __future__ import annotations

import random

from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode, problem_a_config, rollout


def _apply_episode(agent: TabularAgent, episode: Episode) -> None:
    """Credit assignment for one episode: every (state, action) receives the terminal reward
    as its return (every-visit Monte Carlo, no discounting)."""
    for state_key, action in episode.steps:
        agent.update(state_key, action, episode.reward)


def train(config: ExperimentConfig) -> TabularAgent:
    """Train a ``TabularAgent`` by Monte Carlo control over ``config.num_episodes`` hands.

    Seeds the global RNG once (covering both the env's shuffles and the agent's epsilon-greedy
    draws) so the run is reproducible, then returns the trained agent.
    """
    random.seed(config.seed)
    agent = TabularAgent(epsilon=config.epsilon)
    env_config = problem_a_config()
    for _ in range(config.num_episodes):
        episode = rollout(agent, env_config)
        _apply_episode(agent, episode)
    return agent
