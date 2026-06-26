"""Session environment — the Problem B MDP (DESIGN D14, build stage B0).

Many hands from one persisting, depleting shoe (counting on), against a finite bankroll with a
hard ruin barrier. Reward is the per-hand log-wealth increment `log(W_after / W_before)` so the
return is log-growth (Kelly). Reuses the Phase-2 engine's counting/session fields
(`true_count`, `decks_remaining`, `bankroll`, `current_bet`) — no engine changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionConfig:
    """Knobs for a Problem B session. Representative fields only — finalized at B0.

    (Home TBD: here vs core/config.py alongside ExperimentConfig/DQNConfig — decide at B0.)
    """
    # shoe / table
    penetration: float = 0.75          # reshuffle once this fraction of the shoe is dealt
    # bankroll / risk
    starting_bankroll: float = 100.0   # in min-bet units
    ruin_threshold: float = 1.0        # bankroll at/below this = ruin (terminal)
    # betting
    bet_spread: tuple[int, ...] = (1, 2, 4, 8)   # discrete unit-bet levels (D15)
    # horizon / reproducibility
    hands_per_episode: int | None = None         # None = until reshuffle (one shoe)
    seed: int = 0
    # TODO (B0): finalize fields; wire to problem_b_config().


def problem_b_config():  # -> SimulatorConfig
    """Phase-2 SimulatorConfig for Problem B: 6-deck S17 3:2, **counting on, shoe persists**.

    Mirror of core.env.problem_a_config but with counting enabled and no per-hand reshuffle.
    """
    raise NotImplementedError("B0: build problem_b_config (counting on, persistent shoe)")


class SessionEnv:
    """Drives a session: per hand → observe (count, bankroll) → bet → play → settle → update
    bankroll; terminate on ruin or horizon. Emits per-hand log-wealth increments for training."""

    def __init__(self, config: SessionConfig) -> None:
        raise NotImplementedError("B0: session loop, bankroll bookkeeping, ruin, reward")
