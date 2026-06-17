"""ExperimentConfig — the knobs for a single training run.

Holds only the hyperparameters you actually choose, so the trainer reads from here instead of
hardcoding, and the whole thing is saved with each run for reproducibility (D8). Kept
deliberately minimal (D9): a knob is added when a stage needs it, not before.

Exploration can be constant (``epsilon``) or a decaying schedule (``epsilon_schedule`` with
``epsilon_start`` -> ``epsilon_end``); constant remains the default, so older fixed-epsilon
runs stay reproducible.

Deliberately NOT here, and why:
  * discount factor (gamma) — terminal-only reward, so the return is just the final payout.
  * state-feature flags — A's state is fixed; the toggle arrives with Problem B's count.
  * algorithm name / env ruleset — provenance, recorded in the run's metadata, not tuned.
"""
from __future__ import annotations

from dataclasses import dataclass

from blackjack_rl.schedules import KINDS


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable hyperparameters for one training run.

    num_episodes     : number of hands (episodes) to train on.
    epsilon          : exploration rate when ``epsilon_schedule == "constant"``.
    epsilon_schedule : "constant" | "linear" | "exponential" | "harmonic".
    epsilon_start    : start rate for a decaying schedule.
    epsilon_end      : end rate for a decaying schedule (reached at the final episode).
    seed             : seed for the global RNG (Phase 2 convention: 42).

    We train with exploration but EVALUATE greedily (epsilon = 0), so exploration only shapes
    what gets sampled, never the reported policy. See CONCEPTS.md #15.
    """

    num_episodes: int
    epsilon: float = 0.1
    epsilon_schedule: str = "constant"
    epsilon_start: float = 0.3
    epsilon_end: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {self.num_episodes}")
        if self.epsilon_schedule not in KINDS:
            raise ValueError(f"epsilon_schedule must be one of {KINDS}, got {self.epsilon_schedule!r}")
        for name, value in (
            ("epsilon", self.epsilon),
            ("epsilon_start", self.epsilon_start),
            ("epsilon_end", self.epsilon_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
