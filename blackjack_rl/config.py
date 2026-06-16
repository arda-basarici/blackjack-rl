"""ExperimentConfig — the knobs for a single training run.

Holds only the hyperparameters you actually choose, so the trainer reads from here instead of
hardcoding, and the whole thing is saved with each run for reproducibility (D8). Kept
deliberately minimal (D9): a knob is added when a stage needs it, not before.

Deliberately NOT here, and why:
  * discount factor (gamma) — blackjack has terminal-only reward and no intermediate steps,
    so the return is just the final payout; gamma would do nothing.
  * state-feature flags — A's state is fixed (total, soft, upcard); the toggle arrives with
    Problem B's count, when encode_state actually grows.
  * algorithm name / env ruleset — these are provenance, recorded in the saved run's
    metadata, not hyperparameters you tune.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable hyperparameters for one training run.

    num_episodes : number of hands (episodes) to train on.
    epsilon      : exploration rate for epsilon-greedy. Fixed (no decay) for now; we train
                   with this but EVALUATE greedily (epsilon = 0), so residual exploration
                   never touches the reported policy. See CONCEPTS.md #15.
    seed         : seed for the global RNG (Phase 2 convention: 42).
    """

    num_episodes: int
    epsilon: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {self.num_episodes}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
