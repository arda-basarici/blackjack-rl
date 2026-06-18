"""ExperimentConfig — the knobs for a single training run.

Holds only the hyperparameters you actually choose, so the trainer reads from here instead of
hardcoding, and the whole thing is saved with each run for reproducibility (D8). Kept
deliberately minimal (D9): a knob is added when a stage needs it, not before.

Exploration can be constant (``epsilon``) or a decaying schedule (``epsilon_schedule`` with
``epsilon_start`` -> ``epsilon_end``). The value-update step is a sample average by default;
``step_size`` switches it to a constant-alpha (recency-weighted) update, needed when the
target is non-stationary (decaying epsilon). ``with_splits`` turns on the split action and the
pair-aware state encoding (A11); it defaults off so prior no-split runs stay reproducible.

Deliberately NOT here: discount gamma (terminal-only reward); count features (arrive with
Problem B); algorithm/ruleset (provenance, recorded in the run, not tuned).
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
    step_size        : constant-alpha update if set; None = sample average (1/N).
    with_splits      : enable the split action + pair-aware state (A11); off = no-split A.
    seed             : seed for the global RNG (Phase 2 convention: 42).

    We train with exploration but EVALUATE greedily, so exploration only shapes what gets
    sampled, never the reported policy. See CONCEPTS.md #15.
    """

    num_episodes: int
    epsilon: float = 0.1
    epsilon_schedule: str = "constant"
    epsilon_start: float = 0.3
    epsilon_end: float = 0.0
    step_size: float | None = None
    with_splits: bool = False
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
        if self.step_size is not None and not 0.0 < self.step_size <= 1.0:
            raise ValueError(f"step_size must be in (0, 1], got {self.step_size}")


@dataclass(frozen=True)
class DQNConfig:
    """Immutable hyperparameters for one deep-Q-network run (Stage 5).

    A dedicated config rather than overloading ``ExperimentConfig``: the tabular ``step_size`` is
    meaningless for a network, and ``lr`` / replay / target knobs are meaningless for the table, so
    keeping them apart respects one-job-per-config. The exploration + bookkeeping knobs mirror
    ``ExperimentConfig`` (train with exploration, evaluate greedily). ``gamma`` lives here — unlike
    the tabular config — because the TD target uses it, though for terminal-only Problem A it is 1.0.

    num_episodes      : number of hands to train on.
    epsilon[_*]       : exploration rate / decaying schedule (reuses schedules.py).
    hidden            : QNetwork hidden-layer sizes.
    lr                : Adam learning rate.
    gamma             : TD discount (1.0 for Problem A — reward is terminal-only).
    batch_size        : replay minibatch size.
    buffer_capacity   : replay ring-buffer size.
    warmup            : transitions to collect before the first gradient step.
    updates_per_step  : gradient steps per *training event* (when one fires).
    train_every       : fire a training event only every this many decisions (the replay ratio;
                        4 = DeepMind's DQN — fewer, less-redundant updates, much faster than 1).
    target_sync_every : hard-sync the target network every this many gradient steps.
    double_dqn        : Double-DQN targets (select next action with the online net, evaluate with
                        the target net) to curb the max-overestimation bias. Off = vanilla DQN.
    encoding          : input encoding — "scalar" (smooth/ordered prior) or "onehot" (total + upcard
                        as categories, sharp where blackjack is sharp). See CONCEPTS §21.
    with_splits       : enable the split action + pair-aware state (A11); off = no-split A.
    seed              : seed for both RNGs (random for engine/replay, torch for weights).
    """

    num_episodes: int
    epsilon: float = 0.1
    epsilon_schedule: str = "constant"
    epsilon_start: float = 0.3
    epsilon_end: float = 0.0
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 1.0
    batch_size: int = 128
    buffer_capacity: int = 50_000
    warmup: int = 1_000
    updates_per_step: int = 1
    train_every: int = 4
    target_sync_every: int = 1_000
    double_dqn: bool = False
    encoding: str = "scalar"
    with_splits: bool = False
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
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        for name, value in (
            ("batch_size", self.batch_size),
            ("buffer_capacity", self.buffer_capacity),
            ("warmup", self.warmup),
            ("updates_per_step", self.updates_per_step),
            ("train_every", self.train_every),
            ("target_sync_every", self.target_sync_every),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.batch_size > self.buffer_capacity:
            raise ValueError("batch_size cannot exceed buffer_capacity")
        if not self.hidden:
            raise ValueError("hidden must have at least one layer")
        if self.encoding not in ("scalar", "onehot"):
            raise ValueError(f"encoding must be 'scalar' or 'onehot', got {self.encoding!r}")
