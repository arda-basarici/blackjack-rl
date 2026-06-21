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

import os
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
    lr                : Adam learning rate. With a decaying ``lr_schedule`` this is the *start* rate.
    lr_schedule       : "constant" | "linear" | "exponential" | "harmonic" (reuses schedules.py).
                        A decaying step size lets the estimate converge to a point instead of
                        oscillating in a fixed band under constant gain — the tabular agent's ``1/n``
                        step, ported to the net (CONCEPTS §26). "constant" = the original behavior.
    lr_end            : end learning rate for a decaying schedule (reached at the final episode;
                        must be > 0 for harmonic/exponential). Ignored when lr_schedule="constant".
    gamma             : TD discount (1.0 for Problem A — reward is terminal-only).
    batch_size        : replay minibatch size.
    buffer_capacity   : replay ring-buffer size.
    warmup            : transitions to collect before the first gradient step.
    updates_per_step  : gradient steps per *training event* (when one fires).
    train_every       : fire a training event only every this many decisions (the replay ratio;
                        4 = DeepMind's DQN — fewer, less-redundant updates, much faster than 1).
    target_sync_every : hard-sync the target network every this many gradient steps.
    target_tau        : if > 0, soft/Polyak target update each step instead of the hard sync — the
                        target becomes a slow EMA of the online net, smoothing the bootstrap target
                        *during* training (stabilizes the learning dynamics, not just the readout).
                        0 = hard sync (default). Typical: 0.005-0.01.
    double_dqn        : Double-DQN targets (select next action with the online net, evaluate with
                        the target net) to curb the max-overestimation bias. Off = vanilla DQN.
    encoding          : input encoding — "scalar" (smooth/ordered prior) or "onehot" (total + upcard
                        as categories, sharp where blackjack is sharp). See CONCEPTS §21.
    exploring_starts  : train with forced (state, action) starts (uniform coverage, greedy follow,
                        epsilon ignored) instead of natural epsilon-greedy play. The DQN capstone.
    log_q_grid        : at each checkpoint, log Q for every action at every cell (all 240) so we
                        can plot per-cell Q-trajectories. Off by default (keeps records small).
    swa               : Stochastic Weight Averaging — average the network weights over the back half
                        of training (snapshotting at each checkpoint) and evaluate the averaged net.
                        Cancels the high-variance-action oscillation by averaging it out. Needs
                        progress_every set (it snapshots at checkpoints).
    with_splits       : enable the split action + pair-aware state (A11); off = no-split A.
    with_surrender    : enable the surrender action (terminal, first-action-only, -0.5 payout). Off
                        by default; threads to the rollout/eval config and the diff so all three allow
                        it consistently. Prep for the full action set (Problem B).
    seed              : seed for both RNGs (random for engine/replay, torch for weights).
    num_threads       : torch CPU threads for training matmuls. 1 = single-thread (bit-reproducible,
                        best for tiny nets where dispatch overhead dominates). 0 = all cores. >1
                        speeds up large nets (256+ hidden, big batches) but parallel float-reduction
                        order means runs may not be bit-identical across machines.
    device            : "cpu" (default), "cuda", or "auto" (cuda if available). For this tiny net +
                        sequential single-hand env loop, GPU isn't faster per run; its value is as a
                        parallel compute lane (a GPU run alongside CPU runs frees the CPU).
    double_after      : curriculum knob. Episodes before this train hit/stand only — ``double`` is
                        kept out of both action selection and the bootstrap max, so the low-variance
                        base policy is learned clean and uncontaminated by the oscillating double-Q.
                        At/after it, ``double`` is enabled and learned on the stable base. 0 (default)
                        = no curriculum (double available from the start).
    reward_baseline   : dealer control variate subtracted from the terminal reward —
                        "none" (default) | "bust" (coarse, bust/no-bust) | "stand" (full dealer-total
                        via a stand reference). Mean-zero + action-independent, so EV and policy are
                        unchanged; it strips the dealer's shared variance so high-variance actions
                        settle (CONCEPTS §27; evaluation/dealer_baseline).
    baseline_c        : coefficient for the "bust" baseline (ignored otherwise). 1.0 default.
    """

    num_episodes: int
    epsilon: float = 0.1
    epsilon_schedule: str = "constant"
    epsilon_start: float = 0.3
    epsilon_end: float = 0.0
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    lr_schedule: str = "constant"
    lr_end: float = 1e-5
    lr_hold_until: int = 0
    gamma: float = 1.0
    batch_size: int = 128
    buffer_capacity: int = 50_000
    warmup: int = 1_000
    updates_per_step: int = 1
    train_every: int = 4
    target_sync_every: int = 1_000
    target_tau: float = 0.0
    double_dqn: bool = False
    encoding: str = "scalar"
    exploring_starts: bool = False
    log_q_grid: bool = False
    swa: bool = False
    with_splits: bool = False
    with_surrender: bool = False
    seed: int = 42
    num_threads: int = 1
    device: str = "cpu"
    double_after: int = 0
    reward_baseline: str = "none"
    baseline_c: float = 1.0

    def torch_threads(self) -> int:
        """Resolve ``num_threads`` to a concrete count: 0 means all available cores."""
        return self.num_threads if self.num_threads > 0 else (os.cpu_count() or 1)

    def resolve_device(self) -> str:
        """Resolve ``device``: 'auto' -> cuda if available else cpu; 'cuda' errors if none present."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA GPU is available")
        return self.device

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {self.num_episodes}")
        if self.num_threads < 0:
            raise ValueError(f"num_threads must be >= 0 (0 = all cores), got {self.num_threads}")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {self.device!r}")
        if self.double_after < 0:
            raise ValueError(f"double_after must be >= 0, got {self.double_after}")
        if self.reward_baseline not in ("none", "bust", "stand"):
            raise ValueError(f"reward_baseline must be 'none', 'bust', or 'stand', got {self.reward_baseline!r}")
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
        if self.lr_schedule not in KINDS:
            raise ValueError(f"lr_schedule must be one of {KINDS}, got {self.lr_schedule!r}")
        if self.lr_schedule in ("exponential", "harmonic") and self.lr_end <= 0.0:
            raise ValueError(f"lr_end must be > 0 for a {self.lr_schedule} schedule, got {self.lr_end}")
        if not 0 <= self.lr_hold_until < self.num_episodes:
            raise ValueError(f"lr_hold_until must be in [0, num_episodes), got {self.lr_hold_until}")
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
        if self.encoding not in ("scalar", "onehot", "thermometer"):
            raise ValueError(f"encoding must be 'scalar', 'onehot', or 'thermometer', got {self.encoding!r}")
        if not 0.0 <= self.target_tau < 1.0:
            raise ValueError(f"target_tau must be in [0, 1), got {self.target_tau}")
