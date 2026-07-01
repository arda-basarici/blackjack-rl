"""Training loops & orchestration for Problem B (build stages B2–B4).

- `train_bet`        — the bet model on log-growth, with fixed basic play (B2d).
- `run_factored`     — assemble count-aware play (dqn) + bet model into the factored policy (B3).
- `train_monolithic` — the end-to-end baseline (B4).
Persists runs via core.persistence (record + model), like the A/DQN experiments.
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.schedules import make_schedule
from blackjack_rl.core.util import format_duration
from blackjack_rl.dqn.deep_q import make_target, soft_update, sync_target, td_update
from blackjack_rl.dqn.replay import ReplayBuffer
from blackjack_rl.session.bet_agent import BetAgent, greedy_bet_curve, session_to_transitions
from blackjack_rl.session.env import GROWTH_BANKROLL, SessionConfig, SessionEnv, growth_config


@dataclass(frozen=True)
class BetTrainConfig:
    """Immutable hyperparameters for one bet-model (B2d) training run.

    A dedicated config (like ``DQNConfig`` for the play side) — the play knobs (with_splits, encoding,
    reward_baseline, curriculum) are meaningless here, and the bet model has its own (the regime, the
    ruin penalty, the bankroll reference). It **composes** a ``SessionConfig`` (the regime — growth/ruin
    — defining the MDP: bankroll, horizon, spread, table rule), so it lives in the session layer next to
    the env, not in ``core`` (``core`` is the base layer and must not depend on ``session`` types).

    The agent's discrete menu is ``session.bet_spread`` and its shoe size is read from the env's sim
    config, so the encoding stays consistent with the environment by construction.

    session           : the regime (use ``growth_config`` / ``ruin_config``). NOTE its ``seed`` is unused
                        during training — ``train_bet`` seeds from ``BetTrainConfig.seed`` (it drives the
                        online loop directly, not ``run_sessions``).
    n_sessions        : number of training sessions (the outer-loop length).
    epsilon[_*]       : exploration rate over the bet menu / decaying schedule (reuses schedules.py).
    hidden            : QNetwork hidden-layer sizes.
    lr / lr_schedule / lr_end : Adam step and its (optional) decay (CONCEPTS §26).
    gamma             : TD discount. **Default 0.0 (myopic).** Kelly is the per-hand log-optimum, so in
                        the growth regime (ruin dormant) the bandit objective learns the count->bet ramp
                        at minimum variance; gamma=1 telescopes the ~1000-hand return and DIVERGES (B2d-3
                        finding). The *ruin* regime needs an intermediate gamma in (0,1) — its own value,
                        not yet characterized — to value ruin-avoidance; set it per-run, never 1.0.
    batch_size / buffer_capacity / warmup / updates_per_step / train_every / target_sync_every /
    target_tau / double_dqn : the standard deep-Q stabilizers (reused from dqn.deep_q).
    ruin_penalty      : finite stand-in for a total-wipeout hand's ``-inf`` log-reward (reward shaping).
                        The *primary* ruin signal is structural — the session terminates, forfeiting
                        future growth — so this only floors the rare exact-zero bankroll (D14).
    bankroll_scale    : the FIXED bankroll normalizer for the encoder (NOT per-session W0) — preserves
                        absolute unit scale so a bankroll-generalizing agent reuses the encoding (D14).
    seed              : seeds both RNGs once (random: engine/epsilon/replay; torch: weights).
    torch_threads     : torch CPU threads (1 = bit-reproducible, best for this tiny net).
    """

    session: SessionConfig = field(default_factory=growth_config)
    n_sessions: int = 20_000
    epsilon: float = 0.1
    epsilon_schedule: str = "constant"
    epsilon_start: float = 0.5
    epsilon_end: float = 0.0
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    lr_schedule: str = "constant"
    lr_end: float = 1e-5
    gamma: float = 0.0
    batch_size: int = 128
    buffer_capacity: int = 50_000
    warmup: int = 1_000
    updates_per_step: int = 1
    train_every: int = 4
    target_sync_every: int = 1_000
    target_tau: float = 0.0
    double_dqn: bool = False
    ruin_penalty: float = -1.0
    reward_scale: float = 1.0
    bankroll_scale: float = GROWTH_BANKROLL
    bankroll_feature: str = "raw"  # bet-encoder ablation: "raw" | "logratio" | "none" (drop bankroll)
    seed: int = 0
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.n_sessions < 1:
            raise ValueError(f"n_sessions must be >= 1, got {self.n_sessions}")


# probe counts for the checkpoint bet-vs-count curve (watch the spread form over training)
_PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)


def train_bet(
    config: BetTrainConfig,
    progress_every: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
    on_snapshot: Callable[[int, dict], None] | None = None,
) -> BetAgent:
    """Train a ``BetAgent`` by deep Q-learning over ``config.n_sessions`` sessions of Problem B, with
    **fixed ``BasicStrategy`` play** — isolating the betting lever (rung 3 of the D17 ladder).

    Seeds both RNGs once from ``config.seed`` (``random`` for engine shuffle / epsilon / replay sampling,
    ``torch`` for weights), so a run is reproducible. Online loop: each session is played with the
    *current* epsilon-greedy net via ``SessionEnv.run`` (the single session-generation seam — the parked
    bankroll-sweep extension swaps a per-session W0 sampler in here); its hands are reconstructed into
    bettor transitions and pushed; after warm-up, ``updates_per_step`` gradient steps fire every
    ``train_every`` transitions, with the target net hard-synced every ``target_sync_every`` steps (or
    soft-updated when ``target_tau > 0``). Anneals epsilon by the schedule and emits a learning curve
    (loss + the bet-vs-count probe) via ``on_checkpoint``. ``on_snapshot(session, state_dict)`` — if
    given — fires at the *same* cadence with a CPU clone of the q-net weights, so a caller can persist the
    trajectory's checkpoints (the policy *wanders through* good ramps it doesn't keep — Test 11); kept a
    separate seam so ``on_checkpoint`` stays a JSON-safe learning-curve point. Returns the trained agent.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)

    env = SessionEnv(config.session)
    play = BasicStrategy()
    agent = BetAgent(
        levels=config.session.bet_spread,
        hidden=config.hidden,
        epsilon=config.epsilon,
        num_decks=env.sim_config.num_decks,
        bankroll_scale=config.bankroll_scale,
        bankroll_feature=config.bankroll_feature,
    )
    target = make_target(agent.q_net)
    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    epsilon_at = make_schedule(
        config.epsilon_schedule, constant=config.epsilon, start=config.epsilon_start,
        end=config.epsilon_end, num_episodes=config.n_sessions,
    )
    lr_at = make_schedule(
        config.lr_schedule, constant=config.lr, start=config.lr, end=config.lr_end,
        num_episodes=config.n_sessions,
    )

    total = config.n_sessions
    start = time.perf_counter()
    last_t, last_done = start, 0  # window for the current-rate / eta readout
    env_steps = 0  # transitions collected (the replay-ratio clock)
    grad_steps = 0
    loss_sum, loss_count = 0.0, 0  # accumulated since the last checkpoint
    probe_bankroll = config.session.starting_bankroll
    probe_depth = env.sim_config.num_decks / 2.0  # representative mid-shoe depth

    for i in range(total):
        agent.epsilon = epsilon_at(i)
        for group in optimizer.param_groups:  # anneal the step (no-op for a constant schedule)
            group["lr"] = lr_at(i)
        cap = env.run(play, agent)  # <- single session-generation seam (current eps-greedy net)
        for transition in session_to_transitions(
            cap, encode=agent.encode_state, n_levels=len(agent.levels), ruin_reward=config.ruin_penalty,
            reward_scale=config.reward_scale,
        ):
            buffer.push(transition)
            env_steps += 1
            ready = len(buffer) >= config.warmup and buffer.can_sample(config.batch_size)
            if ready and env_steps % config.train_every == 0:
                for _ in range(config.updates_per_step):
                    loss = td_update(
                        agent.q_net, target, buffer.sample(config.batch_size), optimizer,
                        config.gamma, double=config.double_dqn,
                    )
                    loss_sum += loss
                    loss_count += 1
                    grad_steps += 1
                    if config.target_tau > 0:
                        soft_update(target, agent.q_net, config.target_tau)
                    elif grad_steps % config.target_sync_every == 0:
                        sync_target(target, agent.q_net)

        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            now = time.perf_counter()
            elapsed = now - start
            interval = now - last_t
            rate = (done - last_done) / interval if interval else 0.0
            eta = (total - done) / rate if rate else 0.0
            last_t, last_done = now, done
            avg_loss = loss_sum / loss_count if loss_count else float("nan")
            curve = greedy_bet_curve(
                agent, _PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth
            )
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  elapsed {format_duration(elapsed)}  "
                f"{rate:,.0f} sess/s  eta {format_duration(eta)}  eps {agent.epsilon:.3f}  "
                f"lr {lr_at(i):.2e}  grad_steps {grad_steps:,}  loss {avg_loss:.5f}  "
                f"bet@0 {curve[0]:.0f}  bet@6 {curve[6]:.0f}",
                file=sys.stderr,
            )
            if on_checkpoint is not None:
                on_checkpoint({
                    "session": done,
                    "epsilon": round(agent.epsilon, 4),
                    "lr": round(lr_at(i), 8),
                    "grad_steps": grad_steps,
                    "buffer": len(buffer),
                    "recent_loss": round(avg_loss, 6) if loss_count else None,
                    "bet_by_count": {c: round(w, 3) for c, w in curve.items()},
                })
            if on_snapshot is not None:
                on_snapshot(done, {k: v.detach().cpu().clone() for k, v in agent.q_net.state_dict().items()})
            loss_sum, loss_count = 0.0, 0

    return agent


def run_factored(config):
    raise NotImplementedError("B3: factored play(EV, count-aware) + bet(Kelly) orchestration")


def train_monolithic(config):  # -> MonolithicAgent
    raise NotImplementedError("B4: end-to-end play+bet on log-growth")
