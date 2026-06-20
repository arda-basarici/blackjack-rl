"""Exploring-starts for the DQN — the forced-coverage capstone for the network agent.

Tests the coverage diagnosis from the natural-play DQN: the residual errors concentrate in *rare*
cells, worst for the rare, high-variance, terminal ``double`` action — because a function
approximator fills under-sampled (state, action) values with *confident extrapolation* (where the
table left them honestly blank). Exploring starts forces every 2-card (state, action) start
uniformly, so each — including ``(soft 20, double)`` — is trained on real returns. **Prediction:**
the inflated double Q-values fall to truth and the over/under-doubling clears.

Engine untouched: reuses ``PreparedDeck`` / ``ForcedFirstAction`` / ``enumerate_start_pairs`` /
``start_cards_for`` from ``training/exploring_starts.py`` (agent-agnostic), and the DQN machinery
(replay, target net, TD update, transition reconstruction, network diff) from ``training/deep_q.py``.
Mirrors ``train_dqn`` but replaces natural epsilon-greedy rollout with uniform forced starts +
greedy follow-on (no epsilon).
"""
from __future__ import annotations

import random
import sys
import time
from typing import Callable

import torch

from simulator.config import SimulatorConfig
from simulator.game_state import Action
from simulator.hand_simulator import HandSimulator

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.env import CapturedHand, Step, problem_a_config
from blackjack_rl.evaluation.network_diff import diff_network
from blackjack_rl.schedules import make_schedule
from blackjack_rl.training.deep_q import (
    full_q_grid,
    hand_to_transitions,
    make_target,
    probe_q_values,
    soft_update,
    sync_target,
    td_update,
)
from blackjack_rl.training.exploring_starts import (
    ForcedFirstAction,
    PreparedDeck,
    StartSpec,
    enumerate_start_pairs,
    start_cards_for,
)
from blackjack_rl.training.replay import ReplayBuffer
from blackjack_rl.util import format_duration


def es_capture(
    agent: DQNAgent, spec: StartSpec, action: Action, cfg: SimulatorConfig
) -> CapturedHand | None:
    """Play one hand from the forced (state, action) seed, greedy thereafter, captured as a
    ``CapturedHand`` for TD reconstruction. ``None`` if the dealer had a natural blackjack (no
    decision taken — correctly discarded, matching A12). Parallels ``env.capture_hand`` but uses a
    forced-prefix deck and a forced first action."""
    pv, soft, can_split, up = spec
    forced = start_cards_for(pv, soft, can_split, up)
    if forced is None:
        raise ValueError(f"start spec not constructible: {spec}")
    deck = PreparedDeck(forced, num_decks=cfg.num_decks)
    policy = ForcedFirstAction(agent, action)
    result = HandSimulator(cfg, deck, policy).play_hand("es", 0.0, 1.0, 0)
    steps = [
        Step(
            player_value=r.player_value,
            player_is_soft=r.player_is_soft,
            dealer_upcard=r.dealer_upcard,
            can_split=r.can_split,
            can_double=r.can_double,
            action=r.action,
        )
        for r in result.decision_records
        if r.action != "none"
    ]
    if not steps:
        return None
    return CapturedHand(steps=steps, reward=result.payout)


def train_dqn_es(
    config: DQNConfig,
    progress_every: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> DQNAgent:
    """Deep Q-learning with **exploring starts**: every episode begins from a uniformly-chosen
    2-card (state, action) pair, then follows the greedy policy (epsilon is ignored — the forced
    start *is* the exploration, the Sutton & Barto variant). Same replay / target-network /
    optimizer / Double-DQN machinery as ``train_dqn``. Both RNGs seeded once for reproducibility."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads())

    device = config.resolve_device()
    agent = DQNAgent(
        epsilon=0.0, with_splits=config.with_splits, hidden=config.hidden, encoding=config.encoding
    )
    agent.q_net.to(device)
    target = make_target(agent.q_net)
    target.to(device)
    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    env_config = problem_a_config()
    start_pairs = enumerate_start_pairs(config.with_splits)
    if not start_pairs:
        raise RuntimeError("no start pairs enumerated")
    # Learning-rate schedule (constant by default): a decaying step lets the estimate converge to a
    # point rather than oscillate under constant gain — the tabular 1/n step ported to the net (§26).
    lr_at = make_schedule(
        config.lr_schedule,
        constant=config.lr,
        start=config.lr,
        end=config.lr_end,
        num_episodes=config.num_episodes,
    )

    total = config.num_episodes
    start = time.perf_counter()
    last_t, last_done = start, 0  # window start for the interval (current) speed + eta
    env_steps = grad_steps = 0
    loss_sum, loss_count = 0.0, 0
    counts: dict = {}  # (value, soft, upcard, action) -> experience count
    swa_sum, swa_n = None, 0  # Stochastic Weight Averaging accumulator (back half of training)

    for i in range(total):
        for group in optimizer.param_groups:  # anneal the step size (no-op for a constant schedule)
            group["lr"] = lr_at(i)
        spec, action = random.choice(start_pairs)
        hand = es_capture(agent, spec, action, env_config)
        if hand is not None:
            for s in hand.steps:
                kc = (s.player_value, s.player_is_soft, s.dealer_upcard, s.action)
                counts[kc] = counts.get(kc, 0) + 1
            for transition in hand_to_transitions(
                hand, agent.actions, config.with_splits, config.encoding
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
            if config.swa and done >= total // 2:  # accumulate weights over the back half
                sd = agent.q_net.state_dict()
                if swa_sum is None:
                    swa_sum = {k: v.detach().clone() for k, v in sd.items()}
                else:
                    for k, v in sd.items():
                        swa_sum[k] += v.detach()
                swa_n += 1
            now = time.perf_counter()
            elapsed = now - start
            interval = now - last_t
            rate = (done - last_done) / interval if interval else 0.0  # current speed, this window
            eta = (total - done) / rate if rate else 0.0
            last_t, last_done = now, done
            avg_loss = loss_sum / loss_count if loss_count else float("nan")
            agreement = diff_network(agent).agreement_unweighted
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  elapsed {format_duration(elapsed)}  "
                f"{rate:,.0f} hands/s  eta {format_duration(eta)}  lr {lr_at(i):.2e}  "
                f"grad_steps {grad_steps:,}  loss {avg_loss:.4f}  agree {agreement:.1%}",
                file=sys.stderr,
            )
            if on_checkpoint is not None:
                cp = {
                    "episode": done,
                    "epsilon": 0.0,
                    "lr": round(lr_at(i), 8),
                    "grad_steps": grad_steps,
                    "buffer": len(buffer),
                    "recent_loss": round(avg_loss, 5) if loss_count else None,
                    "agreement": round(agreement, 4),
                    "probe_q": probe_q_values(agent),
                }
                if config.log_q_grid:
                    cp["q_grid"] = full_q_grid(agent)
                on_checkpoint(cp)
            loss_sum, loss_count = 0.0, 0
    if config.swa and swa_n:  # evaluate the time-averaged weights, not the final snapshot
        agent.q_net.load_state_dict({k: v / swa_n for k, v in swa_sum.items()})
    agent.sample_counts = counts
    return agent
