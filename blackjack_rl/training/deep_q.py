"""Deep Q-learning trainer (CONCEPTS.md section 17).

This unit ships the second stabilizer — the **target network** — and its helpers. The TD training
loop (capture episodes, reconstruct transitions, sample minibatches, optimize) is the next unit
and will live here too.

Why a target network: a TD target ``y = r + gamma * max_a' Q(s', a')`` computed with the *same*
weights being optimized makes the goal move every gradient step — the network chases its own
estimate and can oscillate or diverge. Computing targets from a *frozen* copy, refreshed only
every C steps, gives the online network a stationary goal to descend toward between refreshes.
"""
from __future__ import annotations

import copy
import random
import sys
import time
from collections.abc import Sequence
from typing import Callable

import torch
from torch.nn import functional as F

from simulator.game_state import Action, GameState

from blackjack_rl.agents.dqn import DQNAgent, QNetwork, encode_features
from blackjack_rl.config import DQNConfig
from blackjack_rl.env import CapturedHand, Step, capture_hand, problem_a_config
from blackjack_rl.evaluation.dealer_baseline import baseline
from blackjack_rl.evaluation.network_diff import diff_network, enumerate_cells
from blackjack_rl.schedules import make_schedule
from blackjack_rl.training.replay import Batch, ReplayBuffer, Transition
from blackjack_rl.util import format_duration


def make_target(online: QNetwork) -> QNetwork:
    """Return a frozen copy of ``online`` for computing TD targets: identical architecture and
    weights, set to eval mode, with gradients disabled (it is never optimized — only periodically
    synced via :func:`sync_target`)."""
    target = copy.deepcopy(online)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def sync_target(target: QNetwork, online: QNetwork) -> None:
    """Hard update — copy the online weights into the target. Called every C gradient steps; in
    between, the target stays fixed so the optimization goal is stationary."""
    target.load_state_dict(online.state_dict())


def soft_update(target: QNetwork, online: QNetwork, tau: float) -> None:
    """Polyak / soft target update — nudge the (frozen) target a fraction ``tau`` toward the online
    net *every* step: ``theta_target = (1-tau)*theta_target + tau*theta_online``. The target becomes
    a slow EMA of the online weights, so the bootstrap target it produces is smoothed *during*
    training (a softer alternative to the periodic hard sync; stabilizes the dynamics at the source,
    not just the final readout). The target stays frozen for gradients."""
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.mul_(1.0 - tau).add_(op, alpha=tau)


# --- transition reconstruction (episode -> TD transitions) -------------------

def _legal_mask(step: Step, actions: Sequence[Action]) -> torch.Tensor:
    """Boolean legal-action mask aligned to ``actions``. Hit/stand are always legal at a recorded
    decision; double iff ``can_double``; split iff ``can_split`` (used only in split mode)."""
    flags = {"hit": True, "stand": True, "double": step.can_double, "split": step.can_split,
             "surrender": False}  # surrender is first-action-only -> never legal as a *next* action
    return torch.tensor([flags[a] for a in actions], dtype=torch.bool)


def hand_to_transitions(
    hand: CapturedHand, actions: Sequence[Action], with_splits: bool = False, encoding: str = "scalar",
    reward_baseline: str = "none", baseline_c: float = 1.0,
) -> list[Transition]:
    """Reconstruct TD transitions from a captured **no-split** hand (a single decision chain).

    Within the chain, a non-final decision gets reward 0, ``done=False``, and ``s'`` = the next
    decision (carrying the next state's legal mask, used to max over *legal* next actions in the
    TD target). The final decision gets reward = the hand payout and ``done=True`` (the next_*
    fields are unused placeholders). ``gamma = 1``, so no discount appears here.

    ``reward_baseline`` ("none"|"bust"|"stand") subtracts a mean-zero, action-independent dealer
    control variate from the *terminal* reward — strips the dealer's shared variance so high-variance
    actions settle, without changing EV or the policy (CONCEPTS §27; see evaluation/dealer_baseline).

    NO-SPLIT ONLY: assumes the steps form one chain. Splitting (a tree of sub-hands) is a later
    extension that must decide how the ``split`` action's two successors form its target.
    """
    action_index = {a: i for i, a in enumerate(actions)}
    n = len(actions)
    transitions: list[Transition] = []
    last = len(hand.steps) - 1
    for i, step in enumerate(hand.steps):
        state = torch.tensor(encode_features(step, with_splits, encoding), dtype=torch.float32)
        if i == last:  # terminal decision: carries the payout, no bootstrap
            terminal_reward = hand.reward
            if step.action != "surrender":  # surrender: fixed -0.5, dealer never plays -> no CV
                terminal_reward -= baseline(
                    reward_baseline, start_total=step.player_value, upcard=step.dealer_upcard,
                    dealer_final=step.final_dealer_value, c=baseline_c,
                )
            transitions.append(
                Transition(
                    state=state,
                    action=action_index[step.action],
                    reward=terminal_reward,
                    next_state=torch.zeros_like(state),
                    done=True,
                    next_legal_mask=torch.zeros(n, dtype=torch.bool),
                )
            )
        else:  # intermediate decision: reward 0, bootstrap from the next state
            nxt = hand.steps[i + 1]
            transitions.append(
                Transition(
                    state=state,
                    action=action_index[step.action],
                    reward=0.0,
                    next_state=torch.tensor(encode_features(nxt, with_splits, encoding), dtype=torch.float32),
                    done=False,
                    next_legal_mask=_legal_mask(nxt, actions),
                )
            )
    return transitions


# --- the TD update (one gradient step of deep Q-learning) --------------------

def td_target(
    target: QNetwork, batch: Batch, gamma: float = 1.0, online: QNetwork | None = None,
    mask_action: int | None = None,
) -> torch.Tensor:
    """The TD target y for each transition: ``r`` for terminal steps, else
    ``r + gamma * Q(s', a*)`` where a* is the best legal next action.

    Two ways to pick and value a*:
    - **vanilla DQN** (``online is None``): the target net both selects and evaluates —
      ``a* = argmax Q_target``, value ``= max Q_target``. Simple, but the ``max`` over noisy
      estimates systematically *overestimates*.
    - **Double DQN** (``online`` given): select with the online net (``a* = argmax Q_online``) but
      read the value from the target net (``Q_target(s', a*)``). Decoupling selection from
      evaluation removes most of that upward bias.

    Illegal next actions are masked to -inf before the argmax so a* is always legal. The bootstrap
    is zeroed on terminal steps via ``torch.where`` (never a ``-inf * 0 = NaN``). No gradient flows
    through the target.
    """
    with torch.no_grad():
        legal = batch.next_legal_masks
        if mask_action is not None:  # curriculum stage one: keep double out of the bootstrap max
            legal = legal.clone()
            legal[:, mask_action] = False
        if online is None:  # vanilla: target net selects and evaluates
            masked = target(batch.next_states).masked_fill(~legal, float("-inf"))
            next_value = masked.max(dim=1).values                            # [B]
        else:  # Double DQN: online selects (legal only), target evaluates
            online_q = online(batch.next_states).masked_fill(~legal, float("-inf"))
            best = online_q.argmax(dim=1, keepdim=True)                      # [B, 1]
            next_value = target(batch.next_states).gather(1, best).squeeze(1)  # [B]
        bootstrap = torch.where(batch.dones, torch.zeros_like(next_value), next_value)
        return batch.rewards + gamma * bootstrap


def td_update(
    online: QNetwork,
    target: QNetwork,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    gamma: float = 1.0,
    double: bool = False,
    mask_action: int | None = None,
) -> float:
    """One gradient step of deep Q-learning on ``batch``: Huber (smooth-L1) loss between the online
    ``Q(s, a)`` of the taken actions and the TD target from the frozen target net. ``double`` uses
    Double-DQN targets; ``mask_action`` (a curriculum aid) excludes that action index from the
    bootstrap max. Returns the loss value (for the learning curve). Gradients flow only through the
    online net."""
    batch = batch.to(next(online.parameters()).device)  # buffer is CPU; move the minibatch to the net
    q_taken = online(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # [B]
    y = td_target(target, batch, gamma, online=online if double else None, mask_action=mask_action)
    loss = F.smooth_l1_loss(q_taken, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# --- instrumentation: probe-cell Q trajectories ------------------------------

# A handful of cells to watch the Q-values of over training (the over-double headline cells plus
# the famous near-tie). Lets us see *how* the wrong policy forms — when greedy flips, whether
# Q(double) spikes late or is inflated from the start.
PROBE_CELLS: tuple[tuple[int, bool, int], ...] = (
    (20, True, 8), (19, True, 6), (16, True, 4), (13, True, 5), (16, False, 10),
)


def _probe_state(value: int, is_soft: bool, upcard: int) -> GameState:
    return GameState(
        player_value=value, player_is_soft=is_soft, player_card_count=2, dealer_upcard=upcard,
        can_hit=True, can_stand=True, can_double=True, can_split=False, can_surrender=False,
    )


def probe_q_values(agent: DQNAgent, cells: tuple[tuple[int, bool, int], ...] = PROBE_CELLS) -> dict:
    """For each probe cell, the net's current Q for every action — a checkpoint snapshot of the
    value trajectory that drives the policy."""
    out: dict[str, dict] = {}
    for value, is_soft, upcard in cells:
        q = agent.q_values(_probe_state(value, is_soft, upcard))
        label = f"{'soft' if is_soft else 'hard'}{value}_v{upcard}"
        out[label] = {a: round(float(q[i]), 3) for i, a in enumerate(agent.actions)}
    return out


def full_q_grid(agent: DQNAgent) -> dict:
    """Q for every action at every one of the 240 canonical cells — a full snapshot, so per-cell
    Q-trajectories can be plotted for *any* disagreement. 240 forward passes; logged at each
    checkpoint only when ``DQNConfig.log_q_grid`` is set (keeps records small otherwise)."""
    out: dict[str, dict] = {}
    for value, is_soft, upcard in enumerate_cells():
        q = agent.q_values(_probe_state(value, is_soft, upcard))
        out[f"{'soft' if is_soft else 'hard'}{value}_v{upcard}"] = {
            a: round(float(q[i]), 3) for i, a in enumerate(agent.actions)
        }
    return out


# --- the training loop (deep Q-learning over Problem A hands) -----------------

def train_dqn(
    config: DQNConfig,
    progress_every: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> DQNAgent:
    """Train a ``DQNAgent`` by deep Q-learning over ``config.num_episodes`` hands on Problem A.

    Seeds both RNGs once — ``random`` (engine shuffle, epsilon, replay sampling) and ``torch``
    (weight init) — so the run is reproducible. Each hand: play it (epsilon-greedy on the current
    net), reconstruct transitions, push them; after a warm-up, do ``updates_per_step`` gradient
    steps per decision, hard-syncing the target every ``target_sync_every`` steps. Anneals epsilon
    by the schedule and emits a learning curve via ``on_checkpoint``. Returns the trained agent.
    No checkpoint/resume yet (A7): a crash loses the run, but a same-seed rerun reproduces it.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    # Threads: 1 (default) avoids multi-thread dispatch overhead on tiny nets and keeps runs
    # bit-reproducible; large nets / big batches benefit from more (config.num_threads, 0 = all).
    torch.set_num_threads(config.torch_threads())

    device = config.resolve_device()
    agent = DQNAgent(
        epsilon=config.epsilon, with_splits=config.with_splits, hidden=config.hidden,
        encoding=config.encoding, with_surrender=config.with_surrender,
    )
    agent.q_net.to(device)
    target = make_target(agent.q_net)
    target.to(device)
    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    env_config = problem_a_config(with_surrender=config.with_surrender)
    epsilon_at = make_schedule(
        config.epsilon_schedule,
        constant=config.epsilon,
        start=config.epsilon_start,
        end=config.epsilon_end,
        num_episodes=config.num_episodes,
    )
    # Learning-rate schedule: constant by default (original behavior). A decaying step lets the
    # estimate converge to a point instead of swinging in a fixed band under constant gain — the
    # tabular agent's 1/n step, ported to the net (CONCEPTS §26). lr is the start, lr_end the end.
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
    env_steps = 0  # decisions collected (the replay-ratio clock)
    grad_steps = 0
    loss_sum, loss_count = 0.0, 0  # accumulated since the last checkpoint
    counts: dict = {}  # (value, soft, upcard, action) -> experience count
    swa_sum, swa_n = None, 0  # Stochastic Weight Averaging accumulator (back half of training)
    double_idx = agent.actions.index("double") if "double" in agent.actions else None

    for i in range(total):
        agent.epsilon = epsilon_at(i)
        for group in optimizer.param_groups:  # anneal the step size (no-op for a constant schedule)
            group["lr"] = lr_at(i)
        # curriculum: hold double out (selection + bootstrap max) until episode double_after
        stage_one = i < config.double_after
        agent.double_enabled = not stage_one
        mask_action = double_idx if stage_one else None
        hand = capture_hand(agent, env_config)
        for s in hand.steps:
            kc = (s.player_value, s.player_is_soft, s.dealer_upcard, s.action)
            counts[kc] = counts.get(kc, 0) + 1
        for transition in hand_to_transitions(
            hand, agent.actions, config.with_splits, config.encoding,
            reward_baseline=config.reward_baseline, baseline_c=config.baseline_c,
        ):
            buffer.push(transition)
            env_steps += 1
            ready = len(buffer) >= config.warmup and buffer.can_sample(config.batch_size)
            if ready and env_steps % config.train_every == 0:
                for _ in range(config.updates_per_step):
                    loss = td_update(
                        agent.q_net, target, buffer.sample(config.batch_size), optimizer,
                        config.gamma, double=config.double_dqn, mask_action=mask_action,
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
            # cheap, deterministic policy-quality snapshot (240 cells) — measures convergence (A10)
            agreement = diff_network(agent).agreement_unweighted
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  elapsed {format_duration(elapsed)}  "
                f"{rate:,.0f} hands/s  eta {format_duration(eta)}  eps {agent.epsilon:.3f}  "
                f"lr {lr_at(i):.2e}  grad_steps {grad_steps:,}  loss {avg_loss:.4f}  agree {agreement:.1%}",
                file=sys.stderr,
            )
            if on_checkpoint is not None:
                cp = {
                    "episode": done,
                    "epsilon": round(agent.epsilon, 4),
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
