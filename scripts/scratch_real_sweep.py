"""Real-reward lever sweep (B2d-3 scratch, EXPLORATORY tier): does any lever turn the flat policy into
a count->bet ramp on the *real noisy* log-reward? Oracle already proved the code is sound, so this
probes the noise/coverage/scale wall directly.

One flexible runner; launch many in parallel (single-thread each, free on a 22-core box). Real per-hand
log-reward x reward_scale (additive to the argmax; tests optimization-scale), full gamma support
(target net + sync for gamma>0). Streams tagged checkpoints to logs/live.log; saves the final
bet-vs-count curve as JSON. THROWAWAY diagnostic, not a committed deliverable.

    python scripts/scratch_real_sweep.py <tag> <n_sessions> <gamma> <scale> <batch> <eps> <lr>
"""
from __future__ import annotations

import random
import subprocess
import sys
import time
from math import isfinite
from pathlib import Path

import torch

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.dqn.deep_q import make_target, sync_target, td_update
from blackjack_rl.dqn.replay import ReplayBuffer, Transition
from blackjack_rl.session.bet_agent import BetAgent, greedy_bet_curve
from blackjack_rl.session.env import SessionEnv, growth_config

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
SEED = 0
WARMUP = 1_000
TRAIN_EVERY = 4
TARGET_SYNC = 1_000
BUFFER = 50_000
RUIN_REWARD = -1.0
RESULTS_DIR = Path(__file__).resolve().parent.parent / "logs" / "real_sweep"


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def real_transitions(capture, encode, n_levels: int, scale: float) -> list[Transition]:
    all_legal = torch.ones(n_levels, dtype=torch.bool)
    no_legal = torch.zeros(n_levels, dtype=torch.bool)
    out: list[Transition] = []
    hands = capture.hands
    for i, rec in enumerate(hands):
        if rec.bet_level is None:
            raise ValueError("requires an IndexedBetPolicy capture")
        state = torch.tensor(
            encode(rec.true_count, rec.decks_remaining, rec.bankroll_before), dtype=torch.float32
        )
        r = (rec.log_reward if isfinite(rec.log_reward) else RUIN_REWARD) * scale
        if rec.done:
            out.append(Transition(
                state=state, action=rec.bet_level, reward=r,
                next_state=torch.zeros_like(state), done=True, next_legal_mask=no_legal,
            ))
        else:
            nxt = hands[i + 1]
            next_state = torch.tensor(
                encode(nxt.true_count, nxt.decks_remaining, nxt.bankroll_before), dtype=torch.float32
            )
            out.append(Transition(
                state=state, action=rec.bet_level, reward=r,
                next_state=next_state, done=False, next_legal_mask=all_legal,
            ))
    return out


def run(tag: str, n_sessions: int, gamma: float, scale: float, batch: int, eps: float, lr: float,
        seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    env = SessionEnv(growth_config())
    play = BasicStrategy()
    agent = BetAgent(levels=growth_config().bet_spread, epsilon=eps, num_decks=env.sim_config.num_decks)
    target = make_target(agent.q_net)
    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=BUFFER)

    probe_bankroll = growth_config().starting_bankroll
    probe_depth = env.sim_config.num_decks / 2.0
    env_steps, grad_steps = 0, 0
    loss_sum, loss_count = 0.0, 0.0
    _log(f"[{tag}] START n={n_sessions} gamma={gamma:g} scale={scale:g} batch={batch} eps={eps:g} lr={lr:g} seed={seed}")
    start = time.perf_counter()

    for i in range(n_sessions):
        cap = env.run(play, agent)
        for t in real_transitions(cap, agent.encode_state, len(agent.levels), scale):
            buffer.push(t)
            env_steps += 1
            if len(buffer) >= WARMUP and buffer.can_sample(batch) and env_steps % TRAIN_EVERY == 0:
                loss = td_update(agent.q_net, target, buffer.sample(batch), optimizer, gamma, double=False)
                loss_sum += loss
                loss_count += 1
                grad_steps += 1
                if grad_steps % TARGET_SYNC == 0:
                    sync_target(target, agent.q_net)

        if (i + 1) % max(1, n_sessions // 8) == 0:
            curve = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth)
            ramp = " ".join(f"{c:+d}:{curve[c]:g}" for c in PROBE_COUNTS)
            avg = loss_sum / loss_count if loss_count else float("nan")
            _log(f"[{tag}] {i + 1}/{n_sessions} loss={avg:.4f} | bet[{ramp}]")
            loss_sum, loss_count = 0.0, 0.0

    elapsed = time.perf_counter() - start
    curve = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "n_sessions": n_sessions, "gamma": gamma, "scale": scale, "batch": batch,
        "eps": eps, "lr": lr, "seed": seed, "elapsed_s": round(elapsed, 1),
        "curve": {str(c): curve[c] for c in PROBE_COUNTS},
    }, indent=2))
    # persist the trained agent: weights + construction config (to rebuild the shell) + provenance.
    # ~20 KB; lets every downstream wiring (eval / figures / B3) reuse it without retraining.
    torch.save({
        "state_dict": agent.q_net.state_dict(),
        "construction": {
            "levels": list(agent.levels), "hidden": [64, 64], "in_dim": 3,
            "num_decks": agent.num_decks, "bankroll_scale": agent.bankroll_scale,
        },
        "train_config": {
            "n_sessions": n_sessions, "gamma": gamma, "scale": scale, "batch": batch,
            "eps": eps, "lr": lr, "seed": seed, "regime": "growth",
        },
        "git_hash": _git_hash(),
        "curve": {str(c): curve[c] for c in PROBE_COUNTS},
    }, RESULTS_DIR / f"{tag}.pt")
    ramp = " ".join(f"{c:+d}:{curve[c]:g}" for c in PROBE_COUNTS)
    _log(f"[{tag}] DONE ({elapsed:.0f}s) final bet[{ramp}]  saved {tag}.pt")


def main() -> None:
    tag = sys.argv[1]
    n_sessions, gamma, scale = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    batch, eps, lr = int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else SEED
    run(tag, n_sessions, gamma, scale, batch, eps, lr, seed)


if __name__ == "__main__":
    main()
