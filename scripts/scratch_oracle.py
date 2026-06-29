"""Oracle-reward diagnostic (B2d-3 scratch): is the bettor *code* able to learn count->bet at all?

Replaces the realized per-hand log-reward with its EXPECTATION from the committed edge-by-count
reference -- deterministic given (count, bet, bankroll):

    oracle(c, b, W) = scale * [ (b/W)*mean_return(c) - 0.5*(b/W)^2 * variance(c) ]   (2nd-order E[log])

whose argmax over the spread is discrete-Kelly (b* = mean_return/variance * W = f* * W). Removing the
per-hand noise removes the entire job of buffer/batch/replay-ratio/coverage (which exist to average
noise), so a flat or wandering result here can ONLY mean a code/representation bug -- not an SNR wall.
Run at gamma=0 (clean one-step regression; target net + sync drop out) and scaled to O(1) so a flat
result cannot be blamed on the reward-scale mismatch either.

PASS signature: a STABLE monotone ramp (<=0 -> 1, +2 -> 2, +4 -> 5, +6 -> 8) that HOLDS (deterministic
reward => no argmax wandering). FAIL: flat or wandering -> hunt the bug before any more training.

    .venv\\Scripts\\python.exe scripts/scratch_oracle.py sanity            # print reward table, no train
    .venv\\Scripts\\python.exe scripts/scratch_oracle.py <n_sessions> <scale>   # train
"""
from __future__ import annotations

import random
import sys
import time
from datetime import datetime
from math import isfinite

import torch

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.core.paths import LOGS_DIR
from blackjack_rl.dqn.deep_q import make_target, td_update
from blackjack_rl.dqn.replay import ReplayBuffer, Transition
from blackjack_rl.session.bet_agent import BetAgent, greedy_bet_curve
from blackjack_rl.session.env import SessionEnv, growth_config
from blackjack_rl.session.references import load_edge_reference

PROBE_COUNTS: tuple[int, ...] = (-4, -2, 0, 2, 4, 6, 8)
SEED = 0
EPSILON = 0.1
GAMMA = 0.0
WARMUP = 1_000
BATCH = 128
TRAIN_EVERY = 4
BUFFER = 50_000


def _log(line: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "live.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def make_oracle_reward(edges, scale: float):
    """Deterministic expected log-reward (2nd-order) from the measured per-count mean/variance."""
    keys = sorted(edges)

    def reward(true_count: float, bet: float, bankroll: float) -> float:
        c = min(keys, key=lambda k: abs(k - round(true_count)))
        e = edges[c]
        f = bet / bankroll
        return scale * (f * e.mean_return - 0.5 * f * f * e.variance)

    return reward


def sanity(edges, kelly_curve, scale: float, bankroll: float = 400.0) -> None:
    """Print, per probe count, the oracle reward across the spread + its argmax vs discrete-Kelly.
    Verifies the TARGET is correct before we train a net against it."""
    spread = tuple(range(1, 9))
    reward = make_oracle_reward(edges, scale)
    _log(f"[oracle-sanity] scale={scale:g} bankroll={bankroll:.0f}  (reward across bets 1..8)")
    for c in PROBE_COUNTS:
        rs = [reward(c, b, bankroll) for b in spread]
        argmax_bet = spread[max(range(len(rs)), key=lambda i: rs[i])]
        kelly_bet = min(spread, key=lambda b: abs(b - kelly_curve.get(c, 0.0) * bankroll))
        row = " ".join(f"{r:+.2f}" for r in rs)
        flag = "OK" if argmax_bet == kelly_bet else "<-- MISMATCH"
        _log(f"  TC{c:+d}: [{row}]  argmax={argmax_bet}  kelly={kelly_bet}  {flag}")


def oracle_transitions(capture, encode, n_levels: int, reward_fn) -> list[Transition]:
    """gamma=0 transitions with the oracle reward (next-state irrelevant; done=True everywhere)."""
    no_legal = torch.zeros(n_levels, dtype=torch.bool)
    out: list[Transition] = []
    for rec in capture.hands:
        if rec.bet_level is None:
            raise ValueError("oracle_transitions requires an IndexedBetPolicy capture")
        state = torch.tensor(
            encode(rec.true_count, rec.decks_remaining, rec.bankroll_before), dtype=torch.float32
        )
        r = reward_fn(rec.true_count, rec.bet, rec.bankroll_before)
        if not isfinite(r):
            raise ValueError(f"non-finite oracle reward {r}")
        out.append(Transition(
            state=state, action=rec.bet_level, reward=r,
            next_state=torch.zeros_like(state), done=True, next_legal_mask=no_legal,
        ))
    return out


def train_oracle(n_sessions: int, scale: float) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    ref = load_edge_reference()
    reward_fn = make_oracle_reward(ref.edges, scale)
    env = SessionEnv(growth_config())
    play = BasicStrategy()
    agent = BetAgent(levels=growth_config().bet_spread, epsilon=EPSILON, num_decks=env.sim_config.num_decks)
    target = make_target(agent.q_net)  # unused at gamma=0, kept for td_update signature
    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=BUFFER)

    probe_bankroll = growth_config().starting_bankroll
    probe_depth = env.sim_config.num_decks / 2.0
    env_steps = 0
    loss_sum, loss_count = 0.0, 0.0
    _log(f"[oracle] START n={n_sessions} scale={scale:g} gamma={GAMMA} eps={EPSILON} seed={SEED}")
    start = time.perf_counter()

    for i in range(n_sessions):
        cap = env.run(play, agent)
        for t in oracle_transitions(cap, agent.encode_state, len(agent.levels), reward_fn):
            buffer.push(t)
            env_steps += 1
            if len(buffer) >= WARMUP and buffer.can_sample(BATCH) and env_steps % TRAIN_EVERY == 0:
                loss = td_update(agent.q_net, target, buffer.sample(BATCH), optimizer, GAMMA, double=False)
                loss_sum += loss
                loss_count += 1

        if (i + 1) % max(1, n_sessions // 8) == 0:
            curve = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth)
            ramp = " ".join(f"{c:+d}:{curve[c]:g}" for c in PROBE_COUNTS)
            avg = loss_sum / loss_count if loss_count else float("nan")
            _log(f"[oracle] {i + 1}/{n_sessions} loss={avg:.4f} | bet[{ramp}]")
            loss_sum, loss_count = 0.0, 0.0

    elapsed = time.perf_counter() - start
    curve = greedy_bet_curve(agent, PROBE_COUNTS, bankroll=probe_bankroll, decks_remaining=probe_depth)
    ramp = " ".join(f"{c:+d}:{curve[c]:g}" for c in PROBE_COUNTS)
    _log(f"[oracle] DONE ({elapsed:.0f}s) final bet[{ramp}]")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        ref = load_edge_reference()
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
        sanity(ref.edges, ref.kelly_curve, scale)
        return
    n_sessions = int(sys.argv[1])
    scale = float(sys.argv[2])
    _log(f"=== oracle diagnostic {datetime.now().strftime('%H:%M:%S')} ===")
    train_oracle(n_sessions, scale)


if __name__ == "__main__":
    main()
