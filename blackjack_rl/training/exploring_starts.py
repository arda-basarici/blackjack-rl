"""Monte-Carlo Exploring Starts — forced-coverage capstone for Problem A.

The investigation's thesis is that the residual gap to basic strategy is *coverage*: rare
(state, action) pairs are rarely experienced, so their values stay wrong. Exploring starts breaks
the link between how often a state is *reached* and how often it is *learned* — every episode begins
from a deliberately chosen (state, action) pair, then follows the greedy policy. It is the canonical
MC-control variant (Sutton & Barto, with blackjack as the worked example) and needs no epsilon.

This module is self-contained scaffolding for that experiment. It does NOT modify the engine: it
*subclasses* its ``Deck`` and *wraps* a ``Strategy``, then hands both to the unmodified
``HandSimulator``. Existing ``blackjack_rl`` code (agent, MC update, encode, persistence) is reused
by import, never edited. See ARCHITECTURE.md (A-number for ES) and DESIGN.md section 8.
"""
from __future__ import annotations

import random
import sys
import time
from typing import Callable

from simulator.card import Card, Deck, Rank, Suit
from simulator.game_state import Action, GameState
from simulator.hand_simulator import HandSimulator

from strategies.base import Strategy

from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.config import ExperimentConfig
from blackjack_rl.env import Episode, problem_a_config
from blackjack_rl.state import StateKey, encode_state
from blackjack_rl.training.monte_carlo import (
    _apply_episode,
    _greedy_table,
    _min_state_visits,
)
from blackjack_rl.util import format_duration

# Deal order the engine uses (HandSimulator.play_hand): player-1, dealer-up, player-2, dealer-hole.
# PreparedDeck serves a forced prefix in exactly this order, then the random shoe takes over.

# A start spec is (player_value, is_soft, can_split, dealer_upcard) — the four fields needed to
# construct a hand. Note this is NOT the StateKey order; the engine's encoded key is
# (player_value, is_soft, dealer_upcard[, can_split]).
StartSpec = tuple[int, bool, bool, int]


def card_of_value(value: int) -> Card:
    """A concrete card with the given blackjack value (2-10, or 11 for an ace)."""
    if value == 11:
        rank = Rank.ACE
    elif value == 10:
        rank = Rank.TEN
    elif 2 <= value <= 9:
        rank = Rank(value)
    else:
        raise ValueError(f"no single card has blackjack value {value}")
    return Card(rank, Suit.HEARTS)


def player_cards_for(player_value: int, is_soft: bool, can_split: bool) -> list[Card] | None:
    """The two player cards that realise a target decision state, or ``None`` if it cannot be made
    from exactly two cards (hard 4/20 and soft 12 have no two-distinct-card realisation)."""
    if can_split:
        if is_soft:                       # the only soft pair is A,A (value 12)
            return [card_of_value(11), card_of_value(11)]
        if player_value % 2 == 0 and 2 <= player_value // 2 <= 10:
            half = player_value // 2
            return [card_of_value(half), card_of_value(half)]
        return None
    if is_soft:
        other = player_value - 11         # ace + other
        if 2 <= other <= 9:               # soft 13..20 (soft 12 is A,A, a pair)
            return [card_of_value(11), card_of_value(other)]
        return None
    for a in range(2, 11):                # hard, non-pair: distinct non-ace values
        b = player_value - a
        if 2 <= b <= 10 and b != a:
            return [card_of_value(a), card_of_value(b)]
    return None


def start_cards_for(
    player_value: int, is_soft: bool, can_split: bool, dealer_upcard: int
) -> list[Card] | None:
    """Forced prefix [player-1, dealer-upcard, player-2], or ``None`` if not 2-card constructible."""
    players = player_cards_for(player_value, is_soft, can_split)
    if players is None:
        return None
    return [players[0], card_of_value(dealer_upcard), players[1]]


class PreparedDeck(Deck):
    """A ``Deck`` that deals ``forced`` cards first (in order), then a normal shuffled shoe.

    Constructs a chosen start state without touching the engine. The hole card (4th deal) is NOT
    forced, so dealer dynamics remain the engine's own. (Counting is OFF in Problem A.)
    """

    def __init__(self, forced: list[Card], num_decks: int = 6) -> None:
        super().__init__(num_decks=num_decks)
        self._forced: list[Card] = list(forced)

    def deal(self) -> Card:
        if self._forced:
            card = self._forced.pop(0)
            self._dealt_count += 1
            return card
        return super().deal()


class ForcedFirstAction(Strategy):
    """Returns ``action`` on the first decision, then delegates to ``policy`` (greedy follow-on)."""

    def __init__(self, policy: Strategy, action: Action) -> None:
        self._policy = policy
        self._action: Action = action
        self._used = False

    def decide(self, state: GameState) -> Action:
        if not self._used:
            self._used = True
            return self._action
        return self._policy.decide(state)

    def name(self) -> str:
        return "ForcedFirstAction"


# --- enumeration of start (state, action) pairs --------------------------------------------------

def _candidate_states(with_splits: bool) -> list[StartSpec]:
    """The 2-card decision states to force-start from, for every dealer upcard (2..11)."""
    specs: list[StartSpec] = []
    for up in range(2, 12):
        for pv in range(5, 20):
            specs.append((pv, False, False, up))           # hard non-pair
        for pv in range(13, 21):
            specs.append((pv, True, False, up))            # soft non-pair
        if with_splits:
            for half in range(2, 11):
                specs.append((2 * half, False, True, up))  # 2,2 .. 10,10
            specs.append((12, True, True, up))             # A,A
    return specs


def enumerate_start_pairs(with_splits: bool) -> list[tuple[StartSpec, Action]]:
    """Every legal (start state, first action) pair. Surrender is never forced; split only on pairs
    when ``with_splits``. Only 2-card-constructible states are kept."""
    actions: tuple[Action, ...] = ("hit", "stand", "double")
    pairs: list[tuple[StartSpec, Action]] = []
    for spec in _candidate_states(with_splits):
        pv, soft, can_split, up = spec
        if start_cards_for(pv, soft, can_split, up) is None:
            continue
        for a in actions:
            pairs.append((spec, a))
        if with_splits and can_split:
            pairs.append((spec, "split"))
    return pairs


# --- exploring-starts rollout + training ---------------------------------------------------------

def es_rollout(agent: TabularAgent, spec: StartSpec, action: Action, env_config) -> Episode | None:
    """Play one episode from the forced (state, action) seed, greedy thereafter. ``None`` if the
    dealer had a natural blackjack (no decision taken — correctly discarded)."""
    pv, soft, can_split, up = spec
    forced = start_cards_for(pv, soft, can_split, up)
    if forced is None:
        raise ValueError(f"start spec not constructible: {spec}")
    deck = PreparedDeck(forced, num_decks=env_config.num_decks)
    policy = ForcedFirstAction(agent, action)
    result = HandSimulator(env_config, deck, policy).play_hand("es", 0.0, 1.0, 0)
    steps: list[tuple[StateKey, Action, float]] = [
        (encode_state(r, agent.with_splits), r.action, r.payout)
        for r in result.decision_records
        if r.action != "none"
    ]
    if not steps:
        return None
    return Episode(steps=steps, reward=result.payout)


def train_exploring_starts(
    config: ExperimentConfig,
    progress_every: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> TabularAgent:
    """Monte-Carlo control with exploring starts (no epsilon). Uses ``config.step_size`` (constant
    alpha); ``config.epsilon*`` are ignored. Reproducible: the global RNG is seeded once."""
    random.seed(config.seed)
    agent = TabularAgent(epsilon=0.0, step_size=config.step_size, with_splits=config.with_splits)
    env_config = problem_a_config()
    start_pairs = enumerate_start_pairs(config.with_splits)
    if not start_pairs:
        raise RuntimeError("no start pairs enumerated")

    total = config.num_episodes
    start = time.perf_counter()
    prev_greedy: dict[StateKey, Action] = {}

    for i in range(total):
        spec, action = random.choice(start_pairs)
        episode = es_rollout(agent, spec, action, env_config)
        if episode is not None:
            _apply_episode(agent, episode)
        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0.0
            eta = (total - done) / rate if rate else 0.0
            greedy = _greedy_table(agent)
            print(
                f"  {done:,}/{total:,} ({done / total:.0%})  "
                f"elapsed {format_duration(elapsed)}  {rate:,.0f} hands/s  "
                f"eta {format_duration(eta)}  states {len(greedy)}",
                file=sys.stderr,
            )
            if on_checkpoint is not None:
                churn = sum(1 for s, a in greedy.items() if prev_greedy.get(s) != a)
                on_checkpoint({
                    "episode": done, "epsilon": 0.0, "policy_churn": churn,
                    "min_state_visits": _min_state_visits(agent), "states": len(greedy),
                })
                prev_greedy = greedy
    return agent


# --- run orchestration + CLI ---------------------------------------------------------------------
# Mirrors experiment.run_experiment (train -> evaluate agent + basic -> diff -> save), but trains
# with exploring starts and stamps the record with method="exploring_starts" so it is distinct in
# the ledger. Reuses the evaluate/diff/save/qtable helpers by import; nothing existing is modified.

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from strategies.basic_strategy import BasicStrategy

from blackjack_rl.evaluation.metrics import GreedyPolicy, evaluate_policy
from blackjack_rl.evaluation.policy_diff import diff_policy
from blackjack_rl.experiment import DEFAULT_RUNS_DIR, RunResult, _qtable_records
from blackjack_rl.persistence import save_run


def run_exploring_starts(
    config: ExperimentConfig,
    eval_hands: int = 200_000,
    eval_seed: int = 0,
    min_visits: int = 1000,
    ev_tol: float = 0.02,
    runs_dir: Path | None = None,
    progress_every: int | None = None,
    verbose: bool = False,
) -> RunResult:
    """Train with exploring starts, evaluate (agent + basic), diff, and persist one run."""

    def log(message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    update = "sample-avg" if config.step_size is None else f"alpha={config.step_size}"
    log(
        f"[{started:%Y-%m-%d %H:%M:%S}] exploring-starts training "
        f"{config.num_episodes:,} episodes (seed {config.seed}, {update}, "
        f"splits={config.with_splits}) ..."
    )
    learning_curve: list[dict] = []
    agent = train_exploring_starts(
        config, progress_every=progress_every, on_checkpoint=learning_curve.append
    )
    train_seconds = time.perf_counter() - t0
    log(f"  training done in {format_duration(train_seconds)}")

    eval_start = time.perf_counter()
    log(f"evaluating agent over {eval_hands:,} hands ...")
    agent_edge = evaluate_policy(GreedyPolicy(agent), n_hands=eval_hands, seed=eval_seed)
    log(f"evaluating basic strategy over {eval_hands:,} hands ...")
    basic_edge = evaluate_policy(BasicStrategy(), n_hands=eval_hands, seed=eval_seed)
    log("diffing learned policy vs basic strategy ...")
    report = diff_policy(agent, min_visits=min_visits, ev_tol=ev_tol)
    eval_seconds = time.perf_counter() - eval_start

    finished = datetime.now().astimezone()
    total_seconds = time.perf_counter() - t0
    log(
        f"[{finished:%Y-%m-%d %H:%M:%S}] eval + diff done in "
        f"{format_duration(eval_seconds)} (total {format_duration(total_seconds)})"
    )

    record = {
        "method": "exploring_starts",
        "config": asdict(config),
        "eval": {"hands": eval_hands, "seed": eval_seed},
        "timing": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(eval_seconds, 1),
            "total_seconds": round(total_seconds, 1),
        },
        "metrics": {"agent": asdict(agent_edge), "basic": asdict(basic_edge)},
        "diff": {
            "min_visits": min_visits,
            "ev_tol": ev_tol,
            "agreement_unweighted": report.agreement_unweighted,
            "agreement_weighted": report.agreement_weighted,
            "category_counts": report.category_counts,
            "cells": [asdict(cell) for cell in report.cells],
        },
        "qtable": _qtable_records(agent),
        "learning_curve": learning_curve,
    }
    target = runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR
    run_dir = save_run(target, record)
    log(f"saved run to {run_dir}")
    return RunResult(run_dir=run_dir, agent_edge=agent_edge, basic_edge=basic_edge, diff=report)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Monte-Carlo Exploring Starts capstone run (Problem A)")
    p.add_argument("--episodes", type=int, default=5_000_000)
    p.add_argument("--step-size", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-splits", action="store_true", help="pair-aware state + split action")
    p.add_argument("--eval-hands", type=int, default=200_000)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--min-visits", type=int, default=1000)
    p.add_argument("--ev-tol", type=float, default=0.02)
    p.add_argument("--progress-every", type=int, default=None)
    p.add_argument("--runs-dir", type=Path, default=None)
    args = p.parse_args(argv)

    config = ExperimentConfig(
        num_episodes=args.episodes,
        epsilon=0.0,                      # exploring starts: greedy follow-on, no epsilon
        step_size=args.step_size,
        with_splits=args.with_splits,
        seed=args.seed,
    )
    result = run_exploring_starts(
        config,
        eval_hands=args.eval_hands,
        eval_seed=args.eval_seed,
        min_visits=args.min_visits,
        ev_tol=args.ev_tol,
        runs_dir=args.runs_dir,
        progress_every=args.progress_every,
        verbose=True,
    )
    print(f"saved:   {result.run_dir}")
    print(f"agent:   {result.agent_edge.edge * 100:.3f}% +/- {result.agent_edge.std_error * 100:.3f}")
    print(f"basic:   {result.basic_edge.edge * 100:.3f}% +/- {result.basic_edge.std_error * 100:.3f}")
    print(f"agree:   {result.diff.agreement_weighted:.3f} weighted, "
          f"{result.diff.agreement_unweighted:.3f} unweighted")


if __name__ == "__main__":
    main()
