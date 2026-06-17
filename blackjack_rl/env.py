"""Episode-capture wrapper around the Phase 2 engine — the boundary between the blackjack
world and our RL code. See DESIGN.md D7 / A1 / A11.

The policy plays straight through the engine; we read the episode back from the HandResult's
per-decision records — each decision's state key, action, and the return the engine attributes
to it. For a single (no-split) hand every decision carries the hand's total; for a split, each
sub-hand's decisions carry that sub-hand's payout and the split decision carries the net (b).
So credit assignment follows the tree correctly, with no extra bookkeeping here.

Reproducibility: the engine shuffles with Python's global `random`; seed it ONCE before rolling
out (never per hand). See CONCEPTS.md #14.
"""
from dataclasses import dataclass
from typing import Iterator

from simulator.card import Deck
from simulator.config import SimulatorConfig, vegas_strip
from simulator.game_state import Action
from simulator.hand_simulator import HandSimulator
from strategies.base import Strategy

from blackjack_rl.state import StateKey, encode_state


@dataclass
class Episode:
    """One played hand as RL sees it.

    steps  : (state_key, action, return) per decision, in play order. `return` is the MC return
             credited to that decision — the payout of the (sub-)hand it belongs to (the split
             decision gets the net of its sub-hands). For a single-chain hand every step's
             return equals `reward`.
    reward : the hand's total payout in bet units (bet = 1) — used for the house-edge metric.

    `steps` may be empty (a dealt blackjack resolves with no decision).
    """

    steps: list[tuple[StateKey, Action, float]]
    reward: float


def problem_a_config() -> SimulatorConfig:
    """Rules for Problem A: the 6-deck S17 3:2 anchor config, counting OFF. A fresh shoe per
    rollout makes hands independent and counting-free — a clean MDP."""
    cfg = vegas_strip()
    cfg.card_counting_allowed = False
    return cfg


def rollout(
    policy: Strategy, config: SimulatorConfig | None = None, with_splits: bool = False
) -> Episode:
    """Play one hand with `policy`; return the captured Episode.

    The policy plays through the engine directly; we read each decision's (state, action, its
    own return) from the HandResult's per-decision records. `with_splits` must match how the
    policy encodes states (the trainer passes ``config.with_splits``); it only affects key
    encoding, not whether the engine splits. Fresh shoe per call (Problem A); seed the global
    RNG once before repeated calls.
    """
    cfg = config if config is not None else problem_a_config()
    deck = Deck(num_decks=cfg.num_decks)  # fresh shoe, shuffled in __init__
    result = HandSimulator(cfg, deck, policy).play_hand(
        session_id="ep", bankroll=0.0, bet_size=1.0, hands_played=0
    )
    steps: list[tuple[StateKey, Action, float]] = [
        (encode_state(r, with_splits), r.action, r.payout)
        for r in result.decision_records
        if r.action != "none"
    ]
    return Episode(steps=steps, reward=result.payout)


def rollout_many(
    policy: Strategy, n: int, config: SimulatorConfig | None = None, with_splits: bool = False
) -> Iterator[Episode]:
    """Yield `n` Episodes. Seed the global RNG once before iterating for reproducibility."""
    cfg = config if config is not None else problem_a_config()
    for _ in range(n):
        yield rollout(policy, cfg, with_splits)
