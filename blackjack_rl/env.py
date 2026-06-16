"""Episode-capture wrapper around the Phase 2 engine — the boundary between the blackjack
world and our RL code. See DESIGN.md D7.

The engine is "engine-calls-strategy" and `HandSimulator.play_hand()` is atomic. To capture
an episode we wrap the acting policy in a recorder that logs (state_key, action) for every
decision as the engine queries it, play one hand, and read the terminal reward off the
`HandResult`. Nothing downstream sees the engine's record format; it stops here.

Reproducibility: the engine shuffles with Python's global `random`, so a run is made
reproducible by seeding it ONCE (`random.seed(...)`) before rolling out — never per hand,
which would make every hand identical. See CONCEPTS.md #14.
"""
from dataclasses import dataclass
from typing import Iterator

from simulator.card import Deck
from simulator.config import SimulatorConfig, vegas_strip
from simulator.game_state import GameState, Action
from simulator.hand_simulator import HandSimulator
from strategies.base import Strategy

from blackjack_rl.state import StateKey, encode_state


@dataclass
class Episode:
    """One played hand as RL sees it.

    steps  : the (state_key, action) pairs the policy went through, in order.
    reward : terminal payout in bet units (bet = 1) — +1 win, +1.5 blackjack, -1 loss,
             0 push; doubles/splits scale accordingly.

    `steps` may be empty (a dealt blackjack resolves with no decision). For a hand with
    splits, `steps` is not a single chain — handled in Stage 3 (D6).
    """
    steps: list[tuple[StateKey, Action]]
    reward: float


class _Recorder(Strategy):
    """Wraps a policy: logs (state_key, action) for each decision, delegates the choice."""

    def __init__(self, policy: Strategy) -> None:
        self._policy = policy
        self.steps: list[tuple[StateKey, Action]] = []

    def decide(self, state: GameState) -> Action:
        action = self._policy.decide(state)
        self.steps.append((encode_state(state), action))
        return action

    def name(self) -> str:
        return self._policy.name()


def problem_a_config() -> SimulatorConfig:
    """Rules for Problem A: the 6-deck S17 3:2 anchor config, counting OFF.

    Starts from `vegas_strip()` (the exact ruleset BasicStrategy's 0.45% edge is measured on)
    and disables counting. Each rollout deals a FRESH shoe, so hands are independent and no
    count can build — which is what makes A a clean, counting-free MDP.
    """
    cfg = vegas_strip()
    cfg.card_counting_allowed = False
    return cfg


def rollout(policy: Strategy, config: SimulatorConfig | None = None) -> Episode:
    """Play one hand with `policy` and return the captured Episode.

    A fresh, shuffled shoe per call makes hands independent (Problem A). Seed the global RNG
    once before calling repeatedly for a reproducible run.
    """
    cfg = config if config is not None else problem_a_config()
    recorder = _Recorder(policy)
    deck = Deck(num_decks=cfg.num_decks)  # fresh shoe, shuffled in __init__
    sim = HandSimulator(cfg, deck, recorder)
    result = sim.play_hand(session_id="ep", bankroll=0.0, bet_size=1.0, hands_played=0)
    return Episode(steps=recorder.steps, reward=result.payout)


def rollout_many(
    policy: Strategy, n: int, config: SimulatorConfig | None = None
) -> Iterator[Episode]:
    """Yield `n` Episodes. Seed the global RNG once before iterating for reproducibility."""
    cfg = config if config is not None else problem_a_config()
    for _ in range(n):
        yield rollout(policy, cfg)
