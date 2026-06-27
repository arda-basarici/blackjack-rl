"""Session environment — the Problem B MDP (DESIGN D11/D14, build stage B0).

Many hands from one persisting, depleting shoe (counting on), against a finite bankroll with a
hard ruin barrier. Per-hand reward is the log-wealth increment ``log(W_after / W_before)`` so the
session return is log-growth (Kelly, D11). Reuses the Phase-2 engine's counting/session surface
(``Deck`` + ``HiLoCount`` → ``true_count``; ``HandSimulator.play_hand`` → settled ``payout``) — no
engine changes.

This is a **capture driver**, the Problem-B analog of ``core.env`` (which captures single Problem-A
hands): given a play ``Strategy`` and a ``BetPolicy`` it plays a whole session and returns a
``SessionCapture`` whose per-hand ``HandRecord``s are exactly the bettor's transitions
``(state, bet, log_reward, done)`` — so the B2 trainer consumes them the way the DQN trainer consumes
``CapturedHand`` today. One code path serves both training-data capture and baseline evaluation, so
every rung of the ladder (D17) is measured on identical terms.

Reproducibility: the engine shuffles with Python's global ``random``; seed ONCE before a batch
(``run_sessions`` does this), never per hand. See CONCEPTS.md #14.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import log
from typing import Iterator, Protocol

from simulator.card import Deck
from simulator.config import SimulatorConfig, vegas_strip
from simulator.counting import HiLoCount
from simulator.hand_simulator import HandSimulator
from strategies.base import Strategy


class BetPolicy(Protocol):
    """The env's bet-side input contract: choose a wager from the pre-deal session state.

    Decoupled from the engine's ``BettingStrategy`` (which needs a full ``GameState``) because the
    bet is decided *before* the deal — only the count, shoe depth, and bankroll are known. The
    flat-bet baseline, the B1 Kelly bettor, and the B2 DQN bettor all implement this.
    """

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        """Return the desired wager (env clamps it to the spread and the bankroll)."""
        ...


BET_SPREAD: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
"""The project's single bet ladder (B2b) — one shared **arithmetic** spread, held constant across
every config and experiment so all measured differences attribute to the variable under study
(bankroll, encoding, algorithm), never to the action set.

Arithmetic because Kelly's target is a ~linear ramp in true count (edge ≈ linear in TC, per-hand
variance ≈ constant), which uniform steps track with uniform error; a geometric ladder (1,2,4,8)
would over-resolve the bottom and jump at the top. The top of 8u ≈ full Kelly at TC +6 against the
growth bankroll, and that same 8u is over-betting headroom at the (smaller) ruin bankroll — one
ladder serving both regimes. (This is a *reasoned* design choice, not a measured arithmetic-beats-
geometric result.)"""

GROWTH_BANKROLL = 400.0
RUIN_BANKROLL = 200.0


@dataclass(frozen=True)
class SessionConfig:
    """Knobs for a Problem B session — the MDP/risk parameters (env config lives with the env,
    like ``problem_a_config``; the *training* hyperparameters stay in core.config).

    starting_bankroll : initial wealth, in **min-bet units** (problem_b_config sets min_bet = 1).
    ruin_threshold    : bankroll at/below this ends the session (terminal). Must be >= the spread
                        floor (enforced): below the minimum wager you cannot place a hand, which *is*
                        ruin — and it keeps the bankroll cap from forcing a sub-minimum bet.
    bet_spread        : the discrete wager levels the bettor may choose (D15). Defaults to the
                        project-wide ``BET_SPREAD`` (1..8, fixed); the canonical regimes are the
                        named ``growth_config`` / ``ruin_config`` below, which differ only in bankroll.
    max_hands         : horizon cap (terminal). A session ends at ruin or here, whichever first —
                        so the env always terminates even under positive (counting) drift.
    seed              : global-RNG seed consumed by ``run_sessions`` (the batch entry point), so a
                        batch of sessions is reproducible. A single ``SessionEnv.run`` does not seed.
    """

    starting_bankroll: float = 100.0
    ruin_threshold: float = 1.0
    bet_spread: tuple[int, ...] = BET_SPREAD
    max_hands: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.starting_bankroll <= self.ruin_threshold:
            raise ValueError(
                f"starting_bankroll ({self.starting_bankroll}) must exceed "
                f"ruin_threshold ({self.ruin_threshold}) — else the session starts ruined"
            )
        if self.ruin_threshold < 0:
            raise ValueError(f"ruin_threshold must be >= 0, got {self.ruin_threshold}")
        if not self.bet_spread:
            raise ValueError("bet_spread must have at least one level")
        if any(level <= 0 for level in self.bet_spread):
            raise ValueError(f"bet_spread levels must be > 0, got {self.bet_spread}")
        if self.ruin_threshold < min(self.bet_spread):
            raise ValueError(
                f"ruin_threshold ({self.ruin_threshold}) must be >= the spread floor "
                f"({min(self.bet_spread)}) — else the bankroll cap can force a sub-minimum bet"
            )
        if self.max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {self.max_hands}")


def problem_b_config(min_bet: float = 1.0) -> SimulatorConfig:
    """Phase-2 ``SimulatorConfig`` for Problem B: 6-deck S17 3:2, **counting on, shoe persists**.

    Mirror of ``core.env.problem_a_config`` but with Hi-Lo counting enabled and no per-hand
    reshuffle (the shoe depletes across hands; the env reshuffles at the config's penetration). Bets
    and bankroll are kept in **min-bet units** by setting ``min_bet = 1`` (the default), so the
    spread ``(1, 2, …)`` and the bankroll share one scale and the Kelly math reads in units.
    """
    cfg = vegas_strip()
    cfg.card_counting_allowed = True
    cfg.shuffle_every_round = False
    cfg.min_bet = min_bet
    return cfg


def growth_config(seed: int = 42) -> SessionConfig:
    """Growth regime (D14): the bankroll is fat enough that the spread tops out near full Kelly at
    the highest count we size for (TC +6: ``f* ≈ 1.96%`` → ``0.0196 × 400 ≈ 8u``), so the bettor
    rides the Kelly ramp and the ruin barrier stays **dormant** — ruin is a tail non-event. The
    ruin-dormant half of the two-config growth-vs-ruin comparison."""
    return SessionConfig(starting_bankroll=GROWTH_BANKROLL, bet_spread=BET_SPREAD, seed=seed)


def ruin_config(seed: int = 42) -> SessionConfig:
    """Ruin regime (D14): the bankroll is lean enough that the barrier is reachable by **over**-
    betting — the spread's 8u top is ~2× full Kelly here, so the ladder offers headroom the bettor
    must decline. The learnable result is a **bet-ceiling that grows without courting ruin**,
    tightening as the bankroll shrinks. (Note: measured — full-Kelly-fraction sizing never ruins
    here; only over-betting does. So the result is 'learn the ceiling', not 'bend below continuous
    Kelly'.) The non-degenerate axis of the growth-vs-ruin comparison."""
    return SessionConfig(starting_bankroll=RUIN_BANKROLL, bet_spread=BET_SPREAD, seed=seed)


@dataclass(frozen=True)
class HandRecord:
    """One hand as the bettor's MDP sees it — a transition the B2 trainer reconstructs from.

    The state at decision time is ``(true_count, decks_remaining, bankroll_before)``; the action is
    ``bet``; the reward is ``log_reward``. The *next* state is the following record's pre-deal state
    (or terminal when ``done``), so it is not stored here — exactly as ``CapturedHand`` leaves s' to
    the trainer. ``payout`` is the engine's signed net (already scaled by the bet).

    ``log_reward = log(bankroll_after / bankroll_before)``, except a total wipe-out
    (``bankroll_after == 0``) is ``-inf`` — mathematically honest; B2 chooses any finite ruin
    penalty as a reward-shaping decision there.
    """

    true_count: float
    decks_remaining: float
    bankroll_before: float
    bet: float
    payout: float
    bankroll_after: float
    log_reward: float
    done: bool


@dataclass(frozen=True)
class SessionCapture:
    """A played session: the per-hand ``hands`` plus terminal summary. ``ruined`` distinguishes the
    two terminal causes (ruin vs the ``max_hands`` horizon)."""

    hands: list[HandRecord]
    ruined: bool
    final_bankroll: float
    starting_bankroll: float

    @property
    def n_hands(self) -> int:
        return len(self.hands)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SessionEnv:
    """Drives one session: per hand → read (count, depth, bankroll) pre-deal → choose bet → play →
    settle bankroll → emit a ``HandRecord``; terminate on ruin or the horizon. Owns the persistent
    shoe and reshuffles it at the config's penetration."""

    def __init__(self, config: SessionConfig, sim_config: SimulatorConfig | None = None) -> None:
        self.config = config
        self.sim_config = sim_config if sim_config is not None else problem_b_config()

    def run(self, play: Strategy, bet: BetPolicy) -> SessionCapture:
        """Play one session off the *current* global-RNG state (does not seed — see ``run_sessions``).

        A fresh shuffled shoe and the full starting bankroll, as a counter sitting down at a new
        table. ``play`` decides hand actions through the engine; ``bet`` sizes each wager from the
        pre-deal state.
        """
        cfg = self.sim_config
        deck = Deck(num_decks=cfg.num_decks, counting_system=HiLoCount())
        bankroll = self.config.starting_bankroll
        spread_lo = float(min(self.config.bet_spread))
        spread_hi = float(max(self.config.bet_spread))

        records: list[HandRecord] = []
        ruined = False
        for hands_played in range(self.config.max_hands):
            if deck.needs_shuffle(cfg.penetration, cfg.shuffle_every_round):
                deck.build()

            true_count = deck.true_count
            decks_remaining = deck.cards_remaining() / 52.0
            wager = bet.bet(
                true_count=true_count, decks_remaining=decks_remaining, bankroll=bankroll
            )
            wager = min(_clamp(wager, spread_lo, spread_hi), bankroll)  # never wager more than held

            result = HandSimulator(cfg, deck, play).play_hand(
                session_id="b", bankroll=bankroll, bet_size=wager, hands_played=hands_played
            )

            bankroll_after = max(0.0, bankroll + result.payout)
            ruined = bankroll_after <= self.config.ruin_threshold
            done = ruined or hands_played == self.config.max_hands - 1
            log_reward = log(bankroll_after / bankroll) if bankroll_after > 0 else float("-inf")

            records.append(
                HandRecord(
                    true_count=true_count,
                    decks_remaining=decks_remaining,
                    bankroll_before=bankroll,
                    bet=wager,
                    payout=result.payout,
                    bankroll_after=bankroll_after,
                    log_reward=log_reward,
                    done=done,
                )
            )
            bankroll = bankroll_after
            if ruined:
                break

        return SessionCapture(
            hands=records,
            ruined=ruined,
            final_bankroll=bankroll,
            starting_bankroll=self.config.starting_bankroll,
        )


def run_sessions(
    config: SessionConfig,
    play: Strategy,
    bet: BetPolicy,
    n: int,
    sim_config: SimulatorConfig | None = None,
) -> Iterator[SessionCapture]:
    """Yield ``n`` reproducible sessions: seed the global RNG **once** from ``config.seed`` (CONCEPTS
    #14), then play ``n`` independent sessions (each a fresh shuffled shoe + full bankroll) off the
    one stream — varied but reproducible as a batch. The B-side analog of ``core.env.rollout_many``.
    """
    random.seed(config.seed)
    env = SessionEnv(config, sim_config)
    for _ in range(n):
        yield env.run(play, bet)
