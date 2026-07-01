"""Bet model — the headline lever (DESIGN D12/D15, build stage B2).

A discrete-spread bet policy: state = (true_count, decks_remaining, bankroll) → action = a
unit-bet level from the spread. Value-based DQN (reuses dqn.QNetwork), trained on log-growth
(Kelly); ruin-aware because bankroll is in-state (D14). Audited vs the analytic full-Kelly curve.

The D17 baseline ladder, bottom to top — each a ``BetPolicy`` (env.py), measured on identical terms:
- ``FlatBet`` (B0) — the constant-wager floor (rung 1); betting variation is the lever above it.
- ``KellyBet`` (B2c) — the analytic full-Kelly bettor (rung 2): sizes from the measured edge-by-count
  curve, the reference the learned ``BetAgent`` is audited against.
- ``BetAgent`` (B2d) — the learned DQN bettor (rung 3); "does RL rediscover Kelly?"
"""
from __future__ import annotations

import random
from math import isfinite, log
from typing import Callable, Sequence

import torch

from blackjack_rl.dqn.agent import QNetwork
from blackjack_rl.dqn.replay import Transition
from blackjack_rl.session.env import BET_SPREAD, GROWTH_BANKROLL, SessionCapture

TC_SCALE = 10.0  # true-count normalizer: counts rarely exceed ~10 in magnitude -> input ~O(1)
FEATURE_DIM = 3  # (true_count, remaining_fraction, bankroll) — the default (raw-bankroll) state width

# how the bankroll enters the bet encoder — the encoding-ablation seam (mirrors dqn.agent's `encoding`):
#   "raw"      : bankroll / scale — absolute wealth on a fixed reference (the default; D14)
#   "logratio" : log(bankroll / scale) — log-wealth vs the reference (Kelly's natural scale; W*f* additive)
#   "none"     : bankroll dropped — the net sizes bets on count + depth only (the wealth-vs-edge test)
BANKROLL_FEATURES: tuple[str, ...] = ("raw", "logratio", "none")


def bet_feature_dim(bankroll_feature: str = "raw") -> int:
    """Input width for the bet encoder — 2 (count, depth) when bankroll is dropped, else 3."""
    if bankroll_feature not in BANKROLL_FEATURES:
        raise ValueError(f"unknown bankroll_feature {bankroll_feature!r}; expected one of {BANKROLL_FEATURES}")
    return 2 if bankroll_feature == "none" else 3


class FlatBet:
    """Constant wager regardless of count or bankroll — a ``BetPolicy`` (env.py). The flat-bet
    floor of the D17 baseline ladder; betting variation is the lever everything above this adds."""

    def __init__(self, amount: float = 1.0) -> None:
        if amount <= 0:
            raise ValueError(f"flat bet amount must be > 0, got {amount}")
        self.amount = amount

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        return self.amount


class KellyBet:
    """Analytic full-Kelly bettor — rung 2 of the D17 ladder (Kelly-bet + basic play), the reference
    the learned ``BetAgent`` (B2d) is audited against (DESIGN D17, build stage B2c).

    Sizes each wager from the measured edge-by-count curve: ``bet = kelly_fraction · f*(count) ·
    bankroll``, where ``f*(count)`` is the full-Kelly fraction at the count from a ``kelly_curve``
    (``references.load_edge_reference().kelly_curve`` — passed in, so the bettor stays pure). Two modes:

    - **discrete** (default) — snap the desired wager to the *nearest* level of the bet ``spread``, the
      SAME finite menu the DQN bettor chooses from. This is the **comparison baseline**: matching the
      action set isolates "analytic vs learned" from "continuous vs discrete" (decision A). At a
      non-positive edge ``f*`` is 0, so the snap lands on the spread minimum — the mandatory table bet
      a counter must place even when Kelly says don't.
    - **continuous** (``discretize=False``) — return ``f*·bankroll`` unrounded (the env still bounds it
      to the spread's [min, max]). The analytic **ceiling**: the discrete→continuous gap is the cost of
      the finite menu.

    ``kelly_fraction`` (default 1.0 = full Kelly) scales the bet: ``c<1`` is fractional Kelly (less
    growth, much less volatility — CONCEPTS §30); ``c>1`` is deliberate over-betting (for the
    ruin-config experiments). A ``BetPolicy`` (env.py): the env clamps the returned wager to the spread
    and the bankroll, so this need not.
    """

    def __init__(
        self,
        kelly_curve: dict[int, float],
        *,
        spread: tuple[int, ...] = BET_SPREAD,
        kelly_fraction: float = 1.0,
        discretize: bool = True,
    ) -> None:
        if not kelly_curve:
            raise ValueError("kelly_curve must be non-empty")
        if kelly_fraction <= 0:
            raise ValueError(f"kelly_fraction must be > 0, got {kelly_fraction}")
        if not spread:
            raise ValueError("spread must have at least one level")
        self.kelly_curve = kelly_curve
        self.spread = spread
        self.kelly_fraction = kelly_fraction
        self.discretize = discretize
        self._keys = sorted(kelly_curve)

    def _fstar(self, true_count: float) -> float:
        """Full-Kelly fraction at ``true_count``: the *nearest measured integer bucket* (clamps outside
        the measured range; the curve already floors at 0 where the edge is non-positive)."""
        rounded = round(true_count)
        nearest = min(self._keys, key=lambda k: abs(k - rounded))
        return self.kelly_curve[nearest]

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        desired = self.kelly_fraction * self._fstar(true_count) * bankroll
        if not self.discretize:
            return desired
        return float(min(self.spread, key=lambda level: abs(level - desired)))


def encode_bet_state(
    true_count: float,
    decks_remaining: float,
    bankroll: float,
    *,
    num_decks: float = 6.0,
    bankroll_scale: float = GROWTH_BANKROLL,
    bankroll_feature: str = "raw",
) -> list[float]:
    """The bet model's state features — the single, isolated encoder (the seam a future learned-count
    agent swaps, CONCEPTS/DESIGN). Always ``true_count / TC_SCALE`` (the edge signal — the count already
    folds in decks-remaining) and ``decks_remaining / num_decks`` (the **remaining-shoe fraction** in
    (0, 1]: scale-free, shoe-size-invariant). ``bankroll_feature`` selects the third — the **encoding
    ablation** for the wealth-vs-edge question (see :data:`BANKROLL_FEATURES`):

    - ``"raw"`` — ``bankroll / bankroll_scale``: bankroll on an **absolute** scale via a FIXED reference
      (DESIGN D14 — lets a bankroll-generalizing agent reuse the encoder). The wealth-scaling this can
      induce (bet ∝ wealth, ignoring edge) is what the ablation probes.
    - ``"logratio"`` — ``log(bankroll / bankroll_scale)``: log-wealth vs the reference — Kelly's natural
      scale (``log(bet) = log(W) + log(f*(count))`` is additive), symmetric around the reference.
    - ``"none"`` — bankroll dropped: the net sizes bets on **count + depth only** (the decisive test —
      does edge-gating emerge / sharpen once the net cannot key on wealth?).
    """
    feats = [true_count / TC_SCALE, decks_remaining / num_decks]
    if bankroll_feature == "raw":
        feats.append(bankroll / bankroll_scale)
    elif bankroll_feature == "logratio":
        feats.append(log(max(bankroll, 1e-9) / bankroll_scale))  # floor guards log(0) at a wiped bankroll
    elif bankroll_feature != "none":
        raise ValueError(f"unknown bankroll_feature {bankroll_feature!r}; expected one of {BANKROLL_FEATURES}")
    return feats


class BetAgent:
    """count, depth, bankroll → discrete bet level: the value-based DQN bettor, rung 3 of the D17 ladder
    (the B2d core; "does RL rediscover Kelly?").

    State ``(true_count, decks_remaining, bankroll)`` encoded by :func:`encode_bet_state`; a ``QNetwork``
    (reused from the play-side DQN) outputs one Q per ``levels`` entry; greedy = argmax. Implements the
    ``IndexedBetPolicy`` protocol (env.py): ``select_level`` is the epsilon-greedy *training* behaviour;
    ``bet`` is the greedy wager (the ``BetPolicy`` contract used for evaluation). The env drives an indexed
    bettor through ``select_level`` and records the chosen index, so :func:`session_to_transitions`
    reconstructs exact action indices — see DESIGN D17, decision 1a.

    ``levels`` is a constructor parameter (default the project ``BET_SPREAD``): the Wonging seam — a
    sit-out agent is ``levels=(0, *BET_SPREAD)`` with no structural change, and ``levels`` is decoupled
    from the env's ``bet_spread`` (which only sets clamp bounds). ``bankroll_scale`` is the fixed bankroll
    reference (decoupled from ``W0``, see :func:`encode_bet_state`). All levels are always legal — the env
    clamps the wager to bankroll/min_wager — so there is no action masking.

    Determinism: the constructor does not seed any RNG. Weights init from torch's global RNG and
    ``select_level`` exploration draws from ``random``; the trainer seeds both once (the A7 convention).
    """

    def __init__(
        self,
        levels: Sequence[float] = BET_SPREAD,
        *,
        hidden: Sequence[int] = (64, 64),
        epsilon: float = 0.1,
        num_decks: float = 6.0,
        bankroll_scale: float = GROWTH_BANKROLL,
        bankroll_feature: str = "raw",
    ) -> None:
        if not levels:
            raise ValueError("levels must have at least one entry")
        self.levels: tuple[float, ...] = tuple(float(x) for x in levels)
        self.epsilon = epsilon
        self.num_decks = num_decks
        self.bankroll_scale = bankroll_scale
        self.bankroll_feature = bankroll_feature  # the encoding-ablation seam (self-describing for persistence)
        self.hidden: tuple[int, ...] = tuple(int(h) for h in hidden)  # self-describing for persistence
        self.q_net = QNetwork(bet_feature_dim(bankroll_feature), len(self.levels), self.hidden)

    def encode_state(self, true_count: float, decks_remaining: float, bankroll: float) -> list[float]:
        """This agent's state vector — :func:`encode_bet_state` bound to its ``num_decks`` /
        ``bankroll_scale`` / ``bankroll_feature``. The trainer passes this as the ``encode`` for
        :func:`session_to_transitions`."""
        return encode_bet_state(
            true_count, decks_remaining, bankroll,
            num_decks=self.num_decks, bankroll_scale=self.bankroll_scale,
            bankroll_feature=self.bankroll_feature,
        )

    def q_values(self, *, true_count: float, decks_remaining: float, bankroll: float) -> torch.Tensor:
        """Raw Q over every level, unmasked. Inference only (no grad); the trainer calls ``self.q_net``
        directly when it needs gradients."""
        device = next(self.q_net.parameters()).device
        x = torch.tensor(
            self.encode_state(true_count, decks_remaining, bankroll), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            return self.q_net(x)

    def greedy_level(self, *, true_count: float, decks_remaining: float, bankroll: float) -> int:
        """Argmax level index — no exploration (the deterministic target policy)."""
        q = self.q_values(true_count=true_count, decks_remaining=decks_remaining, bankroll=bankroll)
        return int(torch.argmax(q).item())

    # --- IndexedBetPolicy contract -------------------------------------------
    def select_level(self, *, true_count: float, decks_remaining: float, bankroll: float) -> int:
        """Epsilon-greedy level index (the training behaviour). Every level is always legal, so the
        exploratory draw is a plain uniform pick over the menu — no legal-action masking."""
        if random.random() < self.epsilon:
            return random.randrange(len(self.levels))
        return self.greedy_level(true_count=true_count, decks_remaining=decks_remaining, bankroll=bankroll)

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        """Greedy wager — the deterministic policy as a plain ``BetPolicy`` (used for evaluation). The env
        drives *training* through ``select_level``; this is the eval/contract path (epsilon ignored)."""
        return self.levels[
            self.greedy_level(true_count=true_count, decks_remaining=decks_remaining, bankroll=bankroll)
        ]


def greedy_bet_curve(
    agent: BetAgent, counts: Sequence[int], *, bankroll: float, decks_remaining: float
) -> dict[int, float]:
    """The agent's greedy wager at each true count (fixed bankroll + shoe depth) — the bet-vs-count
    diagnostic: overlaid on the analytic Kelly curve (B2d-3) and watched forming over training (the
    checkpoint probe). Greedy, so it reads the *policy*, not the exploring behaviour."""
    return {
        c: agent.bet(true_count=float(c), decks_remaining=decks_remaining, bankroll=bankroll)
        for c in counts
    }


def session_to_transitions(
    capture: SessionCapture,
    *,
    encode: Callable[[float, float, float], list[float]],
    n_levels: int,
    ruin_reward: float = -1.0,
    reward_scale: float = 1.0,
) -> list[Transition]:
    """Reconstruct the bettor's TD transitions from a played session — the bet-side analog of
    ``dqn.deep_q.hand_to_transitions``.

    One transition per ``HandRecord`` (unlike the play side, every hand carries a real reward — the bet
    pays off the same hand it is placed; CONCEPTS §31): state = the pre-deal (count, depth, bankroll)
    run through ``encode``; action = the recorded ``bet_level`` (the index the agent *chose*, exact even
    where the env clamped the wager — decision 1a); reward = the log-increment, with a total wipeout's
    ``-inf`` clipped to the finite ``ruin_reward`` (the primary ruin signal is structural — the session
    terminates, forfeiting future growth). ``gamma = 1``, so no discount appears here.

    Every bet level is always legal, so each non-terminal next-state carries an all-True legal mask; the
    last hand is terminal (``done``) with placeholder next-state fields. Requires an indexed capture —
    raises if any ``bet_level`` is None (a plain bettor's capture has no action to credit).
    """
    all_legal = torch.ones(n_levels, dtype=torch.bool)
    no_legal = torch.zeros(n_levels, dtype=torch.bool)
    transitions: list[Transition] = []
    for i, rec in enumerate(capture.hands):
        if rec.bet_level is None:
            raise ValueError(
                "session_to_transitions requires an IndexedBetPolicy capture (bet_level is None); "
                "FlatBet/KellyBet captures have no discrete action to reconstruct"
            )
        state = torch.tensor(
            encode(rec.true_count, rec.decks_remaining, rec.bankroll_before), dtype=torch.float32
        )
        reward = (rec.log_reward if isfinite(rec.log_reward) else ruin_reward) * reward_scale
        if rec.done:  # terminal hand: no bootstrap (placeholder next-state)
            transitions.append(Transition(
                state=state, action=rec.bet_level, reward=reward,
                next_state=torch.zeros_like(state), done=True, next_legal_mask=no_legal,
            ))
        else:  # bootstrap from the next hand's pre-deal state
            nxt = capture.hands[i + 1]
            next_state = torch.tensor(
                encode(nxt.true_count, nxt.decks_remaining, nxt.bankroll_before), dtype=torch.float32
            )
            transitions.append(Transition(
                state=state, action=rec.bet_level, reward=reward,
                next_state=next_state, done=False, next_legal_mask=all_legal,
            ))
    return transitions
