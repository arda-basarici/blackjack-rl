"""Bet model — the headline lever (DESIGN D12/D15, build stage B2).

A discrete-spread bet policy: state = (true_count, decks_remaining[, bankroll]) → action = a
unit-bet level from the spread. Value-based DQN (reuses dqn.QNetwork), trained on log-growth
(Kelly); ruin-aware because bankroll is in-state (D14). Audited vs the analytic full-Kelly curve.

The D17 baseline ladder, bottom to top — each a ``BetPolicy`` (env.py), measured on identical terms:
- ``FlatBet`` (B0) — the constant-wager floor (rung 1); betting variation is the lever above it.
- ``KellyBet`` (B2c) — the analytic full-Kelly bettor (rung 2): sizes from the measured edge-by-count
  curve, the reference the learned ``BetAgent`` is audited against.
- ``BetAgent`` (B2d) — the learned DQN bettor (rung 3); "does RL rediscover Kelly?"
"""
from __future__ import annotations

from blackjack_rl.session.env import BET_SPREAD


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


class BetAgent:
    """count[,bankroll] → discrete bet level. Greedy = argmax over the spread's Q-vector."""

    def __init__(self) -> None:
        raise NotImplementedError("B2: bet network (reuse dqn.QNetwork), state encoding, decide()")
