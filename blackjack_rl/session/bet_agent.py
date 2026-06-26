"""Bet model — the headline lever (DESIGN D12/D15, build stage B2).

A discrete-spread bet policy: state = (true_count, decks_remaining[, bankroll]) → action = a
unit-bet level from the spread. Value-based DQN (reuses dqn.QNetwork), trained on log-growth
(Kelly); ruin-aware because bankroll is in-state (D14). Audited vs the analytic full-Kelly curve.

``FlatBet`` lands first (B0): the constant-wager baseline — rung 1 of the D17 ladder (flat-bet +
basic) and the floor the counting bettor is measured against.
"""
from __future__ import annotations


class FlatBet:
    """Constant wager regardless of count or bankroll — a ``BetPolicy`` (env.py). The flat-bet
    floor of the D17 baseline ladder; betting variation is the lever everything above this adds."""

    def __init__(self, amount: float = 1.0) -> None:
        if amount <= 0:
            raise ValueError(f"flat bet amount must be > 0, got {amount}")
        self.amount = amount

    def bet(self, *, true_count: float, decks_remaining: float, bankroll: float) -> float:
        return self.amount


class BetAgent:
    """count[,bankroll] → discrete bet level. Greedy = argmax over the spread's Q-vector."""

    def __init__(self) -> None:
        raise NotImplementedError("B2: bet network (reuse dqn.QNetwork), state encoding, decide()")
