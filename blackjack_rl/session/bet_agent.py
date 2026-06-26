"""Bet model — the headline lever (DESIGN D12/D15, build stage B2).

A discrete-spread bet policy: state = (true_count, decks_remaining[, bankroll]) → action = a
unit-bet level from the spread. Value-based DQN (reuses dqn.QNetwork), trained on log-growth
(Kelly); ruin-aware because bankroll is in-state (D14). Audited vs the analytic full-Kelly curve.
"""
from __future__ import annotations


class BetAgent:
    """count[,bankroll] → discrete bet level. Greedy = argmax over the spread's Q-vector."""

    def __init__(self) -> None:
        raise NotImplementedError("B2: bet network (reuse dqn.QNetwork), state encoding, decide()")
