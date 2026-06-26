"""Outcome & risk metrics for Problem B (DESIGN D17, build stages B2+).

Two axes, never collapsed (mirrors §7's discipline, adapted to B):
- outcome — log-growth rate per hand; final-bankroll distribution,
- risk    — probability of ruin (headline for a finite bankroll); drawdown / variance.
Evaluated through the engine on the same terms for every rung of the baseline ladder.
"""
from __future__ import annotations


def growth_rate(bankroll_path):  # -> float
    raise NotImplementedError("B2: mean per-hand log-wealth increment (the log-growth rate)")


def ruin_probability(*, n_sessions: int, seed: int = 0):  # -> float
    raise NotImplementedError("B2: finite-bankroll session sims → fraction that hit ruin")


def bankroll_distribution(*, n_sessions: int, seed: int = 0):
    raise NotImplementedError("B2: final-bankroll distribution across sessions")
