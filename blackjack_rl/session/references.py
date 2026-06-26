"""Reconstructed ground-truth references for Problem B (DESIGN D17, build stage B1).

B has "no clean table" (§3), but we still audit against reconstructed truth:
- `edge_by_count`  — empirical house edge vs true_count (basic strategy, many hands),
- `kelly_bet_curve`— the analytic full-Kelly bet fraction implied by edge-by-count,
- `index_plays`    — the known count-deviation index plays, to audit learned deviations.
"""
from __future__ import annotations


def edge_by_count(*, n_hands: int, seed: int = 0):  # -> dict[int, float]
    raise NotImplementedError("B1: measure edge as a function of true_count via the engine")


def kelly_bet_curve(edge_by_count):  # -> dict[int, float]
    raise NotImplementedError("B1: full-Kelly fraction from edge-by-count (the bet reference)")


def index_plays():  # -> reference deviation table
    raise NotImplementedError("B1: known index-play deviations (play-side reference)")
