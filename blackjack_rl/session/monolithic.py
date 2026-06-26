"""Monolithic baseline — the comparison (DESIGN D12, build stage B4).

One network that learns *both* play and bet end-to-end on log-growth (sees the full state:
hand + count + bankroll). The honest question: does end-to-end joint learning recover the
principled factored decomposition (play-EV + Kelly-bet), or does conflating them hurt?
"""
from __future__ import annotations


class MonolithicAgent:
    """End-to-end play+bet on log-growth. Compared against the factored play+bet system."""

    def __init__(self) -> None:
        raise NotImplementedError("B4: joint state/action, end-to-end log-growth training")
