"""Epsilon schedules for exploration — constant or decaying (toward GLIE).

A schedule maps an episode index (0 .. num_episodes-1) to an exploration rate. Decaying
schedules let us explore hard early (sample rare actions like soft doubles) and anneal toward
~0 late (so the final Q-values reflect near-optimal continuations, keeping the common stiff
hands correct). All decay schedules reach ``end`` exactly at the final episode. See A8.
"""
from __future__ import annotations

from typing import Callable

EpsilonSchedule = Callable[[int], float]
KINDS: tuple[str, ...] = ("constant", "linear", "exponential", "harmonic")


def make_epsilon_schedule(
    kind: str, *, constant: float, start: float, end: float, num_episodes: int
) -> EpsilonSchedule:
    """Build an epsilon(episode_index) function.

    constant     : fixed at ``constant``.
    linear       : start -> end, straight line (can reach 0).
    exponential  : start -> end, geometric (needs end > 0).
    harmonic     : start -> end, 1/k-shaped, long thin tail (needs end > 0).
    """
    last = max(1, num_episodes - 1)

    if kind == "constant":
        return lambda i: constant
    if kind == "linear":
        return lambda i: start + (end - start) * (i / last)
    if kind == "exponential":
        if start <= 0 or end <= 0:
            raise ValueError("exponential schedule needs start > 0 and end > 0")
        return lambda i: start * (end / start) ** (i / last)
    if kind == "harmonic":
        if end <= 0:
            raise ValueError("harmonic schedule needs end > 0")
        c = start / end - 1.0
        return lambda i: start / (1.0 + c * (i / last))
    raise ValueError(f"unknown schedule {kind!r}; expected one of {KINDS}")
