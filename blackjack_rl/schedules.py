"""Value schedules over training — constant or decaying (toward GLIE / convergence).

A schedule maps an episode index (0 .. num_episodes-1) to a value. Originally for *exploration*
(epsilon: explore hard early to sample rare actions, anneal toward ~0 late), the same machinery now
also drives the DQN *learning rate* (a decaying step size lets the value estimate converge to a
point, mimicking the tabular agent's ``1/n`` step — see CONCEPTS §26). All decay schedules reach
``end`` exactly at the final episode. See A8.
"""
from __future__ import annotations

from typing import Callable

Schedule = Callable[[int], float]
EpsilonSchedule = Schedule  # back-compat alias (schedules are generic; epsilon was the first user)
KINDS: tuple[str, ...] = ("constant", "linear", "exponential", "harmonic")


def make_schedule(
    kind: str, *, constant: float, start: float, end: float, num_episodes: int, hold: int = 0
) -> Schedule:
    """Build a value(episode_index) schedule function.

    constant     : fixed at ``constant``.
    linear       : start -> end, straight line (can reach 0).
    exponential  : start -> end, geometric (needs end > 0).
    harmonic     : start -> end, 1/k-shaped, long thin tail (needs end > 0).

    ``hold`` (episodes, decaying kinds only): stay flat at ``start`` until episode ``hold``, then
    decay ``start`` -> ``end`` over the *remaining* episodes. This keeps the step high through an early
    curriculum phase (e.g. learning hit/stand before ``double_after``) and only starts decaying once
    the hard, high-variance action is introduced — flat-then-decay. ``hold=0`` (default) decays from
    episode 0 (original behavior). Ignored for ``constant``.
    """
    last = max(1, num_episodes - 1)
    if kind not in KINDS:
        raise ValueError(f"unknown schedule {kind!r}; expected one of {KINDS}")
    if kind == "constant":
        return lambda i: constant
    if kind == "exponential" and (start <= 0 or end <= 0):
        raise ValueError("exponential schedule needs start > 0 and end > 0")
    if kind == "harmonic" and end <= 0:
        raise ValueError("harmonic schedule needs end > 0")

    def shape(p: float) -> float:  # value at fraction p in [0, 1] of the decay phase
        if kind == "linear":
            return start + (end - start) * p
        if kind == "exponential":
            return start * (end / start) ** p
        return start / (1.0 + (start / end - 1.0) * p)  # harmonic

    if hold <= 0:
        return lambda i: shape(i / last)
    span = max(1, last - hold)
    return lambda i: start if i < hold else shape((i - hold) / span)


# Back-compat alias: the schedule machinery is generic (it also drives the DQN learning rate).
# ``make_epsilon_schedule`` was the original name from when epsilon was its only consumer.
make_epsilon_schedule = make_schedule
