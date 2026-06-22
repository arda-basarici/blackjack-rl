"""Fidelity evaluation for a *network* policy — by interrogation, not stored tables.

The tabular diff (``policy_diff.diff_policy``) reads the agent's stored Q-table and visit counts.
A ``DQNAgent`` has neither: its policy is a *deterministic function* over a tiny cell space. So we
**materialize** its policy by walking every canonical decision cell and querying the network (one
forward pass per cell) for its Q-values, then reuse ``diff_policy`` unchanged — producing a
``DiffReport`` directly comparable to the tabular one.

No visit counts, on purpose: the network has an answer at *every* cell by generalization, so there
is no per-cell coverage to measure — the ``under_visited`` category does not apply, and the diff is
run with ``min_visits=0``. This is the generalization-vs-coverage point (CONCEPTS.md section 18)
made concrete: a table only has entries where it was visited; the net fills the whole grid.

Split-aware: when the agent plays splits (``agent.with_splits``), the 10-pair column is enumerated
and scored against basic strategy too (each pair is its own ``can_split=True`` cell; A11).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from simulator.game_state import Action, GameState

from blackjack_rl.agents.dqn import DQNAgent
from blackjack_rl.evaluation.policy_diff import DiffReport, diff_policy
from blackjack_rl.state import StateKey

# The canonical no-split decision grid: hard totals 5..20, soft totals 13..20 (A,2 .. A,9),
# dealer upcard 2..11 (11 = ace). The same 2-card abstraction the basic-strategy table uses.
_HARD_TOTALS: range = range(5, 21)
_SOFT_TOTALS: range = range(13, 21)
_DEALER_UPCARDS: range = range(2, 12)
# Splittable pairs as (player_value, is_soft): 2,2..10,10 are hard (value = 2x the card), A,A is
# soft 12. Appended as a separate can_split=True column when the agent plays splits (A11) — each
# pair is its own cell so it never collides with the same-total no-split hand (8,8 vs hard 16).
_PAIRS: tuple[tuple[int, bool], ...] = (
    (4, False), (6, False), (8, False), (10, False), (12, False),
    (14, False), (16, False), (18, False), (20, False), (12, True),
)


def enumerate_cells(with_splits: bool = False) -> list[tuple[int, bool, int, bool]]:
    """Every canonical decision cell as ``(player_value, is_soft, dealer_upcard, can_split)``. The
    no-split grid is always present; with ``with_splits`` the 10-pair split column is appended."""
    cells: list[tuple[int, bool, int, bool]] = []
    for upcard in _DEALER_UPCARDS:
        for value in _HARD_TOTALS:
            cells.append((value, False, upcard, False))
        for value in _SOFT_TOTALS:
            cells.append((value, True, upcard, False))
    if with_splits:
        for upcard in _DEALER_UPCARDS:
            for value, is_soft in _PAIRS:
                cells.append((value, is_soft, upcard, True))
    return cells


def _state_for(value: int, is_soft: bool, upcard: int, can_split: bool = False,
               can_surrender: bool = False) -> GameState:
    """A 2-card decision state for one cell: doubling allowed; split allowed iff a splittable pair;
    surrender allowed iff the agent plays it — so the network and basic strategy choose over the
    same actions."""
    return GameState(
        player_value=value,
        player_is_soft=is_soft,
        player_card_count=2,
        dealer_upcard=upcard,
        can_hit=True,
        can_stand=True,
        can_double=True,
        can_split=can_split,
        can_surrender=can_surrender,
    )


@dataclass
class _MaterializedPolicy:
    """A ``DQNAgent``'s policy materialized onto the cells, exposing the ``_DiffAgent`` interface
    (``q``, ``n``, ``greedy_action``) so ``diff_policy`` can read it like a tabular agent."""

    agent: DQNAgent
    q: dict[tuple[StateKey, Action], float] = field(default_factory=dict)
    n: dict[tuple[StateKey, Action], int] = field(default_factory=dict)

    def greedy_action(self, state: GameState) -> Action:
        return self.agent.greedy_action(state)


def materialize(agent: DQNAgent) -> _MaterializedPolicy:
    """Walk every canonical cell, query the network's Q-values there, and fill a tabular-shaped
    view. Visit counts are a constant 1 placeholder — the network has no per-cell coverage. Pair
    cells use a 4-tuple key (with can_split) so they stay distinct from the same-total no-split cell;
    no-split cells keep the original 3-tuple key, so prior no-split runs reproduce exactly."""
    table = _MaterializedPolicy(agent=agent)
    for value, is_soft, upcard, can_split in enumerate_cells(agent.with_splits):
        key: StateKey = (value, is_soft, upcard, True) if can_split else (value, is_soft, upcard)
        st = _state_for(value, is_soft, upcard, can_split, agent.with_surrender)
        q_values = agent.q_values(st)  # over agent.actions order
        for i, action in enumerate(agent.actions):
            table.q[(key, action)] = float(q_values[i].item())
        table.n[(key, agent.actions[0])] = 1  # one entry so the cell appears in the diff
    return table


def diff_network(agent: DQNAgent, ev_tol: float = 0.02) -> DiffReport:
    """Fidelity report for a trained network vs basic strategy. Materializes the net's policy by
    querying every cell, then reuses ``diff_policy`` with ``min_visits=0`` (no ``under_visited`` for
    a generalizing model). Surrender is offered in the comparison iff the agent plays it."""
    return diff_policy(
        materialize(agent), min_visits=0, ev_tol=ev_tol, with_surrender=agent.with_surrender
    )
