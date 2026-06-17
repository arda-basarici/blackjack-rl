"""Fidelity-axis evaluation — compare the learned policy to basic strategy, cell by cell.

Not just *whether* the agent matches the proven-optimal basic strategy, but *where* it doesn't
and *why*. Each disagreement is split into under_visited (too little data), near_equal_ev
(well-visited but ~tied — an honest non-difference), or genuine_disagreement (well-visited, a
real value gap). See DESIGN.md section 6.

Split-aware: a visited cell may be a 3-tuple (no-split) or a 4-tuple (split mode, with
``can_split``). The canonical comparison state carries that ``can_split`` bit, so the pairs
column is compared with split as a real option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from simulator.game_state import Action, GameState
from strategies.base import Strategy
from strategies.basic_strategy import BasicStrategy

from blackjack_rl.state import StateKey

Category = Literal["agree", "under_visited", "near_equal_ev", "genuine_disagreement"]


class _DiffAgent(Protocol):
    """An agent whose learned tables and greedy policy we can inspect."""

    q: dict[tuple[StateKey, Action], float]
    n: dict[tuple[StateKey, Action], int]

    def greedy_action(self, state: GameState) -> Action: ...


@dataclass(frozen=True)
class CellDiff:
    """One state cell: what each policy does there, and how the disagreement (if any) is judged."""

    player_value: int
    is_soft: bool
    dealer_upcard: int
    can_split: bool
    visits: int
    agent_action: Action
    basic_action: Action
    agent_q: float
    basic_q: float
    category: Category


@dataclass(frozen=True)
class DiffReport:
    """All compared cells plus summary fidelity numbers."""

    cells: tuple[CellDiff, ...]
    agreement_unweighted: float
    agreement_weighted: float
    category_counts: dict[str, int]


def classify(
    agree: bool, visits: int, ev_gap: float, min_visits: int, ev_tol: float
) -> Category:
    """Bucket a cell. ``ev_gap`` is |Q(agent's action) - Q(basic's action)| by the agent's own
    estimates. Pure function so the rule is testable in isolation."""
    if agree:
        return "agree"
    if visits < min_visits:
        return "under_visited"
    if ev_gap < ev_tol:
        return "near_equal_ev"
    return "genuine_disagreement"


def _canonical_state(
    player_value: int, is_soft: bool, dealer_upcard: int, can_split: bool = False
) -> GameState:
    """A 2-card decision state for cell-by-cell comparison: doubling allowed, surrender off, and
    split allowed iff this cell is a splittable pair — so the agent and basic strategy choose
    over the same actions for that cell (including the pairs column)."""
    return GameState(
        player_value=player_value,
        player_is_soft=is_soft,
        player_card_count=2,
        dealer_upcard=dealer_upcard,
        can_hit=True,
        can_stand=True,
        can_double=True,
        can_split=can_split,
        can_surrender=False,
    )


def _unpack(state_key: StateKey) -> tuple[int, bool, int, bool]:
    """(player_value, is_soft, dealer_upcard, can_split) — can_split is False for 3-tuple keys."""
    if len(state_key) == 4:
        player_value, is_soft, dealer_upcard, can_split = state_key
        return player_value, is_soft, dealer_upcard, can_split
    player_value, is_soft, dealer_upcard = state_key
    return player_value, is_soft, dealer_upcard, False


def diff_policy(
    agent: _DiffAgent,
    basic: Strategy | None = None,
    min_visits: int = 1000,
    ev_tol: float = 0.02,
) -> DiffReport:
    """Compare ``agent``'s greedy policy to ``basic`` over every cell the agent has visited."""
    basic_strategy: Strategy = basic if basic is not None else BasicStrategy()

    visits_by_state: dict[StateKey, int] = {}
    for (state_key, _action), count in agent.n.items():
        visits_by_state[state_key] = visits_by_state.get(state_key, 0) + count

    cells: list[CellDiff] = []
    for state_key in sorted(visits_by_state):
        player_value, is_soft, dealer_upcard, can_split = _unpack(state_key)
        state = _canonical_state(player_value, is_soft, dealer_upcard, can_split)
        agent_action = agent.greedy_action(state)
        basic_action = basic_strategy.decide(state)
        agent_q = agent.q.get((state_key, agent_action), 0.0)
        basic_q = agent.q.get((state_key, basic_action), 0.0)
        visits = visits_by_state[state_key]
        category = classify(
            agent_action == basic_action, visits, abs(agent_q - basic_q), min_visits, ev_tol
        )
        cells.append(
            CellDiff(
                player_value=player_value, is_soft=is_soft, dealer_upcard=dealer_upcard,
                can_split=can_split, visits=visits, agent_action=agent_action,
                basic_action=basic_action, agent_q=agent_q, basic_q=basic_q, category=category,
            )
        )

    total_visits = sum(visits_by_state.values())
    agreeing = [c for c in cells if c.category == "agree"]
    agreement_unweighted = len(agreeing) / len(cells) if cells else 0.0
    agreement_weighted = (
        sum(c.visits for c in agreeing) / total_visits if total_visits else 0.0
    )
    category_counts: dict[str, int] = {}
    for c in cells:
        category_counts[c.category] = category_counts.get(c.category, 0) + 1

    return DiffReport(
        cells=tuple(cells),
        agreement_unweighted=agreement_unweighted,
        agreement_weighted=agreement_weighted,
        category_counts=category_counts,
    )
