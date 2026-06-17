"""The state contract — the single definition of "what is a state" in blackjack_rl.

Both the environment (recording a trajectory) and the agent (looking up a value) must agree
*exactly* on how a state collapses to a key, or the policy-diff and visit-counts silently
disagree. So that mapping lives here, in one place. See DESIGN.md D2 / section 4, and A11.

Problem A (no splits): (player_value, player_is_soft, dealer_upcard). With splits enabled a
fourth element ``can_split`` is appended — one bit that, together with value + soft, pins the
pair rank (each pairable value maps to a unique pair), so the agent can learn the split
column. The encoding is config-driven (``with_splits``) so no-split runs stay reproducible.
"""
from __future__ import annotations

from typing import Protocol


class StateLike(Protocol):
    """Anything carrying the fields a state key needs — e.g. a GameState or a DecisionRecord."""

    player_value: int
    player_is_soft: bool
    dealer_upcard: int
    can_split: bool


# Hashable key into the Q-table / visit-count tables: a 3-tuple in no-split mode, with a
# fourth ``can_split`` bool appended in split mode.
StateKey = tuple[int, bool, int] | tuple[int, bool, int, bool]


def encode_state(state: StateLike, with_splits: bool = False) -> StateKey:
    """Collapse a state to its key. Default: (player_value, is_soft, dealer_upcard); with
    ``with_splits`` it appends ``can_split``. Hashable — usable directly as a dict key."""
    base = (state.player_value, state.player_is_soft, state.dealer_upcard)
    return (*base, state.can_split) if with_splits else base
