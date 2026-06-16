"""The state contract — the single definition of "what is a state" in blackjack_rl.

Both the environment (recording a trajectory) and the agent (looking up a value) must agree
*exactly* on how a full GameState collapses to a key, or the policy-diff and visit-counts
silently disagree. So that mapping lives here, in one place. See DESIGN.md D2 / section 4.

Problem A state: (player_value, player_is_soft, dealer_upcard). No counting — A is a single
round, so deck composition is irrelevant. Problem B will extend this (count buckets).
"""
from simulator.game_state import GameState

# A hashable key into the Q-table and visit-count tables.
StateKey = tuple[int, bool, int]


def encode_state(state: GameState) -> StateKey:
    """Collapse a GameState to the Problem A state key: (player_value, is_soft, dealer_upcard).

    Hashable, so it can be used directly as a dict key.
    """
    return (state.player_value, state.player_is_soft, state.dealer_upcard)
