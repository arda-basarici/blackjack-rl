"""Tabular Monte Carlo agent — a Q-table policy, exposed as a Phase 2 Strategy.

Holds Q(state, action) value estimates and N(state, action) visit counts. `decide` is the
training behaviour policy (epsilon-greedy); `greedy_action` is the target policy used for
evaluation; `update` folds a return into the running mean. Credit assignment — which return G
goes to which (state, action) — is the trainer's job (A1, DESIGN D1); this agent only stores
and averages, and does not know where G came from.
"""
from __future__ import annotations

import random

from simulator.game_state import Action, GameState
from strategies.base import Strategy

from blackjack_rl.state import StateKey, encode_state

# Actions the agent chooses among in Stage 2: no `split` (D6, learned in Stage 3) and no
# `surrender` (off in problem_a_config). `double` is included but only when the state allows it.
_STAGE2_ACTIONS: tuple[Action, ...] = ("hit", "stand", "double")


class TabularAgent(Strategy):
    """A Q-table policy. Q and N are public so the trainer, evaluator, and persistence can
    read/write them directly."""

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.q: dict[tuple[StateKey, Action], float] = {}
        self.n: dict[tuple[StateKey, Action], int] = {}

    # --- Strategy contract (training behaviour policy) -----------------------
    def decide(self, state: GameState) -> Action:
        """Epsilon-greedy over the legal Stage-2 actions."""
        actions = self._legal_actions(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self._greedy_over(encode_state(state), actions)

    def name(self) -> str:
        return "TabularAgent"

    # --- target policy (used for evaluation) ---------------------------------
    def greedy_action(self, state: GameState) -> Action:
        """Pure argmax over the legal Stage-2 actions — no exploration."""
        return self._greedy_over(encode_state(state), self._legal_actions(state))

    # --- learning bookkeeping (called by the trainer) ------------------------
    def update(self, key: StateKey, action: Action, g: float) -> None:
        """Fold return `g` into the running mean Q(key, action): N += 1; Q += (g - Q) / N."""
        k = (key, action)
        self.n[k] = self.n.get(k, 0) + 1
        q = self.q.get(k, 0.0)
        self.q[k] = q + (g - q) / self.n[k]

    # --- helpers -------------------------------------------------------------
    def _legal_actions(self, state: GameState) -> list[Action]:
        legal = [a for a in state.legal_actions() if a in _STAGE2_ACTIONS]
        return legal or ["stand"]  # hit/stand are always legal; guard the empty case anyway

    def _greedy_over(self, key: StateKey, actions: list[Action]) -> Action:
        best_val = max(self.q.get((key, a), 0.0) for a in actions)
        best = [a for a in actions if self.q.get((key, a), 0.0) == best_val]
        return random.choice(best)  # random tie-break, so no bias toward the first action
