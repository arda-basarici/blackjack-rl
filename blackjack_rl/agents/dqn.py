"""Deep Q-network agent — a neural-net Q-function, exposed as a Phase 2 Strategy.

Where ``TabularAgent`` (agents/tabular.py) stores Q(state, action) in a dict, this agent
*approximates* it with a small neural network: state features in, one Q-value per action out.
The two are deliberately interchangeable behind the Strategy contract (DESIGN D2) — ``decide``
(epsilon-greedy behaviour) and ``greedy_action`` (argmax target) share signatures — so the
existing evaluator, policy-diff, and env harness drive either one unchanged. See DESIGN D4 (the
net as a deliberate literacy experiment) and CONCEPTS.md sections 17-18.

This module is the *function approximator* only: the network, the scalar feature encoding, and
action selection with legal-action masking. Training (replay buffer, target network, the TD
loop) lives separately. With random initial weights this agent plays arbitrarily — by design:
the first test proves the *mechanics* (shapes, masking, determinism), not skill (cf. A5's honest
testing boundary).

``with_splits`` mirrors ``TabularAgent``: off = no-split Problem A (3 features, 3 actions); on
appends the ``can_split`` state feature (it pins the pair rank, A11) and the ``split`` action.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch
from torch import nn

from simulator.game_state import Action, GameState
from strategies.base import Strategy

from blackjack_rl.state import StateLike

# Action sets mirror TabularAgent so the two agents share one action space (and the evaluator /
# policy-diff treat them identically). Split is offered only with_splits; surrender is off in
# problem_a_config. double/split are only ever *selected* when the state actually allows them
# (handled by masking, below).
_STAGE2_ACTIONS: tuple[Action, ...] = ("hit", "stand", "double")
_SPLIT_ACTIONS: tuple[Action, ...] = ("hit", "stand", "double", "split")

_NEG_INF = float("-inf")


# One-hot bin ranges: player totals 4..21 (18 bins), dealer upcards 2..11 (10 bins, 11 = ace).
_TOTAL_LO, _TOTAL_HI = 4, 21
_UPCARD_LO, _UPCARD_HI = 2, 11
ENCODINGS: tuple[str, ...] = ("scalar", "onehot")


def feature_dim(encoding: str = "scalar", with_splits: bool = False) -> int:
    """Input width for ``encoding`` — so the agent and network agree without hardcoding."""
    if encoding == "scalar":
        base = 3
    elif encoding == "onehot":
        base = (_TOTAL_HI - _TOTAL_LO + 1) + 1 + (_UPCARD_HI - _UPCARD_LO + 1)  # total + soft + upcard
    else:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")
    return base + (1 if with_splits else 0)


def _onehot(value: int, lo: int, hi: int) -> list[float]:
    """A one-hot block over [lo, hi], clamped to range."""
    vec = [0.0] * (hi - lo + 1)
    vec[min(max(value, lo), hi) - lo] = 1.0
    return vec


def encode_features(
    state: StateLike, with_splits: bool = False, encoding: str = "scalar"
) -> list[float]:
    """State features for the network — the *same* information either way, but a different prior
    about distance (CONCEPTS.md sections 18-19, 21):

    - **"scalar"**: player_value / soft / dealer_upcard as normalized numbers. A *smoothness prior*
      — nearby totals must behave similarly. Generalizes, but smears the sharp soft-hand / upcard
      boundaries where the optimal action flips between neighbours.
    - **"onehot"**: player_value and dealer_upcard as one-hot blocks (categorical — each value its
      own input, free to differ sharply), with soft kept as a 0/1 flag. Sharp where blackjack is
      sharp; gives up the smooth-ordering generalization.

    With splits, append ``can_split`` as a flag either way (it pins the pair; A11).
    """
    if encoding == "scalar":
        feats = [
            (state.player_value - 4) / 17.0,
            float(state.player_is_soft),
            (state.dealer_upcard - 2) / 9.0,
        ]
    elif encoding == "onehot":
        feats = _onehot(state.player_value, _TOTAL_LO, _TOTAL_HI)
        feats.append(float(state.player_is_soft))
        feats += _onehot(state.dealer_upcard, _UPCARD_LO, _UPCARD_HI)
    else:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")
    if with_splits:
        feats.append(float(state.can_split))
    return feats


class QNetwork(nn.Module):
    """A small multilayer perceptron: feature vector -> one Q-value per action.

    Hidden-layer sizes are a constructor argument so the shape stays configurable and cheap to
    change (the D9 habit). Default is the smallest thing that should plausibly learn Problem A.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102 (nn.Module convention)
        return self.net(x)


class DQNAgent(Strategy):
    """A Q-network policy behind the Strategy contract — interchangeable with ``TabularAgent``.

    ``decide`` is the epsilon-greedy behaviour policy; ``greedy_action`` is the argmax target used
    for evaluation. Both choose only among *legal* actions: illegal actions are masked to -inf
    before the argmax, so the network's raw outputs can never produce an illegal move. (No random
    tie-break is needed: with continuous Q-values exact ties are essentially impossible, so the
    argmax is deterministic — which also makes the greedy policy reproducible from the weights.)

    Determinism note: this constructor does *not* seed any RNG. Weight initialization draws from
    torch's global RNG, so a caller wanting reproducibility seeds it (``torch.manual_seed``) before
    constructing. Training will own RNG state explicitly (A7); the agent stays side-effect-free.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        with_splits: bool = False,
        hidden: Sequence[int] = (64, 64),
        encoding: str = "scalar",
    ) -> None:
        self.epsilon = epsilon
        self.with_splits = with_splits
        self.encoding = encoding
        self._actions: tuple[Action, ...] = _SPLIT_ACTIONS if with_splits else _STAGE2_ACTIONS
        self._action_index: dict[Action, int] = {a: i for i, a in enumerate(self._actions)}
        self.q_net = QNetwork(feature_dim(encoding, with_splits), len(self._actions), hidden)
        # (value, soft, upcard, action) -> training experience count; filled by the trainer
        # (instrumentation: lets us ask which action a cell was actually *tested* on).
        self.sample_counts: dict = {}

    @property
    def actions(self) -> tuple[Action, ...]:
        """The agent's action space, in output order — the trainer aligns masks to this."""
        return self._actions

    # --- Strategy contract (training behaviour policy) -----------------------
    def decide(self, state: GameState) -> Action:
        """Epsilon-greedy over the legal actions (incl. split iff with_splits)."""
        legal = self._legal_actions(state)
        if random.random() < self.epsilon:
            return random.choice(legal)
        return self._greedy_over(state, legal)

    def name(self) -> str:
        return "DQNAgent"

    # --- target policy (used for evaluation) ---------------------------------
    def greedy_action(self, state: GameState) -> Action:
        """Pure argmax over the legal actions — no exploration."""
        return self._greedy_over(state, self._legal_actions(state))

    # --- helpers -------------------------------------------------------------
    def q_values(self, state: GameState) -> torch.Tensor:
        """Raw Q(state, .) for every action in ``self._actions``, unmasked. Inference only
        (no grad) — the training loop calls ``self.q_net`` directly when it needs gradients."""
        x = torch.tensor(encode_features(state, self.with_splits, self.encoding), dtype=torch.float32)
        with torch.no_grad():
            return self.q_net(x)

    def _legal_actions(self, state: GameState) -> list[Action]:
        legal: list[Action] = [a for a in state.legal_actions() if a in self._actions]
        return legal or ["stand"]  # hit/stand are always legal; guard the empty case anyway

    def _greedy_over(self, state: GameState, legal: list[Action]) -> Action:
        q = self.q_values(state)
        # Mask: every illegal action -> -inf, so the argmax can only land on a legal one.
        masked = torch.full_like(q, _NEG_INF)
        legal_idx = [self._action_index[a] for a in legal]
        masked[legal_idx] = q[legal_idx]
        return self._actions[int(torch.argmax(masked).item())]
