"""Deep Q-learning trainer (CONCEPTS.md section 17).

This unit ships the second stabilizer — the **target network** — and its helpers. The TD training
loop (capture episodes, reconstruct transitions, sample minibatches, optimize) is the next unit
and will live here too.

Why a target network: a TD target ``y = r + gamma * max_a' Q(s', a')`` computed with the *same*
weights being optimized makes the goal move every gradient step — the network chases its own
estimate and can oscillate or diverge. Computing targets from a *frozen* copy, refreshed only
every C steps, gives the online network a stationary goal to descend toward between refreshes.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from simulator.game_state import Action

from blackjack_rl.agents.dqn import QNetwork, encode_features
from blackjack_rl.env import CapturedHand, Step
from blackjack_rl.training.replay import Batch, Transition


def make_target(online: QNetwork) -> QNetwork:
    """Return a frozen copy of ``online`` for computing TD targets: identical architecture and
    weights, set to eval mode, with gradients disabled (it is never optimized — only periodically
    synced via :func:`sync_target`)."""
    target = copy.deepcopy(online)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def sync_target(target: QNetwork, online: QNetwork) -> None:
    """Hard update — copy the online weights into the target. Called every C gradient steps; in
    between, the target stays fixed so the optimization goal is stationary."""
    target.load_state_dict(online.state_dict())


# --- transition reconstruction (episode -> TD transitions) -------------------

def _legal_mask(step: Step, actions: Sequence[Action]) -> torch.Tensor:
    """Boolean legal-action mask aligned to ``actions``. Hit/stand are always legal at a recorded
    decision; double iff ``can_double``; split iff ``can_split`` (used only in split mode)."""
    flags = {"hit": True, "stand": True, "double": step.can_double, "split": step.can_split}
    return torch.tensor([flags[a] for a in actions], dtype=torch.bool)


def hand_to_transitions(
    hand: CapturedHand, actions: Sequence[Action], with_splits: bool = False
) -> list[Transition]:
    """Reconstruct TD transitions from a captured **no-split** hand (a single decision chain).

    Within the chain, a non-final decision gets reward 0, ``done=False``, and ``s'`` = the next
    decision (carrying the next state's legal mask, used to max over *legal* next actions in the
    TD target). The final decision gets reward = the hand payout and ``done=True`` (the next_*
    fields are unused placeholders). ``gamma = 1``, so no discount appears here.

    NO-SPLIT ONLY: assumes the steps form one chain. Splitting (a tree of sub-hands) is a later
    extension that must decide how the ``split`` action's two successors form its target.
    """
    action_index = {a: i for i, a in enumerate(actions)}
    n = len(actions)
    transitions: list[Transition] = []
    last = len(hand.steps) - 1
    for i, step in enumerate(hand.steps):
        state = torch.tensor(encode_features(step, with_splits), dtype=torch.float32)
        if i == last:  # terminal decision: carries the payout, no bootstrap
            transitions.append(
                Transition(
                    state=state,
                    action=action_index[step.action],
                    reward=hand.reward,
                    next_state=torch.zeros_like(state),
                    done=True,
                    next_legal_mask=torch.zeros(n, dtype=torch.bool),
                )
            )
        else:  # intermediate decision: reward 0, bootstrap from the next state
            nxt = hand.steps[i + 1]
            transitions.append(
                Transition(
                    state=state,
                    action=action_index[step.action],
                    reward=0.0,
                    next_state=torch.tensor(encode_features(nxt, with_splits), dtype=torch.float32),
                    done=False,
                    next_legal_mask=_legal_mask(nxt, actions),
                )
            )
    return transitions


# --- the TD update (one gradient step of deep Q-learning) --------------------

def td_target(target: QNetwork, batch: Batch, gamma: float = 1.0) -> torch.Tensor:
    """The TD target y for each transition: ``r`` for terminal steps, else
    ``r + gamma * max_{a' legal} Q_target(s', a')``.

    Illegal next actions are masked to -inf before the max so the bootstrap never leaks the value
    of an unavailable action, and the bootstrap is zeroed on terminal steps. ``torch.where`` (not a
    ``* (1 - done)`` multiply) selects the zero, so the terminal rows' -inf max is never multiplied
    — avoiding a ``-inf * 0 = NaN``. No gradient flows through the target net.
    """
    with torch.no_grad():
        q_next = target(batch.next_states)                                   # [B, n_actions]
        masked = q_next.masked_fill(~batch.next_legal_masks, float("-inf"))
        max_next = masked.max(dim=1).values                                  # [B]
        bootstrap = torch.where(batch.dones, torch.zeros_like(max_next), max_next)
        return batch.rewards + gamma * bootstrap


def td_update(
    online: QNetwork,
    target: QNetwork,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    gamma: float = 1.0,
) -> float:
    """One gradient step of deep Q-learning on ``batch``: Huber (smooth-L1) loss between the online
    ``Q(s, a)`` of the taken actions and the TD target from the frozen target net. Returns the loss
    value (for the learning curve). Gradients flow only through the online net."""
    q_taken = online(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # [B]
    y = td_target(target, batch, gamma)
    loss = F.smooth_l1_loss(q_taken, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
