"""Dealer-outcome control variates for the reward — strip the dealer's shared variance out of the
training signal so the high-variance actions (double) settle (CONCEPTS §26 / §27 family).

The terminal reward's swing is mostly the *dealer's* outcome, which is shared across hit/stand/double
and so is noise w.r.t. *your* decision. We subtract a mean-zero, action-independent baseline `b` from
the terminal reward — unbiased (EV preserved), leak-free (same `b` for every action, so the argmax /
policy is untouched), and correlated with the swing (so variance drops). Two forms:

- **bust**   : ``b = c * (dealer_busted - p(upcard))`` — coarse; cancels the bust/no-bust chunk only.
- **stand**  : ``b = score(start_total vs dealer_final) - V_stand(start_total, upcard)`` — the full
               dealer-total control via a fixed *stand* reference; cancels essentially all the dealer
               variance and is still leak-free (uses the pre-action total, never the realized one).

Tables are precomputed from a self-contained dealer model (infinite-deck, S17, conditioned on no
dealer blackjack — decisions only happen when the dealer didn't have one). Infinite-deck is a tiny
approximation vs the engine's 6-deck shoe; it keeps the baseline self-contained and is what makes the
cancellation clean. `V_stand` and `p_bust` are also reusable elsewhere.
"""
from __future__ import annotations

import random
from functools import lru_cache

# infinite-deck value distribution: 2..9 (1/13 each), 10/J/Q/K -> 10 (4/13), ace -> 11 (1/13)
_VALUES = list(range(2, 11)) + [10, 10, 10] + [11]  # 13 cards, value-coded
DEALER_FINALS = (0, 17, 18, 19, 20, 21)  # 0 == bust


def _hand_value(cards: list[int]) -> tuple[int, bool]:
    """(total, is_soft) with aces (coded 11) demoted from 11->1 as needed."""
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


@lru_cache(maxsize=None)
def dealer_outcome_dist(upcard: int, hits_soft_17: bool = False, n: int = 400_000, seed: int = 0) -> dict:
    """P(dealer final ∈ {bust,17..21} | upcard), excluding dealer-blackjack starts (post-peek)."""
    rng = random.Random(1000 * seed + upcard)  # int seed (tuple seeds are rejected on py3.13)
    counts = {f: 0 for f in DEALER_FINALS}
    got = 0
    while got < n:
        hole = rng.choice(_VALUES)
        if {upcard, hole} == {11, 10}:  # a natural blackjack — never faced at decision time
            continue
        cards = [upcard, hole]
        total, soft = _hand_value(cards)
        while not (total > 17 or (total == 17 and not (soft and hits_soft_17))):
            cards.append(rng.choice(_VALUES))
            total, soft = _hand_value(cards)
        counts[0 if total > 21 else total] += 1
        got += 1
    return {f: counts[f] / got for f in DEALER_FINALS}


def score(player_total: int, dealer_final: int) -> float:
    """Stand outcome at 1x stake: +1 win, 0 push, -1 loss. ``dealer_final`` 0 = bust."""
    if player_total > 21:
        return -1.0
    if dealer_final == 0:
        return 1.0
    return 1.0 if player_total > dealer_final else (-1.0 if player_total < dealer_final else 0.0)


@lru_cache(maxsize=None)
def stand_value(player_total: int, upcard: int, hits_soft_17: bool = False) -> float:
    """E[stand reward] for a player total vs an upcard — the V_stand baseline mean."""
    dist = dealer_outcome_dist(upcard, hits_soft_17)
    return sum(p * score(player_total, d) for d, p in dist.items())


@lru_cache(maxsize=None)
def bust_prob(upcard: int, hits_soft_17: bool = False) -> float:
    """P(dealer busts | upcard) — the bust-baseline mean."""
    return dealer_outcome_dist(upcard, hits_soft_17)[0]


def baseline(kind: str, *, start_total: int, upcard: int, dealer_final: int,
             c: float = 1.0, hits_soft_17: bool = False) -> float:
    """The mean-zero control `b` to subtract from a terminal reward. ``start_total`` is the player's
    total *before* the action (leak-free); ``dealer_final`` is the dealer's played-out total (0=bust).

    kind="none"  -> 0
    kind="bust"  -> c * ( busted - p(upcard) )
    kind="stand" -> score(start_total, dealer_final) - V_stand(start_total, upcard)
    """
    if kind == "none":
        return 0.0
    if kind == "bust":
        busted = 1.0 if dealer_final == 0 else 0.0
        return c * (busted - bust_prob(upcard, hits_soft_17))
    if kind == "stand":
        return score(start_total, dealer_final) - stand_value(start_total, upcard, hits_soft_17)
    raise ValueError(f"unknown baseline kind {kind!r}; expected none|bust|stand")
