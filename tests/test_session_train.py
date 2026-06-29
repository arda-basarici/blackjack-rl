"""Tests for the Problem-B bet-model trainer (B2d): ``train_bet`` + ``BetTrainConfig``.

A smoke trainer, not a skill test (cf. A5's honest boundary): the runs are tiny, so we assert the
*mechanics* — reproducibility from the seed, a sane learning curve, and that the optimizer actually
moved the weights — not that the agent learned to bet well (that's B2d-3's measured comparison).
"""
import math

import pytest
import torch

from blackjack_rl.session.bet_agent import BetAgent
from blackjack_rl.session.env import SessionConfig
from blackjack_rl.session.train import _PROBE_COUNTS, BetTrainConfig, train_bet


def _tiny_config(seed: int = 0) -> BetTrainConfig:
    """A fast config that still crosses warm-up so gradient steps actually fire."""
    return BetTrainConfig(
        session=SessionConfig(starting_bankroll=400.0, max_hands=20, seed=99),
        n_sessions=60, warmup=50, batch_size=16, buffer_capacity=2_000,
        train_every=1, target_sync_every=10, seed=seed,
    )


def _params_equal(a: BetAgent, b: BetAgent) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.q_net.parameters(), b.q_net.parameters()))


def test_bet_train_config_rejects_bad_n_sessions():
    with pytest.raises(ValueError):
        BetTrainConfig(n_sessions=0)


def test_train_bet_returns_agent_over_session_spread():
    cfg = _tiny_config()
    agent = train_bet(cfg)
    assert isinstance(agent, BetAgent)
    assert agent.levels == tuple(float(x) for x in cfg.session.bet_spread)


def test_train_bet_reproducible_from_seed():
    """Same seed -> identical weights -> identical policy (reproducible by construction)."""
    assert _params_equal(train_bet(_tiny_config(seed=1)), train_bet(_tiny_config(seed=1)))


def test_train_bet_updates_weights():
    """The optimizer moves the weights off their (same-seed) initialization."""
    cfg = _tiny_config(seed=2)
    torch.manual_seed(cfg.seed)
    fresh = BetAgent(levels=cfg.session.bet_spread, hidden=cfg.hidden, bankroll_scale=cfg.bankroll_scale)
    trained = train_bet(cfg)
    assert not _params_equal(fresh, trained)


def test_train_bet_emits_learning_curve():
    cps: list[dict] = []
    train_bet(_tiny_config(), progress_every=20, on_checkpoint=cps.append)
    assert len(cps) == 3  # 60 sessions / 20
    last = cps[-1]
    assert set(last) >= {"session", "epsilon", "lr", "grad_steps", "buffer", "recent_loss", "bet_by_count"}
    assert last["grad_steps"] > 0  # crossed warm-up
    assert last["recent_loss"] is not None and math.isfinite(last["recent_loss"])
    assert set(last["bet_by_count"]) == set(_PROBE_COUNTS)  # the probe curve spans the count range
