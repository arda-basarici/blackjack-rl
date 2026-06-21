"""Tests for curriculum learning: the double gate (selection) and the bootstrap-max mask, plus
the double_after config + an end-to-end stage-one-only run that never doubles."""
from __future__ import annotations

import torch

from blackjack_rl.agents.dqn import DQNAgent, QNetwork
from blackjack_rl.config import DQNConfig
from blackjack_rl.evaluation.network_diff import _state_for
from blackjack_rl.training.deep_q import td_target
from blackjack_rl.training.replay import Batch


def test_gate_removes_double_from_legal_actions() -> None:
    agent = DQNAgent(epsilon=0.0, encoding="onehot")
    st = _state_for(11, False, 6)  # can_double = True here
    agent.double_enabled = True
    assert "double" in agent._legal_actions(st)
    agent.double_enabled = False
    assert "double" not in agent._legal_actions(st)


def test_gated_greedy_skips_double_even_when_best() -> None:
    agent = DQNAgent(epsilon=0.0, encoding="onehot")
    st = _state_for(11, False, 6)
    di = agent.actions.index("double")
    q = torch.full((len(agent.actions),), -1.0)
    q[di] = 5.0  # make double look clearly best
    agent.q_values = lambda s: q  # type: ignore[method-assign]
    agent.double_enabled = True
    assert agent.greedy_action(st) == "double"
    agent.double_enabled = False
    assert agent.greedy_action(st) != "double"  # gate overrides the high Q


def test_td_target_mask_action_never_increases_the_max() -> None:
    torch.manual_seed(0)
    net = QNetwork(in_dim=4, out_dim=3, hidden=(8,))
    b = Batch(
        states=torch.zeros(1, 4), actions=torch.tensor([0]), rewards=torch.tensor([0.0]),
        next_states=torch.randn(1, 4), dones=torch.tensor([False]),
        next_legal_masks=torch.ones(1, 3, dtype=torch.bool),
    )
    full = td_target(net, b, gamma=1.0)
    for a in range(3):  # removing any candidate from the max can only lower (or keep) it
        assert td_target(net, b, gamma=1.0, mask_action=a).item() <= full.item() + 1e-6


def test_double_after_config_validation() -> None:
    assert DQNConfig(num_episodes=10).double_after == 0
    DQNConfig(num_episodes=10, double_after=5)
    try:
        DQNConfig(num_episodes=10, double_after=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for double_after=-1")


def test_stage_one_only_run_never_doubles(tmp_path) -> None:
    from blackjack_rl.dqn_experiment import run_dqn

    cfg = DQNConfig(num_episodes=300, double_after=10_000, warmup=10, batch_size=8,
                    buffer_capacity=500, encoding="onehot", seed=0)
    res = run_dqn(cfg, eval_hands=200, runs_dir=tmp_path, progress_every=None, save=False)
    actions = {k[3] for k in res.agent.sample_counts}
    assert "double" not in actions  # double never enabled -> never taken


def test_replay_buffer_clear_resets_and_is_reusable() -> None:
    from blackjack_rl.training.replay import ReplayBuffer, Transition

    def _t() -> Transition:
        return Transition(state=torch.zeros(4), action=0, reward=0.0, next_state=torch.zeros(4),
                          done=True, next_legal_mask=torch.ones(3, dtype=torch.bool))
    buf = ReplayBuffer(capacity=10)
    for _ in range(5):
        buf.push(_t())
    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0 and buf._pos == 0
    buf.push(_t())                       # still usable after clearing
    assert len(buf) == 1


def test_clear_buffer_on_double_config_validation() -> None:
    assert DQNConfig(num_episodes=100).clear_buffer_on_double is False
    DQNConfig(num_episodes=100, double_after=50, clear_buffer_on_double=True)  # ok with a switch point
    try:
        DQNConfig(num_episodes=100, double_after=0, clear_buffer_on_double=True)  # no switch to clear at
    except ValueError:
        return
    raise AssertionError("expected ValueError for clear_buffer_on_double without double_after")


def test_clear_buffer_on_double_drops_backlog() -> None:
    """At the double switch the buffer is flushed, so shortly after it carries far fewer
    transitions than the un-cleared run that keeps its whole hit/stand backlog."""
    from blackjack_rl.training.deep_q import train_dqn

    def buffer_after_switch(clear: bool) -> int:
        curve: list[dict] = []
        cfg = DQNConfig(num_episodes=400, double_after=200, clear_buffer_on_double=clear,
                        warmup=10, batch_size=8, buffer_capacity=10_000, encoding="onehot", seed=0)
        train_dqn(cfg, progress_every=100, on_checkpoint=curve.append)
        return next(cp["buffer"] for cp in curve if cp["episode"] == 300)  # 100 eps after the switch

    assert buffer_after_switch(clear=True) < buffer_after_switch(clear=False)
