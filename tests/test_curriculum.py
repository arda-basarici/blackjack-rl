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
