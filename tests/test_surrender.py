"""Tests for the surrender action (flag-gated, terminal, first-action-only)."""
from __future__ import annotations

from blackjack_rl.dqn.agent import DQNAgent
from blackjack_rl.config import DQNConfig
from blackjack_rl.env import CapturedHand, Step, problem_a_config
from blackjack_rl.dqn.deep_q import _legal_mask, hand_to_transitions


def test_action_set_includes_surrender_when_flagged() -> None:
    assert DQNAgent(with_surrender=False).actions == ("hit", "stand", "double")
    assert DQNAgent(with_surrender=True).actions == ("hit", "stand", "double", "surrender")
    # combined, order preserved
    assert DQNAgent(with_splits=True, with_surrender=True).actions == (
        "hit", "stand", "double", "split", "surrender")


def test_problem_config_surrender_flag() -> None:
    assert problem_a_config().surrender_allowed is False           # default unchanged
    assert problem_a_config(with_surrender=True).surrender_allowed is True


def test_surrender_never_legal_as_next_action() -> None:
    actions = ("hit", "stand", "double", "surrender")
    step = Step(player_value=16, player_is_soft=False, dealer_upcard=10, can_split=False,
                can_double=True, action="hit", can_surrender=True)
    mask = _legal_mask(step, actions)
    assert mask[actions.index("surrender")].item() is False  # surrender is first-action-only


def test_surrender_terminal_skips_reward_baseline() -> None:
    agent = DQNAgent(with_surrender=True, encoding="onehot")
    step = Step(player_value=16, player_is_soft=False, dealer_upcard=10, can_split=False,
                can_double=True, action="surrender", final_dealer_value=0, can_surrender=True)
    hand = CapturedHand(steps=[step], reward=-0.5)
    # even with the stand baseline on, a surrender terminal keeps its raw -0.5 (no dealer played)
    t = hand_to_transitions(hand, agent.actions, encoding="onehot", reward_baseline="stand")
    assert abs(float(t[0].reward) - (-0.5)) < 1e-6


def test_config_accepts_with_surrender() -> None:
    assert DQNConfig(num_episodes=10).with_surrender is False
    assert DQNConfig(num_episodes=10, with_surrender=True).with_surrender is True


def test_surrender_run_completes(tmp_path) -> None:
    from blackjack_rl.dqn.experiment import run_dqn
    cfg = DQNConfig(num_episodes=300, with_surrender=True, warmup=10, batch_size=8,
                    buffer_capacity=500, encoding="onehot", seed=0)
    res = run_dqn(cfg, eval_hands=200, runs_dir=tmp_path, progress_every=None, save=False)
    assert "surrender" in res.agent.actions  # full action set trained + evaluated end to end


def test_diff_scores_surrender_only_when_agent_plays_it() -> None:
    """The diff offers surrender as an option iff the agent plays it, so the surrender cells (hard 16
    vs 9/10/A, hard 15 vs 10) are scored — not silently dropped as they were before."""
    from blackjack_rl.dqn.network_diff import diff_network

    with_s = diff_network(DQNAgent(epsilon=0.0, with_surrender=True, encoding="onehot"))
    assert any(c.basic_action == "surrender" for c in with_s.cells)   # basic surrenders some cell
    without_s = diff_network(DQNAgent(epsilon=0.0, with_surrender=False, encoding="onehot"))
    assert all(c.basic_action != "surrender" for c in without_s.cells)  # off -> never offered
