"""Tests for the thermometer (cumulative/unary) encoding — between scalar and one-hot."""
from __future__ import annotations

from blackjack_rl.dqn.agent import _thermometer, encode_features, feature_dim
from blackjack_rl.core.config import DQNConfig
from blackjack_rl.core.env import Step


def test_thermometer_block_is_cumulative() -> None:
    # range [4, 21]; value 4 -> only first bit; value 21 -> all bits; value 13 -> 10 leading ones
    assert _thermometer(4, 4, 21) == [1.0] + [0.0] * 17
    assert _thermometer(21, 4, 21) == [1.0] * 18
    v13 = _thermometer(13, 4, 21)
    assert v13 == [1.0] * 10 + [0.0] * 8
    # monotone non-increasing (a real thermometer: ones then zeros)
    assert all(v13[i] >= v13[i + 1] for i in range(len(v13) - 1))
    # clamped
    assert _thermometer(99, 4, 21) == [1.0] * 18
    assert _thermometer(1, 4, 21) == [1.0] + [0.0] * 17


def test_thermometer_feature_dim_matches_onehot() -> None:
    assert feature_dim("thermometer") == feature_dim("onehot")
    assert feature_dim("thermometer", with_splits=True) == feature_dim("onehot", with_splits=True)


def test_encode_features_thermometer_shape_and_values() -> None:
    st = Step(player_value=13, player_is_soft=False, dealer_upcard=6,
              can_split=False, can_double=True, action="hit")
    f = encode_features(st, encoding="thermometer")
    assert len(f) == feature_dim("thermometer")
    # neighbours share most bits (the generalization property): 12 vs 13 differ in one position
    st12 = Step(player_value=12, player_is_soft=False, dealer_upcard=6,
                can_split=False, can_double=True, action="hit")
    f12 = encode_features(st12, encoding="thermometer")
    assert sum(a != b for a, b in zip(f, f12)) == 1


def test_config_accepts_thermometer() -> None:
    DQNConfig(num_episodes=10, encoding="thermometer")
    try:
        DQNConfig(num_episodes=10, encoding="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for encoding='bogus'")
