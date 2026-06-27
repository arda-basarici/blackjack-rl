"""Tests for blackjack_rl.session.references — the Problem-B ground-truth references (B1).

Three kinds of check:
- ``edge_by_count`` — contract + determinism + a light statistical sanity (pooled edge near zero,
  per-bucket variance ~ blackjack's ~1.3). Not a precise edge-curve measurement: that needs the
  millions-of-hands runner, out of scope for a fast test.
- ``kelly_bet_curve`` — pure unit checks against hand-built ``CountEdge`` inputs (no engine).
- ``index_plays`` — the literature table's shape, semantics, and a few spot indices.
"""
import pytest

from blackjack_rl.session.references import (
    CountAccumulator,
    CountEdge,
    IndexPlay,
    IndexPlayTable,
    accumulate_edges,
    edge_by_count,
    index_plays,
    kelly_bet_curve,
)

VALID_ACTIONS = {"hit", "stand", "double", "split", "surrender", "none"}


# --- edge_by_count -------------------------------------------------------------------

def test_edge_by_count_contract_and_buckets():
    edges = edge_by_count(n_hands=8000, seed=0)

    assert edges, "expected at least a few populated count buckets"
    assert all(isinstance(tc, int) for tc in edges)
    assert list(edges) == sorted(edges)  # keyed and ordered by true count
    for tc, e in edges.items():
        assert isinstance(e, CountEdge)
        assert e.true_count == tc
        assert e.n >= 2  # single-hand buckets are dropped (variance needs two)
        assert e.variance >= 0  # can be exactly 0 in tiny extreme buckets (identical outcomes)
        assert e.std_error == pytest.approx((e.variance / e.n) ** 0.5)

    # No hands are invented; a few extreme single-hand buckets may be dropped.
    total = sum(e.n for e in edges.values())
    assert 8000 - 50 <= total <= 8000


def test_edge_by_count_statistical_sanity():
    edges = edge_by_count(n_hands=12000, seed=1)

    # Pooled per-hand return sits near zero (a small house edge), not garbage.
    pooled = sum(e.mean_return * e.n for e in edges.values()) / sum(e.n for e in edges.values())
    assert abs(pooled) < 0.05

    # The dominant near-zero-count buckets carry blackjack's ~1.3 outcome variance.
    for tc in (0, 1, -1):
        if tc in edges and edges[tc].n >= 1000:
            assert 0.8 < edges[tc].variance < 2.0


def test_edge_by_count_is_deterministic():
    a = edge_by_count(n_hands=5000, seed=7)
    b = edge_by_count(n_hands=5000, seed=7)
    assert a == b


def test_edge_by_count_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        edge_by_count(n_hands=0)


# --- CountAccumulator (the mergeable primitive behind the parallel B2a runner) -------

def test_accumulator_merge_equals_single_stream():
    """The load-bearing invariant: merging two partials gives the *same* moments as folding every
    value into one accumulator (Chan's parallel variance is exact). This is what lets the B2a runner
    fan out across cores without changing the measured curve. Buckets overlap and differ in size."""
    left_values = {0: [1.0, -1.0, 2.0, 0.0], 1: [3.0, -2.0], 5: [1.5]}
    right_values = {0: [0.5, -0.5], 1: [4.0, 1.0, -3.0], 7: [2.0, 2.0]}

    left, right, single = CountAccumulator(), CountAccumulator(), CountAccumulator()
    for tc, vals in left_values.items():
        for v in vals:
            left.add(tc, v)
            single.add(tc, v)
    for tc, vals in right_values.items():
        for v in vals:
            right.add(tc, v)
            single.add(tc, v)

    merged_edges, single_edges = left.merge(right).edges(), single.edges()
    assert merged_edges.keys() == single_edges.keys()
    for tc, m in merged_edges.items():  # algebraically exact -> float-close (summation order differs)
        s = single_edges[tc]
        assert m.n == s.n
        assert m.mean_return == pytest.approx(s.mean_return)
        assert m.variance == pytest.approx(s.variance)
    assert left.merge(right).n_total == single.n_total == 14
    assert left.merge(right).pooled_mean == pytest.approx(single.pooled_mean)


def test_accumulator_merge_is_pure():
    """merge returns a new accumulator and leaves both operands untouched."""
    a, b = CountAccumulator(), CountAccumulator()
    a.add(0, 1.0)
    b.add(0, 2.0)
    a.merge(b)
    assert a.buckets == {0: [1.0, 1.0, 0.0]}  # unchanged
    assert b.buckets == {0: [1.0, 2.0, 0.0]}  # unchanged


def test_accumulate_edges_finalizes_to_edge_by_count():
    """edge_by_count is exactly accumulate_edges(...).edges() — the shared-core refactor holds."""
    acc = accumulate_edges(n_hands=4000, seed=3)
    assert acc.edges() == edge_by_count(n_hands=4000, seed=3)


# --- kelly_bet_curve -----------------------------------------------------------------

def _edge(tc: int, mean: float, var: float) -> CountEdge:
    return CountEdge(true_count=tc, mean_return=mean, variance=var, std_error=0.0, n=100)


def test_kelly_curve_is_mean_over_variance_floored_at_zero():
    edges = {
        -2: _edge(-2, -0.02, 1.3),  # negative edge -> bet nothing
        0: _edge(0, 0.0, 1.3),      # zero edge -> bet nothing
        3: _edge(3, 0.013, 1.3),    # favorable -> positive Kelly fraction
    }
    curve = kelly_bet_curve(edges)

    assert curve[-2] == 0.0
    assert curve[0] == 0.0
    assert curve[3] == pytest.approx(0.013 / 1.3)
    assert list(curve) == sorted(curve)  # ordered by true count


def test_kelly_curve_handles_zero_variance():
    curve = kelly_bet_curve({0: _edge(0, 0.01, 0.0)})
    assert curve[0] == 0.0  # no divide-by-zero; degenerate bucket bets nothing


def test_kelly_curve_increases_with_edge():
    edges = {1: _edge(1, 0.005, 1.3), 5: _edge(5, 0.025, 1.3)}
    curve = kelly_bet_curve(edges)
    assert curve[5] > curve[1] > 0.0


# --- index_plays ---------------------------------------------------------------------

def test_index_plays_shape_and_actions():
    table = index_plays()
    assert isinstance(table, IndexPlayTable)
    assert len(table.playing) == 17   # Illustrious 18 minus insurance (not an Action)
    assert len(table.surrender) == 4  # Fab 4

    for play in (*table.playing, *table.surrender):
        assert isinstance(play, IndexPlay)
        assert play.action_below in VALID_ACTIONS
        assert play.action_at_or_above in VALID_ACTIONS
        assert 2 <= play.dealer_upcard <= 11

    # surrenders all deviate to surrender above their index
    assert all(p.action_at_or_above == "surrender" for p in table.surrender)


def test_index_plays_spot_values_and_semantics():
    playing = {p.label: p for p in index_plays().playing}

    # 16 v 10: hit below 0, stand at/above 0 — the most famous deviation.
    p = playing["16 v 10"]
    assert (p.index, p.action_below, p.action_at_or_above) == (0.0, "hit", "stand")

    # T,T v 5 is a pair, split only at a high count.
    p = playing["T,T v 5"]
    assert p.is_pair and p.player_total == 20 and p.index == 5.0
    assert (p.action_below, p.action_at_or_above) == ("stand", "split")


def test_15v10_appears_in_both_groups_with_distinct_indices():
    table = index_plays()
    playing_15v10 = next(p for p in table.playing if p.label == "15 v 10")
    surrender_15v10 = next(p for p in table.surrender if p.label == "15 v 10")
    # surrender (index 0) takes precedence when available; otherwise stand at +4
    assert surrender_15v10.index == 0.0
    assert playing_15v10.index == 4.0
