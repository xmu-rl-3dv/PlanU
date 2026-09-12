import numpy as np
import pytest

from planu_core.adapters.webshop import WebShopAdapter
from planu_core.backup import backup_trajectory, suffix_returns
from planu_core.config import PlanUConfig
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.search import PlanUSearch
from tests.planu_core.fakes import UniformScorer
from tests.planu_core.webshop_fakes import (
    DeterministicWebShopActionProvider,
    FakeWebShopClient,
)


def test_backup_functions_are_exported_from_package():
    from planu_core import backup_trajectory as exported_backup_trajectory
    from planu_core import suffix_returns as exported_suffix_returns

    assert exported_backup_trajectory is backup_trajectory
    assert exported_suffix_returns is suffix_returns


def make_action_node(key: str = "action") -> ActionNode:
    root = LanguageNode(state=[0], state_key=(0,))
    return ActionNode(
        parent=root,
        action=ActionCandidate(key, key, key),
        distribution=QuantileDistribution.from_scalar(
            0.0,
            count=3,
            value_min=-2.0,
            value_max=2.0,
        ),
    )


def snapshot(action: ActionNode):
    return (
        action.visit_count,
        list(action.cumulative_returns),
        action.distribution.values.copy(),
    )


def assert_unchanged(action: ActionNode, before) -> None:
    visit_count, cumulative_returns, distribution_values = before
    assert action.visit_count == visit_count
    assert action.cumulative_returns == cumulative_returns
    np.testing.assert_array_equal(action.distribution.values, distribution_values)


def test_suffix_returns_match_reference_undiscounted_sum():
    assert suffix_returns([0.1, -0.1, 1.0], 1.0) == pytest.approx(
        [1.0, 0.9, 1.0]
    )


def test_backup_updates_each_selected_action_once():
    first = make_action_node("first")
    second = make_action_node("second")
    config = PlanUConfig(
        n_quantiles=3,
        value_min=-2.0,
        value_max=2.0,
        quantile_learning_rate=0.5,
    )

    returns = backup_trajectory(iter([first, second]), [0.25, 1.0], config)

    assert returns == [1.25, 1.0]
    assert first.cumulative_returns == [1.25]
    assert second.cumulative_returns == [1.0]
    assert first.visit_count == 1
    assert second.visit_count == 1
    np.testing.assert_allclose(
        first.distribution.values,
        [1.25 / 12.0, 1.25 / 4.0, 1.25 * 5.0 / 12.0],
    )
    np.testing.assert_allclose(
        second.distribution.values,
        [1.0 / 12.0, 1.0 / 4.0, 5.0 / 12.0],
    )


def test_backup_rolls_back_every_action_when_later_distribution_update_fails(
    monkeypatch,
):
    first = make_action_node("first")
    second = make_action_node("second")
    first.visit_count = 3
    first.cumulative_returns[:] = [-0.5]
    first.distribution.values[:] = [-1.0, 0.25, 0.75]
    second.visit_count = 7
    second.cumulative_returns[:] = [0.5, 1.0]
    second.distribution.values[:] = [-0.75, 0.0, 1.25]
    before = [snapshot(first), snapshot(second)]
    return_lists = [first.cumulative_returns, second.cumulative_returns]
    value_arrays = [first.distribution.values, second.distribution.values]
    sentinel = RuntimeError("sentinel update failure")

    def fail_update(target, learning_rate):
        raise sentinel

    monkeypatch.setattr(second.distribution, "update_scalar", fail_update)

    with pytest.raises(RuntimeError, match="sentinel update failure") as raised:
        backup_trajectory([first, second], [0.25, 1.0], PlanUConfig())

    assert raised.value is sentinel
    for action, action_before, return_list, value_array in zip(
        [first, second], before, return_lists, value_arrays
    ):
        assert_unchanged(action, action_before)
        assert action.cumulative_returns is return_list
        assert action.distribution.values is value_array


def test_backup_snapshots_repeated_action_once_before_any_update(monkeypatch):
    action = make_action_node()
    action.visit_count = 4
    action.cumulative_returns[:] = [-1.0, 0.25]
    action.distribution.values[:] = [-1.5, -0.25, 1.5]
    before = snapshot(action)
    return_list = action.cumulative_returns
    value_array = action.distribution.values
    original_update = action.distribution.update_scalar
    sentinel = RuntimeError("sentinel repeated update failure")
    update_calls = 0

    def fail_second_update(target, learning_rate):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise sentinel
        original_update(target, learning_rate)

    monkeypatch.setattr(
        action.distribution,
        "update_scalar",
        fail_second_update,
    )

    with pytest.raises(
        RuntimeError,
        match="sentinel repeated update failure",
    ) as raised:
        backup_trajectory([action, action], [0.5, 1.0], PlanUConfig())

    assert raised.value is sentinel
    assert update_calls == 2
    assert_unchanged(action, before)
    assert action.cumulative_returns is return_list
    assert action.distribution.values is value_array


def test_suffix_returns_apply_discount():
    assert suffix_returns([0.25, 1.0], 0.5) == [0.75, 1.0]


def test_backup_rejects_length_mismatch_without_mutation():
    first = make_action_node("first")
    second = make_action_node("second")
    before = [snapshot(first), snapshot(second)]

    with pytest.raises(ValueError, match="length"):
        backup_trajectory([first, second], [1.0], PlanUConfig())

    assert_unchanged(first, before[0])
    assert_unchanged(second, before[1])


@pytest.mark.parametrize("reward", [float("-inf"), float("inf"), float("nan")])
def test_backup_rejects_nonfinite_reward_without_mutation(reward):
    first = make_action_node("first")
    second = make_action_node("second")
    before = [snapshot(first), snapshot(second)]

    with pytest.raises(ValueError, match="reward.*finite"):
        backup_trajectory([first, second], [0.25, reward], PlanUConfig())

    assert_unchanged(first, before[0])
    assert_unchanged(second, before[1])


def test_backup_updates_repeated_action_for_each_occurrence():
    action = make_action_node()
    config = PlanUConfig(
        n_quantiles=3,
        value_min=-2.0,
        value_max=2.0,
        quantile_learning_rate=0.5,
    )

    returns = backup_trajectory([action, action], [0.5, 1.0], config)

    assert returns == [1.5, 1.0]
    assert action.visit_count == 2
    assert action.cumulative_returns == [1.5, 1.0]


def test_webshop_backup_persists_success_and_latency_failure_outcomes():
    search = PlanUSearch(
        WebShopAdapter(FakeWebShopClient(), "session-outcomes"),
        UniformScorer(),
        PlanUConfig(max_depth=2),
        action_provider=DeterministicWebShopActionProvider(),
    )

    success = search.run_iteration(0, np.random.default_rng(0))
    failure = search.run_iteration(1, np.random.default_rng(3))

    search_state = success.state_path[1]
    assert list(search_state.children) == [("click", "A-1")]
    action = search_state.children[("click", "A-1")]
    outcome_keys = set(action.children)
    success_key = success.state_path[-1].state_key
    failure_key = failure.state_path[-1].state_key

    assert len(outcome_keys) >= 2
    assert success_key != failure_key
    assert {success_key, failure_key} <= outcome_keys
    assert {success_key[12], failure_key[12]} == {0, 1}
    assert success.action_path[-1] is action
    assert failure.action_path[-1] is action
    assert action.visit_count == 2
    assert action.cumulative_returns == [0.0, 0.0]


def test_empty_suffix_and_backup_return_empty_lists():
    assert suffix_returns([], 1.0) == []
    assert backup_trajectory([], [], PlanUConfig()) == []


@pytest.mark.parametrize(
    "discount",
    [-0.01, 1.01, float("-inf"), float("inf"), float("nan")],
)
def test_suffix_returns_reject_invalid_discount(discount):
    with pytest.raises(ValueError, match="discount"):
        suffix_returns([1.0], discount)


def test_backup_rejects_nonfinite_return_without_mutation():
    action = make_action_node()
    before = snapshot(action)

    with pytest.raises(ValueError, match="return.*finite"):
        backup_trajectory([action, action], [1e308, 1e308], PlanUConfig())

    assert_unchanged(action, before)
