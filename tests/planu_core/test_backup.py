import numpy as np
import pytest

from planu_core.backup import backup_trajectory, suffix_returns
from planu_core.config import PlanUConfig
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode


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
