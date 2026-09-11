import numpy as np
import pytest

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.selection import _stable_softmax, select_action


def make_action(parent, key, values):
    return ActionNode(
        parent=parent,
        action=ActionCandidate(key, key, str(key)),
        distribution=QuantileDistribution.from_values(values, -1.0, 1.0),
    )


def make_root(action_values):
    root = LanguageNode(state=[0], state_key=(0,))
    for key, values in action_values:
        root.children[key] = make_action(root, key, values)
    return root


def test_greedy_selection_uses_distorted_quantile_value_and_preserves_score_order():
    root = make_root(
        [
            ("steady", [0.2, 0.2, 0.2]),
            ("upside", [-1.0, -1.0, 0.9]),
        ]
    )

    selected, scores = select_action(
        root,
        PlanUConfig(risk_distortion=0.2),
        iteration=200,
        rng=np.random.default_rng(7),
    )

    assert selected.action.key == "upside"
    assert list(scores) == ["steady", "upside"]
    assert scores == {"steady": 0.2, "upside": 0.9}


def test_equal_values_greedily_choose_first_inserted_action():
    root = make_root([("first", [0.4]), ("second", [0.4])])

    selected, _ = select_action(
        root,
        PlanUConfig(),
        iteration=0,
        rng=np.random.default_rng(7),
    )

    assert selected.action.key == "first"


def test_always_sample_schedule_produces_valid_reproducible_sequence():
    root = make_root([("a", [0.0]), ("b", [0.0])])
    config = PlanUConfig(
        selection_schedule=SelectionSchedule(
            always_sample_before=20,
            probabilistic_sample_before=20,
        )
    )

    def sample_sequence(seed):
        rng = np.random.default_rng(seed)
        return [
            select_action(root, config, iteration, rng)[0].action.key
            for iteration in range(12)
        ]

    first = sample_sequence(4)
    second = sample_sequence(4)

    assert first == second
    assert set(first) == {"a", "b"}


def test_curiosity_is_l1_normalized_and_weighted_before_selection():
    root = make_root([("known", [0.6]), ("novel", [0.5])])

    selected, scores = select_action(
        root,
        PlanUConfig(curiosity_weight=0.2),
        iteration=0,
        rng=np.random.default_rng(7),
        novelty={"known": 0.0, "novel": 2.0},
    )

    assert selected.action.key == "novel"
    assert scores == {"known": 0.6, "novel": 0.7}


def test_zero_novelty_leaves_base_values_unchanged():
    root = make_root([("first", [0.25]), ("second", [-0.5])])

    _, scores = select_action(
        root,
        PlanUConfig(curiosity_weight=100.0),
        iteration=0,
        rng=np.random.default_rng(7),
        novelty={"first": 0.0, "second": 0.0},
    )

    assert scores == {"first": 0.25, "second": -0.5}


def test_unexpanded_node_is_rejected():
    root = LanguageNode(state=[0], state_key=(0,))

    with pytest.raises(ValueError, match="unexpanded"):
        select_action(root, PlanUConfig(), 0, np.random.default_rng(7))


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
def test_nonfinite_used_novelty_is_rejected(value):
    root = make_root([("action", [0.0])])

    with pytest.raises(ValueError, match="novelty.*finite"):
        select_action(
            root,
            PlanUConfig(),
            0,
            np.random.default_rng(7),
            novelty={"action": value},
        )


def test_nonfinite_unused_novelty_is_rejected():
    root = make_root([("action", [0.0])])

    with pytest.raises(ValueError, match="novelty.*finite"):
        select_action(
            root,
            PlanUConfig(),
            0,
            np.random.default_rng(7),
            novelty={"unused": float("nan")},
        )


def test_nonfinite_base_score_is_rejected():
    root = make_root([("action", [0.0])])
    root.children["action"].distribution.values[0] = np.inf

    with pytest.raises(ValueError, match="base.*finite"):
        select_action(root, PlanUConfig(), 0, np.random.default_rng(7))


def test_nonfinite_final_score_is_rejected():
    root = make_root([("action", [1e308])])

    with pytest.raises(ValueError, match="score.*finite"):
        select_action(
            root,
            PlanUConfig(curiosity_weight=1e308),
            0,
            np.random.default_rng(7),
            novelty={"action": 1.0},
        )


def test_stable_softmax_is_finite_for_huge_logits_and_sums_to_one():
    probabilities = _stable_softmax(
        np.array([1e300, 1e300, -1e300]),
        temperature=0.5,
    )

    assert np.all(np.isfinite(probabilities))
    assert np.sum(probabilities) == pytest.approx(1.0)
    np.testing.assert_allclose(probabilities, [0.5, 0.5, 0.0])


@pytest.mark.parametrize(
    ("values", "temperature"),
    [
        (np.array([]), 1.0),
        (np.array([0.0, float("nan")]), 1.0),
        (np.array([0.0, float("inf")]), 1.0),
        (np.array([0.0]), 0.0),
        (np.array([0.0]), -1.0),
        (np.array([0.0]), float("inf")),
        (np.array([0.0]), float("nan")),
    ],
)
def test_stable_softmax_rejects_invalid_inputs(values, temperature):
    with pytest.raises(ValueError):
        _stable_softmax(values, temperature)


def test_select_action_is_exported_from_package():
    from planu_core import select_action as exported_select_action

    assert exported_select_action is select_action
