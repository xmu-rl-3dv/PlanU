from dataclasses import FrozenInstanceError

import pytest

from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import (
    ActionCandidate,
    ActionScorer,
    CuriosityProvider,
    EnvironmentAdapter,
    EnvironmentState,
    TransitionResult,
)
from planu_core.nodes import ActionNode, LanguageNode


def make_action_node() -> ActionNode:
    return ActionNode(
        parent=LanguageNode(state=[0], state_key=(0,)),
        action=ActionCandidate("open", 4, "open microwave"),
        distribution=QuantileDistribution.from_scalar(0.2, 5, -1.0, 1.0),
    )


def test_action_node_merges_equal_outcomes():
    action = make_action_node()

    first = action.get_or_create_outcome(
        [1], (1,), reward=0.0, terminated=False, truncated=False
    )
    second = action.get_or_create_outcome(
        [1], (1,), reward=0.0, terminated=False, truncated=False
    )

    assert first is second
    assert len(action.children) == 1
    assert first.outcome_visits == 2


def test_preview_outcome_does_not_increment_visits():
    action = make_action_node()

    first = action.get_or_create_outcome(
        [1],
        (1,),
        reward=0.0,
        terminated=False,
        truncated=False,
        increment_visit=False,
    )
    second = action.get_or_create_outcome(
        [1],
        (1,),
        reward=1.0,
        terminated=False,
        truncated=False,
        increment_visit=False,
    )

    assert first is second
    assert first.outcome_visits == 0


def test_different_state_keys_create_distinct_outcomes():
    action = make_action_node()

    first = action.get_or_create_outcome(
        [1], (1,), reward=0.0, terminated=False, truncated=False
    )
    second = action.get_or_create_outcome(
        [2], (2,), reward=0.0, terminated=False, truncated=False
    )

    assert first is not second
    assert action.children == {(1,): first, (2,): second}


def test_created_outcome_records_parent_and_completion_flags():
    action = make_action_node()

    outcome = action.get_or_create_outcome(
        [1], (1,), reward=0.0, terminated=True, truncated=True
    )

    assert outcome.parent is action
    assert outcome.terminated is True
    assert outcome.truncated is True


def test_dataclass_mutable_defaults_are_independent():
    first_candidate = ActionCandidate("first", 1, "first")
    second_candidate = ActionCandidate("second", 2, "second")
    first_result = TransitionResult(
        EnvironmentState("first", None), 0.0, False, False
    )
    second_result = TransitionResult(
        EnvironmentState("second", None), 0.0, False, False
    )
    first_language = LanguageNode("first", "first")
    second_language = LanguageNode("second", "second")
    first_action = make_action_node()
    second_action = make_action_node()

    first_candidate.metadata["source"] = "test"
    first_result.info["source"] = "test"
    first_language.children["action"] = first_action
    first_action.children["outcome"] = first_language
    first_action.cumulative_returns.append(1.0)

    assert second_candidate.metadata == {}
    assert second_result.info == {}
    assert second_language.children == {}
    assert second_action.children == {}
    assert second_action.cumulative_returns == []


def test_action_candidate_is_frozen():
    candidate = ActionCandidate("open", 4, "open microwave")

    with pytest.raises(FrozenInstanceError):
        candidate.text = "close microwave"


def test_task_three_types_are_exported_from_package():
    import planu_core

    expected_exports = {
        "ActionCandidate": ActionCandidate,
        "ActionNode": ActionNode,
        "ActionScorer": ActionScorer,
        "CuriosityProvider": CuriosityProvider,
        "EnvironmentAdapter": EnvironmentAdapter,
        "EnvironmentState": EnvironmentState,
        "LanguageNode": LanguageNode,
        "TransitionResult": TransitionResult,
    }

    for name, expected in expected_exports.items():
        assert getattr(planu_core, name) is expected
