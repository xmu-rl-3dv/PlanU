from dataclasses import FrozenInstanceError
from inspect import signature
from typing import Hashable, Mapping, Optional, Sequence, get_type_hints

import numpy as np
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
        [1], (1,), terminated=False, truncated=False
    )
    second = action.get_or_create_outcome(
        [1], (1,), terminated=False, truncated=False
    )

    assert first is second
    assert len(action.children) == 1
    assert first.outcome_visits == 2


def test_preview_outcome_does_not_increment_visits():
    action = make_action_node()

    first = action.get_or_create_outcome(
        [1],
        (1,),
        terminated=False,
        truncated=False,
        increment_visit=False,
    )
    second = action.get_or_create_outcome(
        [1],
        (1,),
        terminated=False,
        truncated=False,
        increment_visit=False,
    )

    assert first is second
    assert first.outcome_visits == 0


def test_different_state_keys_create_distinct_outcomes():
    action = make_action_node()

    first = action.get_or_create_outcome(
        [1], (1,), terminated=False, truncated=False
    )
    second = action.get_or_create_outcome(
        [2], (2,), terminated=False, truncated=False
    )

    assert first is not second
    assert action.children == {(1,): first, (2,): second}


def test_created_outcome_records_parent_and_completion_flags():
    action = make_action_node()

    outcome = action.get_or_create_outcome(
        [1], (1,), terminated=True, truncated=True
    )

    assert outcome.parent is action
    assert outcome.terminated is True
    assert outcome.truncated is True


@pytest.mark.parametrize(
    ("first_terminated", "second_terminated"),
    [(True, False), (False, True)],
)
def test_action_node_rejects_terminated_mismatch(
    first_terminated: bool,
    second_terminated: bool,
):
    action = make_action_node()
    action.get_or_create_outcome(
        [1],
        (1,),
        terminated=first_terminated,
        truncated=False,
    )

    with pytest.raises(ValueError, match="terminated"):
        action.get_or_create_outcome(
            [1],
            (1,),
            terminated=second_terminated,
            truncated=False,
        )


@pytest.mark.parametrize(
    ("first_truncated", "second_truncated"),
    [(True, False), (False, True)],
)
def test_action_node_rejects_truncated_mismatch(
    first_truncated: bool,
    second_truncated: bool,
):
    action = make_action_node()
    action.get_or_create_outcome(
        [1],
        (1,),
        terminated=False,
        truncated=first_truncated,
    )

    with pytest.raises(ValueError, match="truncated"):
        action.get_or_create_outcome(
            [1],
            (1,),
            terminated=False,
            truncated=second_truncated,
        )


def test_dataclass_mutable_defaults_are_independent():
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

    first_result.info["source"] = "test"
    first_language.children["action"] = first_action
    first_action.children["outcome"] = first_language
    first_action.cumulative_returns.append(1.0)

    assert second_result.info == {}
    assert second_language.children == {}
    assert second_action.children == {}
    assert second_action.cumulative_returns == []


def test_action_candidate_is_frozen():
    candidate = ActionCandidate("open", 4, "open microwave")

    with pytest.raises(FrozenInstanceError):
        candidate.text = "close microwave"


def test_action_candidate_defensively_copies_metadata():
    metadata = {"source": "test"}
    candidate = ActionCandidate("open", 4, "open microwave", metadata)

    metadata["source"] = "changed"

    assert candidate.metadata == {"source": "test"}


def test_action_candidate_metadata_is_immutable():
    candidate = ActionCandidate(
        "open",
        4,
        "open microwave",
        {"source": "test"},
    )

    with pytest.raises(FrozenInstanceError):
        candidate.metadata = {}
    with pytest.raises(TypeError):
        candidate.metadata["source"] = "changed"


def test_action_candidate_equality_and_hash_depend_only_on_key():
    first = ActionCandidate(
        "open",
        {"unhashable": ["payload"]},
        "open microwave",
        {"source": "first"},
    )
    same_key = ActionCandidate(
        "open",
        ["different", "unhashable", "payload"],
        "different text",
        {"source": "second"},
    )
    different_key = ActionCandidate(
        "close",
        {"unhashable": ["payload"]},
        "open microwave",
        {"source": "first"},
    )

    assert first == same_key
    assert first != different_key
    assert hash(first) == hash(same_key)
    assert {first, same_key, different_key} == {first, different_key}


def test_environment_adapter_protocol_matches_stateful_contract():
    expected = {
        "reset": (
            {"seed": Optional[int], "return": EnvironmentState},
            {"seed": None},
        ),
        "clone": (
            {"state": EnvironmentState, "return": EnvironmentState},
            {},
        ),
        "actions": (
            {
                "state": EnvironmentState,
                "state_visit_count": int,
                "return": Sequence[ActionCandidate],
            },
            {"state_visit_count": 0},
        ),
        "preview": (
            {
                "state": EnvironmentState,
                "action": ActionCandidate,
                "rng": np.random.Generator,
                "return": TransitionResult,
            },
            {},
        ),
        "step": (
            {
                "state": EnvironmentState,
                "action": ActionCandidate,
                "rng": np.random.Generator,
                "return": TransitionResult,
            },
            {},
        ),
        "state_key": (
            {"state": EnvironmentState, "return": Hashable},
            {},
        ),
        "is_terminal": (
            {"state": EnvironmentState, "return": bool},
            {},
        ),
    }

    for method_name, (expected_hints, expected_defaults) in expected.items():
        method = getattr(EnvironmentAdapter, method_name)
        parameters = signature(method).parameters

        assert get_type_hints(method) == expected_hints
        assert {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not parameter.empty
        } == expected_defaults


def test_provider_protocol_return_types():
    assert get_type_hints(ActionScorer.score)["return"] == Sequence[float]
    assert get_type_hints(CuriosityProvider.train)["return"] == Optional[
        Mapping[str, float]
    ]


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
