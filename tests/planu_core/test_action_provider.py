from inspect import signature
from typing import Optional, Sequence, get_type_hints

import numpy as np
import pytest

import planu_core
from planu_core import PlanUConfig
from planu_core.interfaces import (
    ActionCandidate,
    ActionProvider,
    EnvironmentState,
)
from planu_core.search import PlanUSearch
from tests.planu_core.fakes import FakeAdapter, UniformScorer


class RecordingActionProvider:
    def __init__(self):
        self.calls = []

    def actions(self, state, state_visit_count=0):
        self.calls.append((state, state_visit_count))
        return [ActionCandidate("advance", 0, "advance")]


class FailingActionProvider:
    def actions(self, state, state_visit_count=0):
        raise RuntimeError("provider failed")


class RecordingAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.action_arguments = []

    def actions(self, state, state_visit_count=0):
        self.action_arguments.append((state, state_visit_count))
        return super().actions(state, state_visit_count)


def test_action_provider_protocol_is_exported_with_expected_contract():
    hints = get_type_hints(ActionProvider.actions)
    parameters = signature(ActionProvider.actions).parameters

    assert planu_core.ActionProvider is ActionProvider
    assert hints == {
        "state": EnvironmentState,
        "state_visit_count": int,
        "return": Sequence[ActionCandidate],
    }
    assert parameters["state_visit_count"].default == 0
    assert list(signature(PlanUSearch.__init__).parameters)[-2:] == [
        "curiosity",
        "action_provider",
    ]

    curiosity = object()
    provider = RecordingActionProvider()
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(),
        curiosity,
        provider,
    )

    assert search.curiosity is curiosity
    assert search.action_provider is provider
    assert get_type_hints(PlanUSearch.__init__)["action_provider"] == Optional[
        ActionProvider
    ]


def test_run_iteration_uses_explicit_provider_with_prior_visit_count():
    adapter = FakeAdapter()
    provider = RecordingActionProvider()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
        action_provider=provider,
    )

    search.run_iteration(0, np.random.default_rng(1))

    assert provider.calls == [(adapter.reset_state, 0)]
    assert adapter.action_calls == 0
    assert list(search.root.children) == ["advance"]


def test_expand_uses_adapter_actions_by_default():
    adapter = RecordingAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)
    root.visit_count = 4

    search.expand(root, state, np.random.default_rng(2))

    assert adapter.action_arguments == [(state, 4)]
    assert list(root.children) == ["advance"]


def test_provider_failure_rolls_back_iteration_tree_mutations():
    adapter = FakeAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(),
        action_provider=FailingActionProvider(),
    )
    root = search._ensure_root(adapter.reset())
    children = root.children

    with pytest.raises(RuntimeError, match="provider failed"):
        search.run_iteration(0, np.random.default_rng(3))

    assert search.root is root
    assert root.visit_count == 0
    assert root.children is children
    assert root.children == {}
    assert adapter.action_calls == 0
    assert adapter.preview_calls == 0
    assert search.scorer.calls == 0
