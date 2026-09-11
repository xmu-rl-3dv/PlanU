import copy
from dataclasses import fields
import inspect

import numpy as np
import pytest

import planu_core.search as search_module
from planu_core import PlanUConfig, SelectionSchedule
from planu_core.adapters.blockworld import BlockWorldAdapter
from planu_core.adapters.overcooked import OvercookedAdapter
from planu_core.adapters.virtualhome import VirtualHomeAdapter
from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentAdapter,
    EnvironmentState,
    TransitionResult,
)
from planu_core.search import PlanUSearch, TrajectoryResult
from tests.planu_core.fakes import FakeAdapter, UniformScorer


class TerminalAdapter(FakeAdapter):
    def reset(self, seed=None):
        super().reset(seed)
        self.reset_state = EnvironmentState(
            observation=np.array([2]),
            runtime={"position": 2},
        )
        return self.reset_state


class EmptyActionsAdapter(FakeAdapter):
    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return []


class FixedScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def score(self, observation, actions):
        self.calls += 1
        return self.scores


class FixedDistributionScorer:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def score(self, observation, actions):
        raise AssertionError("scalar scoring must not run")

    def score_distributions(self, observation, actions, levels):
        self.calls.append((observation, list(actions), tuple(levels)))
        return self.rows


class DuplicateActionAdapter(FakeAdapter):
    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return [
            ActionCandidate("advance", 0, "first"),
            ActionCandidate("advance", 1, "second"),
        ]


class StochasticAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.step_calls = 0

    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return [ActionCandidate("advance", 0, "advance")]

    def preview(self, state, action, rng):
        self.preview_calls += 1
        state = copy.deepcopy(state)
        state.runtime["position"] = 1
        state.observation = np.array([1])
        return TransitionResult(state, 0.0, False, False)

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        self.step_calls += 1
        position = 1 if self.step_calls % 2 else 2
        state.runtime["position"] = position
        state.observation = np.array([position])
        return TransitionResult(state, 0.0, False, False)

    def is_terminal(self, state):
        return False


class BranchingAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.preview_start_positions = []

    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return [
            ActionCandidate("left", -1, "left"),
            ActionCandidate("right", 1, "right"),
        ]

    def preview(self, state, action, rng):
        self.preview_calls += 1
        self.preview_start_positions.append(state.runtime["position"])
        return self._move(copy.deepcopy(state), action)

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        return self._move(state, action)

    def is_terminal(self, state):
        return False

    @staticmethod
    def _move(state, action):
        state.runtime["position"] = action.payload
        state.observation = np.array([action.payload])
        return TransitionResult(state, 0.5, False, False)


class FailingSecondPreviewAdapter(BranchingAdapter):
    def preview(self, state, action, rng):
        self.preview_calls += 1
        rng.random()
        if action.key == "right":
            raise RuntimeError("second preview failed")
        return self._move(copy.deepcopy(state), action)


class InPlaceObservationAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.preview_observation = None

    def preview(self, state, action, rng):
        self.preview_calls += 1
        state = copy.deepcopy(state)
        self._move_in_place(state)
        self.preview_observation = state.observation
        return TransitionResult(
            state,
            1.0,
            False,
            False,
            {"record_outcome": False},
        )

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        self._move_in_place(state)
        return TransitionResult(state, 1.0, False, False)

    @staticmethod
    def _move_in_place(state):
        state.runtime["position"] += 1
        state.observation[...] = state.runtime["position"]


class ConfigurableRewardAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.step_reward = 1.0

    def step(self, state, action, rng):
        result = super().step(state, action, rng)
        result.reward = self.step_reward
        return result


class HiddenResetAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.episode = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode += 1
        return EnvironmentState(
            observation=np.array([0]),
            runtime={"position": 0, "episode": self.episode},
        )

    def state_key(self, state):
        return (
            tuple(np.asarray(state.observation).tolist()),
            state.runtime["position"],
            state.runtime["episode"],
        )


class ConstantOutcomeKeyAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.outcome = 0

    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return [ActionCandidate("advance", 0, "advance")]

    def preview(self, state, action, rng):
        self.preview_calls += 1
        return TransitionResult(
            copy.deepcopy(state),
            0.0,
            False,
            False,
            {"record_outcome": False},
        )

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        self.outcome += 1
        state.runtime["position"] = self.outcome
        state.observation = np.array([self.outcome])
        return TransitionResult(state, 0.0, False, False)

    def state_key(self, state):
        if state.runtime["position"] == 0:
            return "root"
        return "constant-outcome"

    def is_terminal(self, state):
        return False


class ConstantRootKeyAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.episode = 0

    def reset(self, seed=None):
        self.episode += 1
        return EnvironmentState(
            observation=np.array([self.episode]),
            runtime={"position": 0},
        )

    def state_key(self, state):
        return "constant-root"


class ConstantHiddenRuntimeKeyAdapter(ConstantRootKeyAdapter):
    def reset(self, seed=None):
        self.episode += 1
        return EnvironmentState(
            observation=np.array([0]),
            runtime={"position": 0, "episode": self.episode},
        )


class UnfingerprintableRuntimeAdapter(FakeAdapter):
    def reset(self, seed=None):
        return EnvironmentState(
            observation=np.array([0]),
            runtime={"callback": lambda: None},
        )

    def state_key(self, state):
        return "constant-root"


class NoPreviewOutcomeAdapter(FakeAdapter):
    def preview(self, state, action, rng):
        result = super().preview(state, action, rng)
        result.info["record_outcome"] = False
        return result


class PreviewOwnsIsolationAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.original_state = None

    def clone(self, state):
        raise AssertionError("search must not clone before preview")

    def preview(self, state, action, rng):
        self.preview_calls += 1
        assert state is self.original_state
        return self._transition(copy.deepcopy(state))


class AdapterTerminalOnlyAdapter(FakeAdapter):
    @staticmethod
    def _transition(state):
        state.runtime["position"] += 1
        state.observation = np.array([state.runtime["position"]])
        return TransitionResult(
            state=state,
            reward=1.0,
            terminated=False,
            truncated=False,
        )


class TruncatingTransitionAdapter(FakeAdapter):
    def __init__(self, reason=None):
        super().__init__()
        self.reason = reason

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        state.runtime["position"] = 1
        state.observation = np.array([1])
        info = {}
        if self.reason is not None:
            info["truncation_reason"] = self.reason
        return TransitionResult(state, 0.0, False, True, info)

    def is_terminal(self, state):
        return False


class AdapterTruncationOnlyAdapter(FakeAdapter):
    def is_terminal(self, state):
        return False

    def is_truncated(self, state):
        return state.runtime["position"] >= 1


class TruncatedRootAdapter(AdapterTruncationOnlyAdapter):
    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.reset_state = EnvironmentState(
            observation=np.array([1]),
            runtime={"position": 1},
        )
        return self.reset_state


class LegacyHorizonTerminalAdapter(FakeAdapter):
    def __init__(self, goal_reached=False):
        super().__init__()
        self.goal_reached = goal_reached

    def preview(self, state, action, rng):
        self.preview_calls += 1
        return TransitionResult(
            copy.deepcopy(state),
            0.0,
            False,
            False,
            {"record_outcome": False},
        )

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        state.runtime["position"] = 1
        state.observation = np.array([1])
        return TransitionResult(
            state,
            1.0,
            self.goal_reached,
            not self.goal_reached,
            {"truncation_reason": "horizon"}
            if not self.goal_reached
            else {},
        )

    def is_terminal(self, state):
        return state.runtime["position"] >= 1

    def is_truncated(self, state):
        return state.runtime["position"] >= 1


class LegacyHorizonRootAdapter(LegacyHorizonTerminalAdapter):
    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return EnvironmentState(
            observation=np.array([1]),
            runtime={"position": 1},
        )


class LegacyThreeArgumentAdapter(FakeAdapter):
    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        return self._transition(state)


class VisitAwareThreeArgumentAdapter(LegacyThreeArgumentAdapter):
    def __init__(self):
        super().__init__()
        self.step_events = []

    def prepare_step(self, state, action, state_visit_count):
        self.step_events.append(
            ("prepare", state.runtime["position"], action.key, state_visit_count)
        )

    def step(self, state, action, rng):
        self.step_events.append(("step", state.runtime["position"], action.key))
        return super().step(state, action, rng)


def test_search_types_are_exported_from_package():
    import planu_core

    assert planu_core.PlanUSearch is PlanUSearch
    assert planu_core.TrajectoryResult is TrajectoryResult
    assert [field.name for field in fields(TrajectoryResult)] == [
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "truncation_reason",
        "final_observation",
        "selection_scores",
        "action_path",
        "state_path",
    ]


def test_environment_adapter_step_has_three_argument_protocol():
    assert list(inspect.signature(EnvironmentAdapter.step).parameters) == [
        "self",
        "state",
        "action",
        "rng",
    ]


@pytest.mark.parametrize(
    "adapter_type",
    [OvercookedAdapter, VirtualHomeAdapter, BlockWorldAdapter],
)
def test_concrete_adapters_expose_three_argument_step(adapter_type):
    assert list(inspect.signature(adapter_type.step).parameters) == [
        "self",
        "state",
        "action",
        "rng",
    ]


def test_root_is_initially_none():
    search = PlanUSearch(FakeAdapter(), UniformScorer(), PlanUConfig())

    assert search.root is None


def test_search_supports_legacy_three_argument_step_adapter():
    adapter = LegacyThreeArgumentAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    result = search.run_iteration(0, np.random.default_rng(1))

    assert result.actions == ["advance"]
    assert adapter.actions_taken == ["advance"]


def test_search_calls_optional_visit_hook_immediately_before_step():
    adapter = VisitAwareThreeArgumentAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    search.run_iteration(0, np.random.default_rng(2))
    search.run_iteration(1, np.random.default_rng(3))

    assert adapter.step_events == [
        ("prepare", 0, "advance", 0),
        ("step", 0, "advance"),
        ("prepare", 0, "advance", 1),
        ("step", 0, "advance"),
    ]


def test_two_step_run_merges_preview_and_execution_then_backs_up():
    adapter = FakeAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(
            max_depth=3,
            max_iterations=1,
            value_max=3.0,
        ),
    )

    result = search.run_iteration(
        iteration=0,
        rng=np.random.default_rng(1),
    )

    assert result.rewards == [1.0, 1.0]
    assert result.actions == ["advance", "advance"]
    assert result.terminated is True
    assert result.truncated is False
    np.testing.assert_array_equal(result.final_observation, [2])
    assert len(result.selection_scores) == 2
    assert len(result.action_path) == 2
    assert len(result.state_path) == 3
    assert result.state_path[0] is search.root
    assert all(isinstance(node.state, np.ndarray) for node in result.state_path)

    root_action = search.root.children["advance"]
    position_one = root_action.children[((1,), 1)]
    assert root_action.cumulative_returns == [2.0]
    assert position_one is result.state_path[1]
    assert position_one.outcome_visits == 1
    assert len(root_action.children) == 1

    second_action = position_one.children["advance"]
    position_two = second_action.children[((2,), 2)]
    assert second_action.cumulative_returns == [1.0]
    assert position_two is result.state_path[2]
    assert position_two.outcome_visits == 1


def test_preview_terminal_state_uses_adapter_terminal_normalization():
    search = PlanUSearch(
        AdapterTerminalOnlyAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=3, value_max=3.0),
    )

    result = search.run_iteration(0, np.random.default_rng(18))

    assert result.actions == ["advance", "advance"]
    assert result.terminated is True
    assert result.truncated is False
    np.testing.assert_array_equal(result.final_observation, [2])
    position_one = search.root.children["advance"].children[((1,), 1)]
    position_two = position_one.children["advance"].children[((2,), 2)]
    assert position_two is result.state_path[-1]
    assert position_two.terminated is True
    assert position_two.outcome_visits == 1


def test_explicit_horizon_truncation_precedes_legacy_terminal_check():
    search = PlanUSearch(
        LegacyHorizonTerminalAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=3),
    )

    result = search.run_iteration(0, np.random.default_rng(19))

    final_node = result.state_path[-1]
    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == "horizon"
    assert final_node.terminated is False
    assert final_node.truncated is True
    assert final_node.truncation_reason == "horizon"


def test_explicit_goal_at_horizon_remains_terminal():
    search = PlanUSearch(
        LegacyHorizonTerminalAdapter(goal_reached=True),
        UniformScorer(),
        PlanUConfig(max_depth=3),
    )

    result = search.run_iteration(0, np.random.default_rng(20))

    final_node = result.state_path[-1]
    assert result.terminated is True
    assert result.truncated is False
    assert result.truncation_reason is None
    assert final_node.terminated is True
    assert final_node.truncated is False


def test_max_depth_exhaustion_marks_final_state_and_result_truncated():
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    result = search.run_iteration(0, np.random.default_rng(2))

    assert result.terminated is False
    assert result.truncated is True
    assert result.state_path[-1].truncated is True
    assert result.state_path[-1].truncation_reason == "max_depth"
    assert result.truncation_reason == "max_depth"


def test_terminal_root_returns_empty_without_listing_or_scoring_actions(
    monkeypatch,
):
    adapter = TerminalAdapter()
    scorer = UniformScorer()
    backup_calls = []
    monkeypatch.setattr(
        search_module,
        "backup_trajectory",
        lambda actions, rewards, config: backup_calls.append(
            (list(actions), list(rewards))
        ),
    )
    search = PlanUSearch(adapter, scorer, PlanUConfig())

    result = search.run_iteration(
        iteration=0,
        rng=np.random.default_rng(3),
        reset_seed=41,
    )

    assert result.actions == []
    assert result.rewards == []
    assert result.terminated is True
    assert result.truncated is False
    assert result.truncation_reason is None
    np.testing.assert_array_equal(result.final_observation, [2])
    assert result.action_path == []
    assert result.state_path == [search.root]
    assert search.root.terminated is True
    assert search.root.visit_count == 0
    assert adapter.reset_seeds == [41]
    assert adapter.action_calls == 0
    assert scorer.calls == 0
    assert backup_calls == [([], [])]


def test_truncated_root_returns_empty_with_environment_reason():
    adapter = TruncatedRootAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())

    result = search.run_iteration(0, np.random.default_rng(3))

    assert result.actions == []
    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == "environment_truncated"
    assert result.state_path == [search.root]
    assert search.root.truncation_reason == "environment_truncated"
    assert adapter.action_calls == 0
    assert scorer.calls == 0


def test_pure_horizon_root_precedes_legacy_terminal_check():
    search = PlanUSearch(
        LegacyHorizonRootAdapter(),
        UniformScorer(),
        PlanUConfig(),
    )

    result = search.run_iteration(0, np.random.default_rng(21))

    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == "environment_truncated"
    assert search.root.terminated is False
    assert search.root.truncated is True


def test_preview_and_post_step_use_adapter_truncation_state():
    adapter = AdapterTruncationOnlyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=3),
    )
    state = adapter.reset()
    root = search._ensure_root(state)

    search.expand(root, state, np.random.default_rng(3))

    preview_outcome = root.children["advance"].children[((1,), 1)]
    assert preview_outcome.truncated is True
    assert preview_outcome.truncation_reason == "environment_truncated"

    result = search.run_iteration(0, np.random.default_rng(4))

    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == "environment_truncated"


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([], "number"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        (np.array([[1.0]]), "number"),
    ],
)
def test_invalid_scorer_output_raises_before_preview_or_tree_mutation(
    scores,
    message,
):
    adapter = FakeAdapter()
    scorer = FixedScorer(scores)
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)

    with pytest.raises(ValueError, match=message):
        search.expand(root, state, np.random.default_rng(5))

    assert scorer.calls == 1
    assert adapter.clone_calls == 0
    assert adapter.preview_calls == 0
    assert state.runtime == {"position": 0}
    assert root.children == {}


def test_duplicate_action_keys_are_rejected_before_scoring_or_mutation():
    adapter = DuplicateActionAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)

    with pytest.raises(ValueError, match="duplicate action key"):
        search.expand(root, state, np.random.default_rng(6))

    assert scorer.calls == 0
    assert adapter.clone_calls == 0
    assert adapter.preview_calls == 0
    assert root.children == {}


def test_stochastic_action_accumulates_and_reuses_multiple_outcomes():
    adapter = StochasticAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    for iteration in range(4):
        result = search.run_iteration(
            iteration,
            np.random.default_rng(iteration),
        )
        assert result.actions == ["advance"]
        assert result.truncated is True

    action = search.root.children["advance"]
    assert set(action.children) == {((1,), 1), ((2,), 2)}
    assert action.children[((1,), 1)].outcome_visits == 2
    assert action.children[((2,), 2)].outcome_visits == 2


def test_adapter_preview_isolates_each_action_without_core_clone():
    adapter = BranchingAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)

    search.expand(root, state, np.random.default_rng(7))

    assert adapter.preview_start_positions == [0, 0]
    assert adapter.clone_calls == 0
    assert adapter.preview_calls == 2
    assert scorer.calls == 1
    assert state.runtime == {"position": 0}
    np.testing.assert_array_equal(state.observation, [0])


def test_failed_second_preview_is_atomic_and_does_not_advance_caller_rng():
    adapter = FailingSecondPreviewAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig())
    root = search._ensure_root(adapter.reset())
    children = root.children
    children_before = root.children.copy()
    visit_count_before = root.visit_count
    rng = np.random.default_rng(19)
    rng_state_before = copy.deepcopy(rng.bit_generator.state)

    with pytest.raises(RuntimeError, match="second preview failed"):
        search.run_iteration(0, rng)

    assert root.children is children
    assert root.children == children_before
    assert root.visit_count == visit_count_before
    assert rng.bit_generator.state == rng_state_before


def test_tree_and_result_observations_are_independent_snapshots():
    adapter = InPlaceObservationAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1, value_max=3.0),
    )

    result = search.run_iteration(0, np.random.default_rng(20))

    root = search.root
    action = root.children["advance"]
    outcome = action.children[((1,), 1)]
    np.testing.assert_array_equal(root.state, [0])
    assert root.state_key == ((0,), 0)
    np.testing.assert_array_equal(action.preview_state, [1])
    np.testing.assert_array_equal(outcome.state, [1])
    assert outcome.state_key == ((1,), 1)
    np.testing.assert_array_equal(result.final_observation, [1])

    adapter.preview_observation[0] = 7
    adapter.reset_state.observation[0] = 8
    result.final_observation[0] = 9

    np.testing.assert_array_equal(root.state, [0])
    np.testing.assert_array_equal(action.preview_state, [1])
    np.testing.assert_array_equal(outcome.state, [1])


def test_preview_reward_configuration_and_record_outcome_flag():
    with_reward = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(include_preview_reward=True, value_max=3.0),
    )
    state = with_reward.adapter.reset()
    with_reward.expand(
        with_reward._ensure_root(state),
        state,
        np.random.default_rng(8),
    )

    without_reward = PlanUSearch(
        NoPreviewOutcomeAdapter(),
        UniformScorer(),
        PlanUConfig(include_preview_reward=False, value_max=3.0),
    )
    state = without_reward.adapter.reset()
    without_reward.expand(
        without_reward._ensure_root(state),
        state,
        np.random.default_rng(8),
    )

    with_action = with_reward.root.children["advance"]
    without_action = without_reward.root.children["advance"]
    np.testing.assert_array_equal(with_action.distribution.values, [2.0] * 51)
    np.testing.assert_array_equal(
        without_action.distribution.values,
        [1.0] * 51,
    )
    assert with_action.distribution.values.dtype == np.float64
    assert without_action.children == {}
    np.testing.assert_array_equal(without_action.preview_state, [1])


def test_categorical_initialization_maps_rows_and_shifts_preview_reward():
    adapter = BranchingAdapter()
    scorer = FixedDistributionScorer(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    config = PlanUConfig(
        categorical_initialization=True,
        n_quantiles=5,
        value_min=-1.0,
        value_max=1.0,
    )
    search = PlanUSearch(adapter, scorer, config)
    state = adapter.reset()
    root = search._ensure_root(state)

    search.expand(root, state, np.random.default_rng(8))

    assert len(scorer.calls) == 1
    assert scorer.calls[0][2] == config.categorical_levels
    np.testing.assert_allclose(
        root.children["left"].distribution.values,
        [0.6, 0.8, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        root.children["right"].distribution.values,
        [1.0] * 5,
    )


def test_categorical_initialization_requires_distribution_scorer():
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(categorical_initialization=True),
    )
    state = search.adapter.reset()
    root = search._ensure_root(state)

    with pytest.raises(ValueError, match="score_distributions"):
        search.expand(root, state, np.random.default_rng(8))

    assert search.scorer.calls == 0
    assert search.adapter.preview_calls == 0
    assert root.children == {}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "row count"),
        ([[1.0, 0.0]], "shape"),
        ([[1.0, 0.0, 0.0, 0.0, float("nan")]], "finite"),
        ([[1.0, 0.0, 0.0, 0.0, -0.1]], "nonnegative"),
        ([[0.0, 0.0, 0.0, 0.0, 0.0]], "nonzero"),
    ],
)
def test_invalid_distribution_scores_fail_before_preview_or_mutation(
    rows,
    message,
):
    adapter = FakeAdapter()
    scorer = FixedDistributionScorer(rows)
    search = PlanUSearch(
        adapter,
        scorer,
        PlanUConfig(categorical_initialization=True),
    )
    state = adapter.reset()
    root = search._ensure_root(state)

    with pytest.raises(ValueError, match=message):
        search.expand(root, state, np.random.default_rng(8))

    assert adapter.preview_calls == 0
    assert root.children == {}


def test_expand_delegates_preview_isolation_without_cloning_state():
    adapter = PreviewOwnsIsolationAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig())
    state = adapter.reset()
    adapter.original_state = state
    root = search._ensure_root(state)

    search.expand(root, state, np.random.default_rng(21))

    assert adapter.preview_calls == 1
    assert state.runtime == {"position": 0}
    np.testing.assert_array_equal(state.observation, [0])
    np.testing.assert_array_equal(root.children["advance"].preview_state, [1])


@pytest.mark.parametrize("completion_flag", ["terminated", "truncated"])
def test_expand_is_noop_for_completed_state_nodes(completion_flag):
    adapter = FakeAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)
    setattr(root, completion_flag, True)

    search.expand(root, state, np.random.default_rng(9))

    assert adapter.action_calls == 0
    assert scorer.calls == 0
    assert root.children == {}


def test_expand_is_noop_after_node_has_children():
    adapter = FakeAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)
    rng = np.random.default_rng(10)
    search.expand(root, state, rng)

    search.expand(root, state, rng)

    assert adapter.action_calls == 1
    assert scorer.calls == 1
    assert adapter.preview_calls == 1


def test_reset_hidden_state_key_mismatch_is_rejected():
    adapter = HiddenResetAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig())
    first_state = adapter.reset()
    root = search._ensure_root(first_state)
    second_state = adapter.reset()

    with pytest.raises(ValueError, match="persistent PlanU root"):
        search._ensure_root(second_state)

    assert search.root is root
    np.testing.assert_array_equal(search.root.state, [0])


def test_state_key_collision_detection_is_disabled_by_default():
    adapter = ConstantOutcomeKeyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    search.run_iteration(0, np.random.default_rng(1))
    search.run_iteration(1, np.random.default_rng(2))

    outcome = search.root.children["advance"].children["constant-outcome"]
    np.testing.assert_array_equal(outcome.state, [1])
    assert outcome.outcome_visits == 2


def test_outcome_state_key_collision_raises_and_rolls_back():
    adapter = ConstantOutcomeKeyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1, debug_state_keys=True),
    )
    search.run_iteration(0, np.random.default_rng(1))
    root = search.root
    action = root.children["advance"]
    outcome = action.children["constant-outcome"]
    before = (
        root.visit_count,
        action.visit_count,
        list(action.cumulative_returns),
        outcome.outcome_visits,
    )

    with pytest.raises(ValueError, match="state key collision"):
        search.run_iteration(1, np.random.default_rng(2))

    assert search.root is root
    assert (
        root.visit_count,
        action.visit_count,
        action.cumulative_returns,
        outcome.outcome_visits,
    ) == before
    np.testing.assert_array_equal(outcome.state, [1])


def test_root_state_key_collision_raises_without_replacing_root():
    adapter = ConstantRootKeyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=True),
    )
    root = search._ensure_root(adapter.reset())

    with pytest.raises(ValueError, match="state key collision"):
        search._ensure_root(adapter.reset())

    assert search.root is root
    np.testing.assert_array_equal(root.state, [1])


def test_full_state_fingerprint_detects_hidden_runtime_collision():
    adapter = ConstantHiddenRuntimeKeyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=True),
    )
    search._ensure_root(adapter.reset())

    with pytest.raises(ValueError, match="state key collision"):
        search._ensure_root(adapter.reset())


def test_hidden_runtime_collision_merges_when_debugging_is_disabled():
    adapter = ConstantHiddenRuntimeKeyAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=False),
    )
    root = search._ensure_root(adapter.reset())

    assert search._ensure_root(adapter.reset()) is root


def test_unfingerprintable_full_state_errors_only_in_debug_mode():
    adapter = UnfingerprintableRuntimeAdapter()
    regular_search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=False),
    )

    regular_search._ensure_root(adapter.reset())

    debug_search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=True),
    )
    with pytest.raises(ValueError, match="full environment state"):
        debug_search._ensure_root(adapter.reset())


def test_adapter_state_fingerprint_is_preferred_over_pickle():
    adapter = ConstantRootKeyAdapter()
    calls = []

    def state_fingerprint(state):
        calls.append(state.runtime["position"])
        return bytes(np.asarray(state.observation))

    adapter.state_fingerprint = state_fingerprint
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(debug_state_keys=True),
    )
    search._ensure_root(adapter.reset())

    with pytest.raises(ValueError, match="state key collision"):
        search._ensure_root(adapter.reset())

    assert calls == [0, 0]


def test_seeded_sampling_is_reproducible():
    config = PlanUConfig(
        max_depth=1,
        include_preview_reward=False,
        selection_schedule=SelectionSchedule(
            always_sample_before=8,
            probabilistic_sample_before=8,
        ),
    )

    def run_sequence():
        adapter = BranchingAdapter()
        search = PlanUSearch(adapter, UniformScorer(), config)
        rng = np.random.default_rng(17)
        return [
            search.run_iteration(iteration, rng).actions[0]
            for iteration in range(8)
        ]

    first = run_sequence()
    second = run_sequence()

    assert first == second
    assert first == [
        "right",
        "left",
        "right",
        "left",
        "left",
        "left",
        "left",
        "right",
    ]


def test_empty_actions_truncates_unexpanded_state_and_backs_up_empty(
    monkeypatch,
):
    adapter = EmptyActionsAdapter()
    scorer = UniformScorer()
    backup_calls = []
    monkeypatch.setattr(
        search_module,
        "backup_trajectory",
        lambda actions, rewards, config: backup_calls.append(
            (list(actions), list(rewards))
        ),
    )
    search = PlanUSearch(adapter, scorer, PlanUConfig())

    result = search.run_iteration(0, np.random.default_rng(4))

    assert result.actions == []
    assert result.rewards == []
    assert result.terminated is False
    assert result.truncated is True
    assert result.state_path == [search.root]
    assert search.root.children == {}
    assert search.root.truncated is True
    assert search.root.truncation_reason == "no_legal_actions"
    assert result.truncation_reason == "no_legal_actions"
    assert search.root.visit_count == 1
    assert adapter.action_calls == 1
    assert scorer.calls == 0
    assert backup_calls == [([], [])]


def test_backup_failure_restores_existing_tree_mutations(monkeypatch):
    adapter = StochasticAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )
    search.run_iteration(0, np.random.default_rng(21))
    root = search.root
    action = root.children["advance"]
    existing_outcome = action.children[((1,), 1)]
    root_children = root.children
    action_children = action.children
    root_children_before = root.children.copy()
    action_children_before = action.children.copy()
    root_state_before = (
        root.visit_count,
        root.terminated,
        root.truncated,
        root.truncation_reason,
        root.outcome_visits,
    )
    outcome_state_before = (
        existing_outcome.visit_count,
        existing_outcome.terminated,
        existing_outcome.truncated,
        existing_outcome.truncation_reason,
        existing_outcome.outcome_visits,
    )
    action_visit_count_before = action.visit_count

    def fail_backup(actions, rewards, config):
        raise RuntimeError("backup failed")

    monkeypatch.setattr(search_module, "backup_trajectory", fail_backup)

    with pytest.raises(RuntimeError, match="backup failed"):
        search.run_iteration(1, np.random.default_rng(22))

    assert search.root is root
    assert root.children is root_children
    assert action.children is action_children
    assert root.children == root_children_before
    assert action.children == action_children_before
    assert (
        root.visit_count,
        root.terminated,
        root.truncated,
        root.truncation_reason,
        root.outcome_visits,
    ) == root_state_before
    assert (
        existing_outcome.visit_count,
        existing_outcome.terminated,
        existing_outcome.truncated,
        existing_outcome.truncation_reason,
        existing_outcome.outcome_visits,
    ) == outcome_state_before
    assert action.visit_count == action_visit_count_before


def test_backup_failure_discards_root_created_by_iteration(monkeypatch):
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=1),
    )

    def fail_backup(actions, rewards, config):
        raise RuntimeError("backup failed")

    monkeypatch.setattr(search_module, "backup_trajectory", fail_backup)

    with pytest.raises(RuntimeError, match="backup failed"):
        search.run_iteration(0, np.random.default_rng(23))

    assert search.root is None


@pytest.mark.parametrize("reward", [float("nan"), float("inf")])
def test_non_finite_actual_reward_fails_before_outcome_mutation(
    monkeypatch,
    reward,
):
    adapter = ConfigurableRewardAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig(max_depth=1))
    state = adapter.reset()
    root = search._ensure_root(state)
    search.expand(root, state, np.random.default_rng(24))
    action = root.children["advance"]
    outcome = action.children[((1,), 1)]
    root_visit_count_before = root.visit_count
    outcome_visits_before = outcome.outcome_visits
    adapter.step_reward = reward

    def unexpected_backup(actions, rewards, config):
        pytest.fail("backup must not run for a non-finite reward")

    monkeypatch.setattr(search_module, "backup_trajectory", unexpected_backup)

    with pytest.raises(ValueError, match="reward must be finite"):
        search.run_iteration(0, np.random.default_rng(25))

    assert root.visit_count == root_visit_count_before
    assert outcome.outcome_visits == outcome_visits_before


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("time_limit", "time_limit"),
        (None, "environment_truncated"),
    ],
)
def test_transition_truncation_reason_is_recorded(reason, expected):
    search = PlanUSearch(
        TruncatingTransitionAdapter(reason),
        UniformScorer(),
        PlanUConfig(max_depth=3),
    )

    result = search.run_iteration(0, np.random.default_rng(26))

    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == expected
    assert result.state_path[-1].truncation_reason == expected
