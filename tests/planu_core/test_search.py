from dataclasses import fields

import numpy as np
import pytest

import planu_core.search as search_module
from planu_core import PlanUConfig, SelectionSchedule
from planu_core.interfaces import (
    ActionCandidate,
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
        return self._move(state, action)

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


class NoPreviewOutcomeAdapter(FakeAdapter):
    def preview(self, state, action, rng):
        result = super().preview(state, action, rng)
        result.info["record_outcome"] = False
        return result


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


def test_search_types_are_exported_from_package():
    import planu_core

    assert planu_core.PlanUSearch is PlanUSearch
    assert planu_core.TrajectoryResult is TrajectoryResult
    assert [field.name for field in fields(TrajectoryResult)] == [
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "final_observation",
        "selection_scores",
        "action_path",
        "state_path",
    ]


def test_root_is_initially_none():
    search = PlanUSearch(FakeAdapter(), UniformScorer(), PlanUConfig())

    assert search.root is None


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
    np.testing.assert_array_equal(result.final_observation, [2])
    assert result.action_path == []
    assert result.state_path == [search.root]
    assert search.root.terminated is True
    assert search.root.visit_count == 0
    assert adapter.reset_seeds == [41]
    assert adapter.action_calls == 0
    assert scorer.calls == 0
    assert backup_calls == [([], [])]


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


def test_each_preview_uses_an_independent_clone_without_mutating_runtime():
    adapter = BranchingAdapter()
    scorer = UniformScorer()
    search = PlanUSearch(adapter, scorer, PlanUConfig())
    state = adapter.reset()
    root = search._ensure_root(state)

    search.expand(root, state, np.random.default_rng(7))

    assert adapter.preview_start_positions == [0, 0]
    assert adapter.clone_calls == 2
    assert adapter.preview_calls == 2
    assert scorer.calls == 1
    assert state.runtime == {"position": 0}
    np.testing.assert_array_equal(state.observation, [0])


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
    assert search.root.visit_count == 1
    assert adapter.action_calls == 1
    assert scorer.calls == 0
    assert backup_calls == [([], [])]
