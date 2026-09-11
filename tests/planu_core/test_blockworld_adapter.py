import copy
from dataclasses import dataclass
import ast
import importlib.util
from pathlib import Path
import sys
import types
from typing import NamedTuple
from typing import TypeVar

import numpy as np
import pytest

from planu_core import PlanUSearch
from planu_core.adapters.blockworld import (
    BlockWorldAdapter,
    BlockWorldScorer,
    blockworld_config,
)
from planu_core.interfaces import EnvironmentState


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = (
    REPO_ROOT / "blockworld" / "reasoners" / "algorithm" / "planU.py"
)
ALGORITHM_INIT_PATH = WRAPPER_PATH.parent / "__init__.py"
EVALUATE_PATH = REPO_ROOT / "blockworld" / "evaluate_stochastic.py"


class BlockState(NamedTuple):
    step_idx: int
    blocks_state: str
    buffered_action: str


class FakeWorldModel:
    def __init__(self):
        self.rng = np.random.default_rng(999)
        self.step_calls = 0

    def init_state(self):
        return BlockState(0, "root", "")

    def step(self, state, action):
        self.step_calls += 1
        return BlockState(1, action, ""), {"success": True}

    def is_terminal(self, state):
        return state.step_idx >= 1


class RecordingSearchConfig:
    def __init__(self):
        self.fast_nodes = []
        self.fast_details = {"intuition": 0.2, "nested": ["original"]}
        self.reward_call = None

    def get_actions(self, state):
        return ["move"]

    def fast_reward(self, node, action):
        self.fast_nodes.append(node)
        return 0.75, self.fast_details

    def reward(self, node, action, **kwargs):
        self.reward_call = (node, action, kwargs)
        return 2.5, {"scored": True}


def test_actions_preserve_prior_and_fast_reward_visit_count_view():
    world = FakeWorldModel()
    search_config = RecordingSearchConfig()
    adapter = BlockWorldAdapter(world, search_config)
    state = adapter.reset()

    candidate = adapter.actions(state, state_visit_count=1)[0]

    assert candidate.key == "move"
    assert candidate.payload == "move"
    assert candidate.text == "move"
    assert candidate.metadata["prior_score"] == 0.75
    assert "node_view" not in candidate.metadata
    assert search_config.fast_nodes[0].state == state.observation
    assert search_config.fast_nodes[0].cum_rewards == []
    with pytest.raises(TypeError):
        candidate.metadata["prior_score"] = 0.0

    search_config.fast_details["nested"].append("mutated")
    assert candidate.metadata["fast_reward_details"] == {
        "intuition": 0.2,
        "nested": ["original"],
    }

    later = adapter.actions(state, state_visit_count=4)[0]
    assert "node_view" not in later.metadata
    assert len(search_config.fast_nodes[1].cum_rewards) == 3


@pytest.mark.parametrize(
    ("actions", "prior", "message"),
    [
        (["move", "move"], 0.5, "duplicate action key"),
        ([["move"]], 0.5, "hashable"),
        (["move"], np.nan, "prior score must be finite"),
        (["move"], [0.5], "prior score must be scalar"),
    ],
)
def test_actions_reject_invalid_keys_and_priors(actions, prior, message):
    class InvalidConfig:
        def get_actions(self, state):
            return actions

        def fast_reward(self, node, action):
            return prior, {}

    adapter = BlockWorldAdapter(FakeWorldModel(), InvalidConfig())

    with pytest.raises(ValueError, match=message):
        adapter.actions(adapter.reset(), state_visit_count=1)


def test_preview_clones_state_without_stepping_or_recording_an_outcome():
    world = FakeWorldModel()
    adapter = BlockWorldAdapter(world, RecordingSearchConfig())
    state = EnvironmentState(
        observation=BlockState(0, ["root"], ""),
        runtime=None,
    )
    action = adapter.actions(state, state_visit_count=1)[0]

    result = adapter.preview(state, action, np.random.default_rng(1))

    assert world.step_calls == 0
    assert result.state is not state
    assert result.state.observation == state.observation
    assert result.state.observation is not state.observation
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info == {"record_outcome": False}


def test_step_forwards_fast_details_node_view_aux_and_supplied_rng():
    world = FakeWorldModel()
    search_config = RecordingSearchConfig()
    adapter = BlockWorldAdapter(world, search_config)
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]
    rng = np.random.default_rng(42)

    adapter.prepare_step(state, action, state_visit_count=3)
    result = adapter.step(state, action, rng)

    node, payload, kwargs = search_config.reward_call
    assert node is result.info["node_view"]
    assert node is not search_config.fast_nodes[0]
    assert node.state == state.observation
    assert node.cum_rewards == [0.0, 0.0, 0.0]
    assert payload == "move"
    assert kwargs == {
        "intuition": 0.2,
        "nested": ["original"],
        "success": True,
    }
    assert world.rng is rng
    assert result.state.observation == BlockState(1, "move", "")
    assert result.reward == 2.5
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["node_view"] is node
    assert result.info["reward_visit"] == {
        "reward": 2.5,
        "count_before": 3,
        "count_after": 4,
    }
    assert {
        key: value
        for key, value in result.info.items()
        if key not in {"node_view", "reward_visit"}
    } == {"scored": True, "success": True}


def test_step_supports_a_bare_state_result():
    class BareWorld(FakeWorldModel):
        def step(self, state, action):
            self.step_calls += 1
            return BlockState(1, action, "")

    world = BareWorld()
    search_config = RecordingSearchConfig()
    adapter = BlockWorldAdapter(world, search_config)
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]

    result = adapter.step(state, action, np.random.default_rng(3))

    assert search_config.reward_call[2] == {
        "intuition": 0.2,
        "nested": ["original"],
    }
    assert result.info["node_view"] is search_config.reward_call[0]
    assert result.info["node_view"] is not search_config.fast_nodes[0]
    assert result.info["node_view"].cum_rewards == []
    assert result.info["reward_visit"] == {
        "reward": 2.5,
        "count_before": 0,
        "count_after": 1,
    }
    assert {
        key: value
        for key, value in result.info.items()
        if key not in {"node_view", "reward_visit"}
    } == {"scored": True}


def test_repeated_steps_derive_reward_visit_count_from_core_argument():
    class VisitRecordingConfig(RecordingSearchConfig):
        def __init__(self):
            super().__init__()
            self.reward_visits = []

        def reward(self, node, action, **kwargs):
            self.reward_visits.append(len(node.cum_rewards))
            return float(len(self.reward_visits)), {}

    search_config = VisitRecordingConfig()
    adapter = BlockWorldAdapter(FakeWorldModel(), search_config)
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]

    results = []
    for seed in range(3):
        adapter.prepare_step(state, action, state_visit_count=seed)
        results.append(
            adapter.step(
                state,
                action,
                np.random.default_rng(seed),
            )
        )

    assert search_config.reward_visits == [0, 1, 2]
    assert "node_view" not in action.metadata
    assert [result.info["node_view"].cum_rewards for result in results] == [
        [],
        [0.0],
        [0.0, 0.0],
    ]
    assert len({id(result.info["node_view"]) for result in results}) == 3
    assert [
        result.info["reward_visit"]
        for result in results
    ] == [
        {"reward": 1.0, "count_before": 0, "count_after": 1},
        {"reward": 2.0, "count_before": 1, "count_after": 2},
        {"reward": 3.0, "count_before": 2, "count_after": 3},
    ]


def test_different_actions_receive_core_reward_visit_count():
    class MultiActionConfig(RecordingSearchConfig):
        def __init__(self):
            super().__init__()
            self.reward_visits = []

        def get_actions(self, state):
            return ["move", "stack"]

        def reward(self, node, action, **kwargs):
            self.reward_visits.append((action, len(node.cum_rewards)))
            return 1.0, {}

    search_config = MultiActionConfig()
    adapter = BlockWorldAdapter(FakeWorldModel(), search_config)
    state = adapter.reset()
    first, second = adapter.actions(state, state_visit_count=1)

    adapter.prepare_step(state, first, state_visit_count=0)
    adapter.step(state, first, np.random.default_rng(1))
    adapter.prepare_step(state, second, state_visit_count=1)
    adapter.step(state, second, np.random.default_rng(2))

    assert "node_view" not in first.metadata
    assert "node_view" not in second.metadata
    assert search_config.reward_visits == [("move", 0), ("stack", 1)]


def test_failed_world_or_reward_call_does_not_advance_reward_visit_count():
    class FailingWorld(FakeWorldModel):
        def step(self, state, action):
            raise RuntimeError("world failed")

    class FailingRewardConfig(RecordingSearchConfig):
        def reward(self, node, action, **kwargs):
            raise RuntimeError("reward failed")

    world_adapter = BlockWorldAdapter(
        FailingWorld(),
        RecordingSearchConfig(),
    )
    world_state = world_adapter.reset()
    world_action = world_adapter.actions(
        world_state,
        state_visit_count=1,
    )[0]

    with pytest.raises(RuntimeError, match="world failed"):
        world_adapter.step(
            world_state,
            world_action,
            np.random.default_rng(1),
        )
    assert "node_view" not in world_action.metadata

    reward_adapter = BlockWorldAdapter(
        FakeWorldModel(),
        FailingRewardConfig(),
    )
    reward_state = reward_adapter.reset()
    reward_action = reward_adapter.actions(
        reward_state,
        state_visit_count=1,
    )[0]

    with pytest.raises(RuntimeError, match="reward failed"):
        reward_adapter.step(
            reward_state,
            reward_action,
            np.random.default_rng(2),
        )
    assert "node_view" not in reward_action.metadata


def test_failed_terminal_check_does_not_advance_reward_visit_count():
    class FailingTerminalWorld(FakeWorldModel):
        def __init__(self):
            super().__init__()
            self.terminal_calls = 0

        def is_terminal(self, state):
            self.terminal_calls += 1
            raise RuntimeError("terminal failed")

    world = FailingTerminalWorld()
    search_config = RecordingSearchConfig()
    adapter = BlockWorldAdapter(world, search_config)
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]

    with pytest.raises(RuntimeError, match="terminal failed"):
        adapter.step(state, action, np.random.default_rng(3))

    assert search_config.reward_call is not None
    assert world.terminal_calls == 1
    assert search_config.reward_call[0].cum_rewards == []
    assert "node_view" not in action.metadata


@pytest.mark.parametrize(
    ("reward_result", "message"),
    [
        (np.nan, "reward result must be a pair"),
        (([1.0], {}), "reward must be scalar"),
        ((np.inf, {}), "reward must be finite"),
        ((1.0, []), "reward details must be a mapping"),
    ],
)
def test_step_validates_reward_result_shape_and_finiteness(
    reward_result,
    message,
):
    class InvalidRewardConfig(RecordingSearchConfig):
        def reward(self, node, action, **kwargs):
            return reward_result

    adapter = BlockWorldAdapter(FakeWorldModel(), InvalidRewardConfig())
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]

    with pytest.raises(ValueError, match=message):
        adapter.step(state, action, np.random.default_rng(5))
    assert "node_view" not in action.metadata


@dataclass
class NestedState:
    block: BlockState
    history: list
    values: np.ndarray
    labels: dict


def test_state_key_recursively_covers_namedtuple_dataclass_list_and_array():
    observation = NestedState(
        block=BlockState(2, "on(a,b)", "stack a b"),
        history=["pick a", {"ok": True}],
        values=np.array([[1, 2], [3, 4]], dtype=np.int16),
        labels={"goal": ["on(a,b)"]},
    )
    equivalent = copy.deepcopy(observation)
    adapter = BlockWorldAdapter(FakeWorldModel(), RecordingSearchConfig())

    key = adapter.state_key(EnvironmentState(observation, None))

    assert key == adapter.state_key(EnvironmentState(equivalent, None))
    assert hash(key) == hash(
        adapter.state_key(EnvironmentState(equivalent, None))
    )
    changed_step = copy.deepcopy(observation)
    changed_step.block = changed_step.block._replace(step_idx=3)
    changed_buffer = copy.deepcopy(observation)
    changed_buffer.block = changed_buffer.block._replace(
        buffered_action="put a"
    )
    assert key != adapter.state_key(EnvironmentState(changed_step, None))
    assert key != adapter.state_key(EnvironmentState(changed_buffer, None))


def test_state_key_rejects_identity_based_objects():
    adapter = BlockWorldAdapter(FakeWorldModel(), RecordingSearchConfig())

    with pytest.raises(TypeError, match="unsupported state value"):
        adapter.state_key(EnvironmentState(object(), None))


def test_reset_seed_and_step_rng_drive_the_world_model():
    class RandomWorld(FakeWorldModel):
        def step(self, state, action):
            draw = int(self.rng.integers(0, 1_000_000))
            return BlockState(1, str(draw), "")

    world = RandomWorld()
    original_rng = world.rng
    adapter = BlockWorldAdapter(world, RecordingSearchConfig())

    adapter.reset()
    assert world.rng is original_rng
    adapter.reset(seed=17)
    assert int(world.rng.integers(0, 1_000_000)) == int(
        np.random.default_rng(17).integers(0, 1_000_000)
    )

    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=1)[0]
    supplied_rng = np.random.default_rng(91)
    expected_rng = np.random.default_rng(91)
    result = adapter.step(state, action, supplied_rng)

    assert result.state.observation.blocks_state == str(
        int(expected_rng.integers(0, 1_000_000))
    )
    assert world.rng is supplied_rng


def test_repeated_action_accumulates_success_and_failure_outcomes():
    class AlternatingWorld(FakeWorldModel):
        def step(self, state, action):
            self.step_calls += 1
            outcome = "success" if self.step_calls % 2 else "failure"
            return BlockState(1, outcome, ""), {
                "success": outcome == "success"
            }

    class AlternatingConfig(RecordingSearchConfig):
        def __init__(self):
            super().__init__()
            self.reward_visits = []

        def fast_reward(self, node, action):
            self.fast_nodes.append(node)
            return 0.0, {}

        def reward(self, node, action, success):
            self.reward_visits.append(len(node.cum_rewards))
            return (1.0 if success else 0.0), {"success": success}

    world = AlternatingWorld()
    search_config = AlternatingConfig()
    adapter = BlockWorldAdapter(world, search_config)
    search = PlanUSearch(
        adapter,
        BlockWorldScorer(),
        blockworld_config(2, 1, 11, 0.0),
    )
    rng = np.random.default_rng(7)

    first = search.run_iteration(0, rng)
    second = search.run_iteration(1, rng)

    assert first.final_observation.blocks_state == "success"
    assert second.final_observation.blocks_state == "failure"
    assert search_config.fast_nodes[0].cum_rewards == []
    assert search_config.reward_visits == [0, 1]
    action_node = search.root.children["move"]
    assert set(action_node.children) == {
        adapter.state_key(
            EnvironmentState(BlockState(1, "success", ""), None)
        ),
        adapter.state_key(
            EnvironmentState(BlockState(1, "failure", ""), None)
        ),
    }
    assert world.step_calls == 2


def test_reward_visits_rollback_with_search_state_after_core_failure():
    class FailingSecondCoreTerminalWorld(FakeWorldModel):
        def __init__(self):
            super().__init__()
            self.terminal_checks_after_step = 0

        def step(self, state, action):
            result = super().step(state, action)
            self.terminal_checks_after_step = 0
            return result

        def is_terminal(self, state):
            if state.step_idx >= 1:
                self.terminal_checks_after_step += 1
                if (
                    self.step_calls == 2
                    and self.terminal_checks_after_step == 2
                ):
                    raise RuntimeError("core terminal validation failed")
                return self.terminal_checks_after_step >= 2
            return False

    class VisitRecordingConfig(RecordingSearchConfig):
        def __init__(self):
            super().__init__()
            self.reward_visits = []

        def reward(self, node, action, **kwargs):
            self.reward_visits.append(len(node.cum_rewards))
            return 1.0, {}

    search_config = VisitRecordingConfig()
    search = PlanUSearch(
        BlockWorldAdapter(
            FailingSecondCoreTerminalWorld(),
            search_config,
        ),
        BlockWorldScorer(),
        blockworld_config(4, 1, 11, 0.0),
    )
    rng = np.random.default_rng(29)

    first = search.run_iteration(0, rng)
    with pytest.raises(
        RuntimeError,
        match="core terminal validation failed",
    ):
        search.run_iteration(1, rng)
    retry = search.run_iteration(1, rng)
    third = search.run_iteration(2, rng)

    assert first.terminated is True
    assert retry.terminated is True
    assert third.terminated is True
    assert search_config.reward_visits == [0, 1, 1, 2]
    assert [
        search_config.reward_visits[index]
        for index in (0, 2, 3)
    ] == [0, 1, 2]


def test_scorer_order_config_values_and_public_exports():
    import planu_core
    import planu_core.adapters as adapters

    adapter = BlockWorldAdapter(FakeWorldModel(), RecordingSearchConfig())
    state = adapter.reset()
    first = adapter.actions(state, state_visit_count=1)[0]
    second = type(first)(
        key="second",
        payload="second",
        text="second",
        metadata={"prior_score": -1.25},
    )

    np.testing.assert_array_equal(
        BlockWorldScorer().score(state.observation, [second, first]),
        [-1.25, 0.75],
    )
    config = blockworld_config(
        n_iters=13,
        depth=9,
        n_atoms=17,
        risk=-0.25,
    )
    assert config.max_iterations == 13
    assert config.max_depth == 9
    assert config.n_quantiles == 17
    assert config.risk_distortion == -0.25
    assert config.quantile_learning_rate == 0.75
    assert config.value_min == -10.0
    assert config.value_max == 100.0
    assert config.discount == 1.0
    assert config.include_preview_reward is False
    assert config.selection_schedule.sample_probability_at(0) == 0.0
    assert adapters.BlockWorldAdapter is BlockWorldAdapter
    assert planu_core.BlockWorldAdapter is BlockWorldAdapter


def _load_blockworld_wrapper(monkeypatch):
    state_type = TypeVar("State")
    action_type = TypeVar("Action")
    example_type = TypeVar("Example")

    class SearchAlgorithm:
        pass

    reasoners = types.ModuleType("reasoners")
    reasoners.__path__ = [str(WRAPPER_PATH.parents[1])]
    reasoners.SearchAlgorithm = SearchAlgorithm
    reasoners.WorldModel = object
    reasoners.SearchConfig = object
    reasoners.State = state_type
    reasoners.Action = action_type
    reasoners.Example = example_type
    reasoners.Trace = tuple
    algorithm = types.ModuleType("reasoners.algorithm")
    algorithm.__path__ = [str(WRAPPER_PATH.parent)]
    monkeypatch.setitem(sys.modules, "reasoners", reasoners)
    monkeypatch.setitem(sys.modules, "reasoners.algorithm", algorithm)

    module_name = "reasoners.algorithm.planU"
    spec = importlib.util.spec_from_file_location(module_name, WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, SearchAlgorithm


class WrapperWorld(FakeWorldModel):
    def __init__(self, terminal=True):
        super().__init__()
        self.terminal = terminal
        self.rng_ids = []

    def step(self, state, action):
        self.step_calls += 1
        self.rng_ids.append(id(self.rng))
        outcome = "failure" if self.step_calls % 2 else "success"
        return BlockState(1, outcome, ""), {
            "success": outcome == "success"
        }

    def is_terminal(self, state):
        return self.terminal and state.step_idx >= 1


class WrapperSearchConfig(RecordingSearchConfig):
    def __init__(self):
        super().__init__()
        self.reward_visits = []

    def fast_reward(self, node, action):
        self.fast_nodes.append(node)
        return 0.0, {}

    def reward(self, node, action, success):
        self.reward_visits.append(len(node.cum_rewards))
        return (4.0 if success else -1.0), {"success": success}


def test_blockworld_wrapper_executes_shared_search_and_returns_best_trace(
    monkeypatch,
):
    module, SearchAlgorithm = _load_blockworld_wrapper(monkeypatch)
    world = WrapperWorld()
    search_config = WrapperSearchConfig()

    algorithm = module.PlanU(
        n_iters=2,
        depth_limit=1,
        n_atoms=7,
        v_min=-10.0,
        v_max=100.0,
        risk_distortion=0.0,
        quantile_learning_rate=0.5,
        output_trace_in_each_iter=True,
        seed=23,
        disable_tqdm=False,
    )
    result = algorithm(world, search_config)

    assert isinstance(algorithm, SearchAlgorithm)
    assert module.PlanUNode is __import__(
        "planu_core",
        fromlist=["LanguageNode"],
    ).LanguageNode
    assert module.PlanUResult._fields == (
        "terminal_state",
        "cum_reward",
        "trace",
        "trace_of_nodes",
        "tree_state",
        "trace_in_each_iter",
        "tree_state_after_each_iter",
        "aggregated_result",
    )
    assert result.terminal_state == BlockState(1, "success", "")
    assert result.cum_reward == 4.0
    assert result.trace == (
        [BlockState(0, "root", ""), BlockState(1, "success", "")],
        ["move"],
    )
    assert result.trace_of_nodes[-1].state == result.terminal_state
    assert result.tree_state is algorithm.search.root
    assert len(result.trace_in_each_iter) == 2
    assert result.tree_state_after_each_iter is None
    assert result.aggregated_result is None
    assert len(set(world.rng_ids)) == 1
    assert len(result.tree_state.children["move"].children) == 2
    assert search_config.fast_nodes[0].cum_rewards == []
    assert search_config.reward_visits == [0, 1]


def test_blockworld_wrapper_reuses_rng_across_benchmark_examples(monkeypatch):
    module, _ = _load_blockworld_wrapper(monkeypatch)

    class RandomDrawWorld(WrapperWorld):
        def step(self, state, action):
            draw = int(self.rng.integers(0, 1_000_000))
            return BlockState(1, str(draw), ""), {"success": True}

    def run_two_examples():
        algorithm = module.PlanU(n_iters=1, depth_limit=1, seed=37)
        world = RandomDrawWorld()
        config = WrapperSearchConfig()
        return [
            algorithm(world, config).terminal_state.blocks_state,
            algorithm(world, config).terminal_state.blocks_state,
        ]

    expected_rng = np.random.default_rng(37)
    expected = [
        str(int(expected_rng.integers(0, 1_000_000))),
        str(int(expected_rng.integers(0, 1_000_000))),
    ]
    first_wrapper = run_two_examples()
    second_wrapper = run_two_examples()

    assert first_wrapper == expected
    assert first_wrapper[1] != first_wrapper[0]
    assert second_wrapper == first_wrapper


def test_blockworld_wrapper_uses_best_nonterminal_and_handles_empty_path(
    monkeypatch,
):
    module, _ = _load_blockworld_wrapper(monkeypatch)
    nonterminal = module.PlanU(n_iters=2, depth_limit=1, seed=9)
    result = nonterminal(WrapperWorld(terminal=False), WrapperSearchConfig())
    assert result.cum_reward == 4.0
    assert result.trace[1] == ["move"]

    class TerminalRootWorld(WrapperWorld):
        def is_terminal(self, state):
            return True

    class NoActionsConfig(WrapperSearchConfig):
        def get_actions(self, state):
            raise AssertionError("terminal roots must not be expanded")

    empty = module.PlanU(n_iters=1, depth_limit=1, seed=9)(
        TerminalRootWorld(),
        NoActionsConfig(),
    )
    assert empty.cum_reward == 0
    assert empty.trace == ([BlockState(0, "root", "")], [])
    assert empty.trace_of_nodes == [empty.tree_state]


def test_blockworld_wrapper_rejects_unsupported_nondefault_options(
    monkeypatch,
):
    module, _ = _load_blockworld_wrapper(monkeypatch)

    module.PlanU(output_strategy="max_reward")
    with pytest.raises(ValueError, match="output_strategy"):
        module.PlanU(output_strategy="last_iter")
    with pytest.raises(ValueError, match="unknown_option"):
        module.PlanU(unknown_option=True)


def test_blockworld_wrapper_is_thin_and_python39_compatible():
    source = WRAPPER_PATH.read_text()
    tree = ast.parse(source)
    compile(source, str(WRAPPER_PATH), "exec")

    assert "PlanUSearch" in source
    assert "backup_trajectory" not in source
    method_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not {
        "_select",
        "_expand",
        "_simulate",
        "_back_propagate",
        "_chain_back_propagate",
    } & method_names
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for node in ast.walk(tree)
    )

    init_source = ALGORITHM_INIT_PATH.read_text()
    assert "PlanUAggregation" not in init_source
    assert "PlanU, PlanUNode, PlanUResult" in init_source


def _evaluate_method(source, class_name, method_name):
    tree = ast.parse(source)
    original_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = copy.deepcopy(
        next(
            node
            for node in original_class.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    probe_class = ast.ClassDef(
        name="Probe",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    probe_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            probe_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(probe_module)
    namespace = {}
    exec(compile(probe_module, str(EVALUATE_PATH), "exec"), namespace)
    return namespace["Probe"]


def _algorithm_branch(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if (
            isinstance(node.test, ast.Compare)
            and ast.unparse(node.test) == "args.algorithm == 'mcts'"
        ):
            return node
    raise AssertionError("algorithm branch not found")


def test_evaluate_cli_compiles_and_has_reachable_seeded_planu_branch():
    source = EVALUATE_PATH.read_text()
    tree = ast.parse(source)
    compile(source, str(EVALUATE_PATH), "exec")

    assert source.startswith("from __future__ import annotations\n")
    assert source.index("sys.path.insert") < source.index(
        "from reasoners import"
    )
    assert "--seed" in source and "default=100" in source
    assert "--n-iters" in source and "default=10" in source
    assert "--success-probability" in source and "default=0.8" in source
    assert "random.seed(args.seed)" in source
    assert "np.random.seed(args.seed)" in source
    assert "torch.manual_seed(args.seed)" in source
    assert "self.rng.random()" in source
    assert "success_probability=args.success_probability" in source
    assert "rng=np.random.default_rng(args.seed)" in source

    branch = _algorithm_branch(tree)
    assert len(branch.orelse) == 1
    planu_branch = branch.orelse[0]
    assert isinstance(planu_branch, ast.If)
    assert ast.unparse(planu_branch.test) == "args.algorithm == 'planu'"
    assert not any(isinstance(node, ast.Raise) for node in planu_branch.body)
    assert len(planu_branch.orelse) == 1
    assert isinstance(planu_branch.orelse[0], ast.Raise)

    calls = [
        node
        for node in ast.walk(planu_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PlanU"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert keywords["n_iters"] == "args.n_iters"
    assert keywords["seed"] == "args.seed"
    assert keywords["v_max"] == "100.0"
    assert not {
        "log_distributions",
        "distribution_log_path",
        "visualize_key_nodes",
        "chain_propagate",
    } & set(keywords)
    assert "MCTS(" in ast.unparse(branch.body)


def test_evaluate_entrypoint_imports_without_a_deepseek_model(monkeypatch):
    reasoners = types.ModuleType("reasoners")
    reasoners.__path__ = []
    reasoners.LanguageModel = object
    reasoners.Reasoner = object
    reasoners.SearchConfig = object
    reasoners.WorldModel = object

    algorithm = types.ModuleType("reasoners.algorithm")
    algorithm.MCTS = object
    algorithm.PlanU = object

    benchmark = types.ModuleType("reasoners.benchmark")
    benchmark.__path__ = []
    benchmark.BWEvaluator = object
    bw_utils = types.ModuleType("reasoners.benchmark.bw_utils")

    lm = types.ModuleType("reasoners.lm")
    lm.ExLlamaModel = object
    lm.HFModel = object

    torch = types.ModuleType("torch")

    monkeypatch.setitem(sys.modules, "reasoners", reasoners)
    monkeypatch.setitem(sys.modules, "reasoners.algorithm", algorithm)
    monkeypatch.setitem(sys.modules, "reasoners.benchmark", benchmark)
    monkeypatch.setitem(
        sys.modules,
        "reasoners.benchmark.bw_utils",
        bw_utils,
    )
    monkeypatch.setitem(sys.modules, "reasoners.lm", lm)
    monkeypatch.setitem(sys.modules, "torch", torch)

    module_name = "_blockworld_evaluate_stochastic_import_test"
    spec = importlib.util.spec_from_file_location(module_name, EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EVALUATE_PATH),
            "--algorithm",
            "planu",
            "--gpu",
            "0",
            "--version",
            "2",
            "--steps",
            "2",
        ],
    )
    assert module.parse_args().algorithm == "planu"
    assert module.benchmark_paths("v1", "2") == (
        "examples/CoT/blocksworld/prompts/pool_prompt_v1.json",
        "examples/CoT/blocksworld/data/split_v1/"
        "split_v1_step_2_data.json",
    )
    assert module.benchmark_paths("v2", "12") == (
        "examples/CoT/blocksworld/prompts/pool_prompt_v2_step_12.json",
        "examples/CoT/blocksworld/data/split_v2/"
        "split_v2_step_12_data.json",
    )
    for relative_path in module.benchmark_paths("v1", "2"):
        assert (EVALUATE_PATH.parent / relative_path).is_file()
    for relative_path in module.benchmark_paths("v2", "12"):
        assert (EVALUATE_PATH.parent / relative_path).is_file()


def test_evaluate_reward_keeps_fast_prior_and_includes_first_goal_reward():
    source = EVALUATE_PATH.read_text()
    probe_type = _evaluate_method(
        source,
        "BWConfigRAP",
        "new_calculate_reward",
    )
    probe = probe_type()
    probe.reward_alpha = 0.5
    probe.goal_reward_default = 0.0
    probe.goal_reached_reward = 100.0

    assert probe.new_calculate_reward(1.0, 2.0, 0) == 3.0
    assert probe.new_calculate_reward(
        1.0,
        2.0,
        0,
        (True, 0.0),
    ) == 101.5
    assert probe.new_calculate_reward(
        1.0,
        2.0,
        4,
        (False, 8.0),
    ) == 3.0
