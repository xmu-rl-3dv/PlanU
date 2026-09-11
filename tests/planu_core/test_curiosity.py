import copy
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import planu_core
import planu_core.search as search_module
from planu_core import PlanUConfig
from planu_core.curiosity import NullCuriosity, RndCuriosity
from planu_core.interfaces import ActionCandidate, EnvironmentState, TransitionResult
from planu_core.search import PlanUSearch
from tests.planu_core.fakes import FakeAdapter, UniformScorer


class FakeModel:
    def __init__(self, estimate_result=0.0, collect_error=None):
        self.estimate_result = estimate_result
        self.collect_error = collect_error
        self.estimated = []
        self.collected = []
        self.train_calls = 0

    def estimate(self, observation):
        self.estimated.append(observation)
        return self.estimate_result

    def collect_data(self, observation):
        if self.collect_error is not None:
            raise self.collect_error
        self.collected.append(observation)

    def train(self):
        self.train_calls += 1
        return {"loss": 0.25}


class TorchLikeScalar:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def item(self):
        self.calls.append("item")
        return self.value


class RecordingCuriosity:
    def __init__(self, scores=None, observe_error=None, train_error=None):
        self.scores = {} if scores is None else scores
        self.scored = []
        self.observed = []
        self.observe_error = observe_error
        self.train_error = train_error
        self.train_calls = 0

    def score(self, observation):
        value = int(np.asarray(observation).reshape(-1)[0])
        self.scored.append(value)
        return self.scores.get(value, 0.0)

    def observe(self, observation):
        if self.observe_error is not None:
            raise self.observe_error
        self.observed.append(observation)

    def train(self):
        self.train_calls += 1
        if self.train_error is not None:
            raise self.train_error
        return {"trained": True}


class PreviewDifferentAdapter(FakeAdapter):
    def preview(self, state, action, rng):
        self.preview_calls += 1
        state = copy.deepcopy(state)
        state.runtime["position"] += 1
        state.observation[...] = 100 + state.runtime["position"]
        return TransitionResult(
            state,
            0.0,
            state.runtime["position"] == 2,
            False,
        )


class NoveltyAdapter(FakeAdapter):
    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        return [
            ActionCandidate("left", 0, "left"),
            ActionCandidate("right", 1, "right"),
        ]

    def preview(self, state, action, rng):
        self.preview_calls += 1
        return self._move(copy.deepcopy(state), action)

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        return self._move(state, action)

    def is_terminal(self, state):
        return state.runtime["position"] == 1

    @staticmethod
    def _move(state, action):
        state.runtime["position"] = 1
        state.observation = np.array([action.payload])
        return TransitionResult(state, 0.0, True, False)


class FixedScorer:
    def score(self, observation, actions):
        return [0.6, 0.5]


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


def test_null_curiosity_has_no_side_effects():
    observation = np.array([1.0])
    curiosity = NullCuriosity()

    assert curiosity.ready_to_train is False
    assert curiosity.score(observation) == 0.0
    assert curiosity.observe(observation) is None
    assert curiosity.train() is None


@pytest.mark.parametrize("minimum_samples", [True, 1.5, "1", None])
def test_rnd_curiosity_rejects_non_integer_minimum_samples(minimum_samples):
    with pytest.raises(TypeError, match="integer"):
        RndCuriosity(FakeModel(), minimum_samples)


def test_rnd_curiosity_rejects_negative_minimum_samples():
    with pytest.raises(ValueError, match="nonnegative"):
        RndCuriosity(FakeModel(), -1)


def test_rnd_curiosity_rejects_noncallable_observation_converter():
    with pytest.raises(TypeError, match="observation_converter must be callable"):
        RndCuriosity(FakeModel(), observation_converter=object())


def test_rnd_curiosity_is_ready_only_after_exceeding_minimum_samples():
    model = FakeModel()
    curiosity = RndCuriosity(
        model,
        minimum_samples=2,
        observation_converter=lambda observation: observation,
    )

    assert curiosity.sample_count == 0
    assert curiosity.ready_to_train is False
    assert curiosity.train() is None
    curiosity.observe("first")
    curiosity.observe("second")
    assert curiosity.ready_to_train is False
    assert curiosity.train() is None
    curiosity.observe("third")
    assert curiosity.ready_to_train is True
    assert curiosity.train() == {"loss": 0.25}
    assert model.train_calls == 1


def test_rnd_curiosity_increments_count_only_after_successful_collection():
    error = RuntimeError("collection failed")
    model = FakeModel(collect_error=error)
    curiosity = RndCuriosity(
        model,
        minimum_samples=0,
        observation_converter=lambda observation: observation,
    )

    with pytest.raises(RuntimeError, match="collection failed"):
        curiosity.observe("state")

    assert curiosity.sample_count == 0
    assert curiosity.ready_to_train is False


def test_rnd_curiosity_default_converter_lazily_flattens_with_torch(monkeypatch):
    observation = np.array([[1.0, 2.0, 3.0]])
    calls = []

    class TensorSentinel:
        def reshape(self, *shape):
            calls.append(("reshape", shape))
            return self

    tensor = TensorSentinel()
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()

    def as_tensor(value, dtype):
        calls.append(("as_tensor", value, dtype))
        return tensor

    fake_torch.as_tensor = as_tensor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = FakeModel()
    curiosity = RndCuriosity(model)

    curiosity.observe(observation)

    assert calls[0][0] == "as_tensor"
    assert calls[0][1] is observation
    assert calls[0][2] is fake_torch.float32
    assert calls[1] == ("reshape", (-1,))
    assert model.collected == [tensor]
    assert curiosity.sample_count == 1


def test_rnd_curiosity_converter_failure_does_not_collect_or_increment():
    model = FakeModel()

    def fail_conversion(observation):
        raise RuntimeError("conversion failed")

    curiosity = RndCuriosity(model, observation_converter=fail_conversion)

    with pytest.raises(RuntimeError, match="conversion failed"):
        curiosity.observe(np.array([[1.0, 2.0]]))

    assert model.collected == []
    assert curiosity.sample_count == 0


def test_rnd_curiosity_converts_numeric_score_to_float():
    curiosity = RndCuriosity(FakeModel(estimate_result=3))

    score = curiosity.score("state")

    assert score == 3.0
    assert isinstance(score, float)


def test_rnd_curiosity_converts_torch_like_score_to_float():
    value = TorchLikeScalar(1.25)
    curiosity = RndCuriosity(FakeModel(estimate_result=value))

    assert curiosity.score("state") == 1.25
    assert value.calls == ["detach", "cpu", "item"]


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
def test_rnd_curiosity_rejects_nonfinite_scores(value):
    curiosity = RndCuriosity(FakeModel(estimate_result=value))

    with pytest.raises(ValueError, match="finite"):
        curiosity.score("state")


def test_search_novelty_can_change_the_selected_action():
    curiosity = RecordingCuriosity(scores={0: 0.0, 1: 2.0})
    search = PlanUSearch(
        NoveltyAdapter(),
        FixedScorer(),
        PlanUConfig(
            curiosity_weight=0.2,
            include_preview_reward=False,
            max_depth=1,
        ),
        curiosity=curiosity,
    )

    result = search.run_iteration(0, np.random.default_rng(7))

    assert result.actions == ["right"]
    assert result.selection_scores[0] == pytest.approx(
        {"left": 0.6, "right": 0.7}
    )
    assert curiosity.scored == [0, 1]


def test_search_observes_only_executed_states_in_order_then_trains_once():
    adapter = PreviewDifferentAdapter()
    curiosity = RecordingCuriosity()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=3, value_max=3.0),
        curiosity=curiosity,
    )

    result = search.run_iteration(0, np.random.default_rng(8))

    assert curiosity.scored == [101, 102]
    assert [int(observation[0]) for observation in curiosity.observed] == [1, 2]
    assert curiosity.train_calls == 1
    result.final_observation[0] = 99
    adapter.reset_state.observation[0] = 98
    assert [int(observation[0]) for observation in curiosity.observed] == [1, 2]


def test_failed_backup_does_not_update_curiosity(monkeypatch):
    curiosity = RecordingCuriosity()
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=1),
        curiosity=curiosity,
    )

    def fail_backup(actions, rewards, config):
        raise RuntimeError("backup failed")

    monkeypatch.setattr(search_module, "backup_trajectory", fail_backup)

    with pytest.raises(RuntimeError, match="backup failed"):
        search.run_iteration(0, np.random.default_rng(9))

    assert curiosity.observed == []
    assert curiosity.train_calls == 0


def test_nonfinite_curiosity_score_rolls_back_tree_journal():
    adapter = FakeAdapter()
    curiosity = RecordingCuriosity(scores={1: float("nan")})
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1),
        curiosity=curiosity,
    )
    state = adapter.reset()
    root = search._ensure_root(state)
    search.expand(root, state, np.random.default_rng(10))
    children_before = root.children.copy()

    with pytest.raises(ValueError, match="novelty.*finite"):
        search.run_iteration(0, np.random.default_rng(11))

    assert search.root is root
    assert root.visit_count == 0
    assert root.children == children_before
    assert adapter.actions_taken == []
    assert curiosity.observed == []
    assert curiosity.train_calls == 0


@pytest.mark.parametrize(
    ("adapter", "terminated", "truncated"),
    [
        (TerminalAdapter(), True, False),
        (EmptyActionsAdapter(), False, True),
    ],
)
def test_empty_trajectory_trains_once_without_observations(
    adapter,
    terminated,
    truncated,
):
    curiosity = RecordingCuriosity()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(),
        curiosity=curiosity,
    )

    result = search.run_iteration(0, np.random.default_rng(12))

    assert result.actions == []
    assert result.terminated is terminated
    assert result.truncated is truncated
    assert curiosity.scored == []
    assert curiosity.observed == []
    assert curiosity.train_calls == 1


@pytest.mark.parametrize("failure_stage", ["observe", "train"])
def test_curiosity_errors_propagate_after_tree_backup_remains_committed(
    failure_stage,
):
    error = RuntimeError("{} failed".format(failure_stage))
    curiosity = RecordingCuriosity(
        observe_error=error if failure_stage == "observe" else None,
        train_error=error if failure_stage == "train" else None,
    )
    search = PlanUSearch(
        FakeAdapter(),
        UniformScorer(),
        PlanUConfig(max_depth=1),
        curiosity=curiosity,
    )

    with pytest.raises(RuntimeError, match="failed"):
        search.run_iteration(0, np.random.default_rng(13))

    action = search.root.children["advance"]
    assert search.root.visit_count == 1
    assert action.visit_count == 1
    assert action.cumulative_returns == [1.0]
    assert action.children[((1,), 1)].outcome_visits == 1


def test_curiosity_types_are_exported_without_importing_neural_rnd():
    assert planu_core.NullCuriosity is NullCuriosity
    assert planu_core.RndCuriosity is RndCuriosity
    assert "planu_core.rnd" not in sys.modules


def test_importing_planu_core_does_not_load_torch_or_ding():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import planu_core; "
                "assert 'planu_core.rnd' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name == 'ding' or name.startswith('ding.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("relative_path", "module", "names"),
    [
        (
            "mcts/overcooked/rnd.py",
            "planu_core.rnd",
            ["RndNetwork", "RndRewardModel", "collect_states"],
        ),
        (
            "mcts/virtualhome/rnd.py",
            "planu_core.rnd",
            ["RndNetwork", "RndRewardModel", "collect_states"],
        ),
        (
            "mcts/virtualhome/base_reward_model.py",
            "planu_core.base_reward_model",
            ["BaseRewardModel"],
        ),
    ],
)
def test_compatibility_modules_reexport_their_public_names(
    tmp_path,
    relative_path,
    module,
    names,
):
    benchmark_dir = Path(relative_path).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(benchmark_dir)
    env["COMMON_MODULE"] = module
    env["PUBLIC_NAMES"] = ",".join(names)
    env["WRAPPER_MODULE"] = Path(relative_path).stem
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import os
import sys
import types
from pathlib import Path

benchmark_dir = Path(os.environ["PYTHONPATH"]).resolve()
repository_root = benchmark_dir.parents[1]
assert str(repository_root) not in sys.path, sys.path

package = types.ModuleType("planu_core")
package.__path__ = []
sys.modules["planu_core"] = package

common_module_name = os.environ["COMMON_MODULE"]
common_module = types.ModuleType(common_module_name)
public_names = os.environ["PUBLIC_NAMES"].split(",")
for name in public_names:
    setattr(common_module, name, object())
sys.modules[common_module_name] = common_module

wrapper = importlib.import_module(os.environ["WRAPPER_MODULE"])

assert sys.path[0] == str(repository_root), sys.path
assert wrapper.__all__ == public_names
for name in public_names:
    assert getattr(wrapper, name) is getattr(common_module, name)
""",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
