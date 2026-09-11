import ast
import copy
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from planu_core.adapters.overcooked import (
    OvercookedAdapter,
    describe_overcooked_observation,
    overcooked_config,
)
from planu_core.interfaces import ActionCandidate, EnvironmentState
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.search import PlanUSearch


TASK_0_INITIAL = np.array(
    [[0, 5, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float32,
)
TASK_0_AT_BOARD = np.array(
    [[1, 0, 0, 6, 5, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float32,
)
TASK_3_INITIAL = np.array(
    [
        [
            0,
            5,
            0,
            1,
            6,
            0,
            2,
            6,
            0,
            6,
            5,
            0,
            1,
            0,
            2,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
    ],
    dtype=np.float32,
)
TASK_3_AT_BOARD = TASK_3_INITIAL.copy()
TASK_3_AT_BOARD[0, :3] = [1, 0, 0]
# Frozen after all generated cases matched the final active LLMAgent.obs2text
# method from fac4fd3^ in an independent one-off differential run.
LEGACY_PARITY_SHA256 = (
    "32160fdc84619214ee1ee2f3631729c1f1c445c9052bb24c9092e4dfcefc9230"
)


def _item_position(location, original, agent):
    return {
        "origin": original,
        "board_1": (1, 0),
        "board_2": (2, 0),
        "hand": agent,
        "off_grid": (9, 9),
    }[location]


def _overcooked_parity_states():
    """Yield 18 task-0 and 3,600 task-3 deterministic observations.

    Task 0 bounds are 3 tomato locations x 3 bowl locations x 2 agent
    locations. Task 3 bounds are 5 tomato x 5 lettuce x 4 onion x 3 bowl
    locations x 3 agent locations x 4 chop profiles.
    """
    for tomato, bowl, agent in itertools.product(
        ("origin", "board_1", "hand"),
        ("origin", "board_1", "hand"),
        ((1, 1), (0, 0)),
    ):
        observation = np.zeros(18, dtype=np.float32)
        observation[0:2] = _item_position(tomato, (0, 5), agent)
        observation[3:5] = _item_position(bowl, (6, 5), agent)
        observation[9:11] = agent
        yield 0, observation

    chop_profiles = (
        (0, 0, 0),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
    )
    for tomato, lettuce, onion, bowl, agent, chopped in itertools.product(
        ("origin", "board_1", "board_2", "hand", "off_grid"),
        ("origin", "board_1", "board_2", "hand", "off_grid"),
        ("origin", "board_1", "board_2", "hand"),
        ("origin", "board_1", "hand"),
        ((1, 1), (2, 1), (0, 0)),
        chop_profiles,
    ):
        observation = np.zeros(26, dtype=np.float32)
        for index, (location, original) in enumerate(
            zip(
                (tomato, lettuce, onion, bowl),
                ((0, 5), (1, 6), (2, 6), (6, 5)),
            )
        ):
            observation[3 * index : 3 * index + 2] = _item_position(
                location,
                original,
                agent,
            )
        observation[2], observation[5], observation[8] = chopped
        observation[17:19] = agent
        yield 3, observation


class FakeInnerEnv:
    def __init__(self, step_count=0):
        self.step_count = step_count

    @property
    def unwrapped(self):
        return self


class FakeVectorEnv:
    def __init__(
        self,
        initial_observation=TASK_0_INITIAL,
        reward=0.25,
        done=False,
        num_envs=1,
        accepts_seed=True,
    ):
        self.initial_observation = np.array(initial_observation, copy=True)
        self.current_observation = np.array(initial_observation, copy=True)
        self.reward = reward
        self.step_done = done
        self.num_envs = num_envs
        self.accepts_seed = accepts_seed
        self.reset_calls = []
        self.step_calls = []
        self.envs = [FakeInnerEnv()]

    def reset(self, **kwargs):
        if kwargs and not self.accepts_seed:
            raise TypeError("reset() got an unexpected keyword argument 'seed'")
        self.reset_calls.append(kwargs.get("seed"))
        self.current_observation = np.array(self.initial_observation, copy=True)
        self.envs[0].step_count = 0
        return self.current_observation

    def step(self, action):
        self.step_calls.append(np.array(action, copy=True))
        self.envs[0].step_count += 1
        self.current_observation = np.array(self.current_observation, copy=True)
        self.current_observation[0, -1] += 1
        return (
            self.current_observation,
            np.array([self.reward]),
            np.array([self.step_done]),
            [{"macro_action_steps": 1}],
        )


class FixedRng:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


class UniformScorer:
    def score(self, observation, candidates):
        return np.full(len(candidates), 1.0 / len(candidates))


def test_task_zero_prompt_and_action_order_match_reference_fixture():
    description = describe_overcooked_observation(TASK_0_INITIAL[0], task=0)

    assert description == {
        "prompt": (
            "There is a fixed cutting board in the room. "
            "You notice a tomato and a bowl on the different tables. "
            "Currently you don't have anything in hand. "
            "Your goal is to serve the dish of a bowl only containing chopped "
            "tomato. Your next step is to"
        ),
        "action": [
            "pick up the tomato",
            "take the bowl",
            "walk to the cutting board",
            "serve nothing",
            "chop nothing",
        ],
    }


def test_task_three_prompt_and_action_order_match_reference_fixture():
    description = describe_overcooked_observation(TASK_3_INITIAL[0], task=3)

    assert description == {
        "prompt": (
            "There are two fixed cutting boards in the room. Earlier this day, "
            "you made a dish with chopped potatoes. You notice a tomato, a "
            "lettuce, an onion and a bowl on the different tables. Currently "
            "you are standing in front of the first cutting board without "
            "anything in hand. Your goal is to serve the dish of a bowl only "
            "containing chopped tomato and lettuce. Your next step is to"
        ),
        "action": [
            "pick up the tomato",
            "pick up the lettuce",
            "pick up the onion",
            "take the empty bowl",
            "walk to the first cutting board",
            "walk to the second cutting board",
            "serve nothing",
            "chop nothing",
        ],
    }


def test_generated_observations_match_active_legacy_prompt_action_digest():
    records = []
    for task, observation in _overcooked_parity_states():
        description = describe_overcooked_observation(observation, task)
        records.append(
            {
                "task": task,
                "prompt": description["prompt"],
                "actions": description["action"],
            }
        )

    serialized = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
    )

    assert len(records) == 3618
    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == (
        LEGACY_PARITY_SHA256
    )


@pytest.mark.parametrize(
    ("task", "observation", "expected_sentence"),
    [
        (
            0,
            TASK_0_AT_BOARD[0],
            "An unchopped tomato is on the cutting board.",
        ),
        (
            3,
            TASK_3_AT_BOARD[0],
            "An unchopped tomato is on the first cutting board.",
        ),
    ],
)
def test_board_observation_changes_prompt_and_chop_action(
    task,
    observation,
    expected_sentence,
):
    description = describe_overcooked_observation(observation, task)

    assert expected_sentence in description["prompt"]
    assert description["action"][-1] == "chop the tomato"


def test_prompt_disturb_and_shuffle_are_explicit_and_rng_driven():
    class ReverseRng:
        def __init__(self):
            self.calls = 0

        def shuffle(self, values):
            self.calls += 1
            values.reverse()

    rng = ReverseRng()
    description = describe_overcooked_observation(
        TASK_3_INITIAL[0],
        task=3,
        prompt_disturb=False,
        prompt_shuffle=True,
        rng=rng,
    )

    assert rng.calls == 1
    assert "chopped potatoes" not in description["prompt"]
    assert description["prompt"].endswith("Your next step is to")
    assert description["prompt"].startswith(
        "Your goal is to serve the dish of a bowl only containing"
    )


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="task must be 0 or 3"):
        describe_overcooked_observation(TASK_0_INITIAL[0], task=2)


def test_overcooked_config_exact_values_and_task_zero_schedule():
    config = overcooked_config(
        task=0,
        rnd=True,
        max_iterations=1000,
        max_depth=15,
        temperature=0.75,
    )

    assert config.quantile_learning_rate == 0.75
    assert config.curiosity_weight == 0.5
    assert config.include_preview_reward is True
    assert config.max_iterations == 1000
    assert config.max_depth == 15
    assert config.selection_temperature == 1.0
    assert config.selection_schedule.always_sample_before == 50
    assert config.selection_schedule.probabilistic_sample_before == 100
    assert config.selection_schedule.sample_probability == 0.2
    assert config.selection_schedule.sample_probability_at(49) == 1.0
    assert config.selection_schedule.sample_probability_at(50) == 0.2
    assert config.selection_schedule.sample_probability_at(99) == 0.2
    assert config.selection_schedule.sample_probability_at(100) == 0.0


def test_task_three_config_is_greedy_and_disables_curiosity():
    config = overcooked_config(3, False, 7, 4, 1.5)

    assert config.curiosity_weight == 0.0
    assert config.selection_schedule.sample_probability_at(0) == 0.0
    assert config.max_iterations == 7
    assert config.max_depth == 4
    assert config.selection_temperature == 1.0


def test_overcooked_config_defaults_selection_temperature_to_one():
    config = overcooked_config(
        task=0,
        rnd=False,
        max_iterations=3,
        max_depth=2,
    )

    assert config.selection_temperature == 1.0


def test_adapter_rejects_bad_task_probability_and_vector_count():
    env = FakeVectorEnv()
    with pytest.raises(ValueError, match="task must be 0 or 3"):
        OvercookedAdapter(env, 1, 0.2, np.random.default_rng(1))
    with pytest.raises(ValueError, match="between 0 and 1"):
        OvercookedAdapter(env, 0, 1.1, np.random.default_rng(1))
    with pytest.raises(ValueError, match="exactly one"):
        OvercookedAdapter(
            FakeVectorEnv(num_envs=2),
            0,
            0.2,
            np.random.default_rng(1),
        )


def test_reset_calls_vector_env_once_and_copies_observation():
    env = FakeVectorEnv()
    adapter = OvercookedAdapter(env, 0, 0.2, np.random.default_rng(1))

    state = adapter.reset(seed=17)

    assert env.reset_calls == [17]
    assert state.runtime is env
    assert state.observation is not env.current_observation
    np.testing.assert_array_equal(state.observation, env.current_observation)


def test_reset_falls_back_for_gym_021_without_double_successful_reset():
    env = FakeVectorEnv(accepts_seed=False)
    adapter = OvercookedAdapter(env, 0, 0.2, np.random.default_rng(1))

    adapter.reset(seed=17)

    assert env.reset_calls == [None]


@pytest.mark.parametrize(
    ("accepts_seed", "expected_reset_calls"),
    [(True, [17]), (False, [None])],
)
def test_reset_clears_terminal_marker_after_successful_env_reset(
    accepts_seed,
    expected_reset_calls,
):
    env = FakeVectorEnv(accepts_seed=accepts_seed)
    env._planu_terminated = True
    adapter = OvercookedAdapter(env, 0, 0.2, np.random.default_rng(1))

    state = adapter.reset(seed=17)

    assert state.runtime is env
    assert state.runtime._planu_terminated is False
    assert adapter.is_terminal(state) is False
    assert env.reset_calls == expected_reset_calls


def test_reset_failure_preserves_terminal_marker():
    class FailingFallbackEnv(FakeVectorEnv):
        def reset(self, **kwargs):
            if kwargs:
                raise TypeError(
                    "reset() got an unexpected keyword argument 'seed'"
                )
            raise RuntimeError("reset failed")

    env = FailingFallbackEnv()
    env._planu_terminated = True
    adapter = OvercookedAdapter(env, 0, 0.2, np.random.default_rng(1))

    with pytest.raises(RuntimeError, match="reset failed"):
        adapter.reset(seed=17)

    assert env._planu_terminated is True


def test_actions_use_first_vector_observation_and_share_prompt_metadata():
    adapter = OvercookedAdapter(
        FakeVectorEnv(),
        0,
        0.2,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    candidates = adapter.actions(state)

    assert [candidate.key for candidate in candidates] == list(range(5))
    assert [candidate.payload for candidate in candidates] == list(range(5))
    assert [candidate.text for candidate in candidates] == [
        "pick up the tomato",
        "take the bowl",
        "walk to the cutting board",
        "serve nothing",
        "chop nothing",
    ]
    prompts = {candidate.metadata["prompt"] for candidate in candidates}
    assert prompts == {
        describe_overcooked_observation(TASK_0_INITIAL[0], 0)["prompt"]
    }


def test_clone_and_preview_do_not_mutate_original_state():
    env = FakeVectorEnv()
    adapter = OvercookedAdapter(env, 0, 1.0, np.random.default_rng(1))
    state = adapter.reset()
    state_before = copy.deepcopy(state)
    cloned = adapter.clone(state)
    chop = adapter.actions(state)[-1]

    result = adapter.preview(cloned, chop, FixedRng(0.0))

    np.testing.assert_array_equal(state.observation, state_before.observation)
    assert state.runtime.envs[0].step_count == 0
    assert cloned.runtime.envs[0].step_count == 0
    assert result.state.runtime is not cloned.runtime
    assert result.state.runtime.envs[0].step_count == 1
    assert result.reward == 0.25
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["raw_info"] == [{"macro_action_steps": 1}]


def test_preview_directly_clones_the_supplied_state_before_stepping():
    env = FakeVectorEnv()
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    state = adapter.reset()
    supplied_observation = state.observation
    supplied_runtime = state.runtime
    before = copy.deepcopy(state)

    result = adapter.preview(
        state,
        adapter.actions(state)[0],
        np.random.default_rng(2),
    )

    assert state.observation is supplied_observation
    assert state.runtime is supplied_runtime
    np.testing.assert_array_equal(state.observation, before.observation)
    assert state.runtime.envs[0].step_count == 0
    assert state.runtime.step_calls == []
    assert result.state.runtime is not supplied_runtime
    assert result.state.runtime.envs[0].step_count == 1


def test_chop_failure_returns_pre_step_snapshot_and_penalty():
    env = FakeVectorEnv()
    adapter_rng = FixedRng(0.9)
    step_rng = FixedRng(0.1)
    adapter = OvercookedAdapter(env, 0, 0.5, adapter_rng)
    state = adapter.reset()
    chop = adapter.actions(state)[-1]

    result = adapter.step(state, chop, step_rng)

    assert step_rng.calls == 1
    assert adapter_rng.calls == 0
    np.testing.assert_array_equal(result.state.observation, TASK_0_INITIAL)
    assert result.state.runtime.envs[0].step_count == 0
    assert result.reward == -0.001
    assert result.terminated is False
    assert result.truncated is False
    assert result.state.runtime._planu_terminated is False
    assert adapter.is_terminal(result.state) is False
    assert len(env.step_calls) == 1


def test_chop_success_uses_actual_transition_and_non_chop_does_not_draw():
    env = FakeVectorEnv(reward=0.75, done=True)
    rng = FixedRng(0.9)
    adapter = OvercookedAdapter(env, 0, 0.5, rng)
    state = adapter.reset()

    chop_result = adapter.step(state, adapter.actions(state)[-1], rng)

    assert rng.calls == 1
    assert chop_result.state.runtime.envs[0].step_count == 1
    assert chop_result.reward == 0.75
    assert chop_result.terminated is True
    assert chop_result.state.runtime._planu_terminated is True
    assert adapter.is_terminal(chop_result.state) is True

    non_chop = adapter.actions(chop_result.state)[0]
    adapter.step(chop_result.state, non_chop, rng)
    assert rng.calls == 1


def test_step_uses_supplied_rng_instead_of_adapter_default_rng():
    env = FakeVectorEnv()
    adapter_rng = FixedRng(0.0)
    supplied_rng = FixedRng(1.0)
    adapter = OvercookedAdapter(env, 0, 0.5, adapter_rng)
    state = adapter.reset()

    result = adapter.step(state, adapter.actions(state)[-1], supplied_rng)

    assert supplied_rng.calls == 1
    assert adapter_rng.calls == 0
    assert result.reward == 0.25


def test_state_key_includes_runtime_step_count_without_identity():
    adapter = OvercookedAdapter(
        FakeVectorEnv(),
        0,
        0.0,
        np.random.default_rng(1),
    )
    first = adapter.reset()
    second = adapter.clone(first)
    second.runtime.envs[0].step_count = 4

    assert adapter.state_key(first) != adapter.state_key(second)
    assert adapter.state_key(first) == adapter.state_key(adapter.clone(first))


def test_state_key_scans_env_step_and_elapsed_steps_at_all_runtime_levels():
    class UnwrappedEnv:
        def __init__(self):
            self.env_step = 0

    class WrappedEnv:
        def __init__(self):
            self._elapsed_steps = 0
            self.unwrapped = UnwrappedEnv()

    env = FakeVectorEnv()
    env.env_step = 0
    env.envs = [WrappedEnv()]
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    observation = np.array(TASK_0_INITIAL, copy=True)

    baseline = EnvironmentState(observation, copy.deepcopy(env))
    wrapper_changed = EnvironmentState(observation, copy.deepcopy(env))
    wrapper_changed.runtime.env_step = 1
    inner_changed = EnvironmentState(observation, copy.deepcopy(env))
    inner_changed.runtime.envs[0]._elapsed_steps = 1
    unwrapped_changed = EnvironmentState(observation, copy.deepcopy(env))
    unwrapped_changed.runtime.envs[0].unwrapped.env_step = 1

    keys = {
        adapter.state_key(state)
        for state in (
            baseline,
            wrapper_changed,
            inner_changed,
            unwrapped_changed,
        )
    }
    assert len(keys) == 4


def test_is_terminal_defaults_false_and_reads_explicit_runtime_state():
    adapter = OvercookedAdapter(
        FakeVectorEnv(),
        0,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()
    assert adapter.is_terminal(state) is False

    state.runtime.envs[0].terminated = True
    assert adapter.is_terminal(state) is True


def test_preview_persists_done_as_private_runtime_terminal_marker():
    env = FakeVectorEnv(done=True)
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    state = adapter.reset()

    result = adapter.preview(
        state,
        adapter.actions(state)[0],
        np.random.default_rng(2),
    )

    assert result.terminated is True
    assert result.state.runtime._planu_terminated is True
    assert adapter.is_terminal(result.state) is True


def test_nonfinite_vector_reward_is_rejected():
    env = FakeVectorEnv(reward=math.inf)
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    state = adapter.reset()

    with pytest.raises(ValueError, match="reward must be finite"):
        adapter.preview(
            adapter.clone(state),
            adapter.actions(state)[0],
            np.random.default_rng(2),
        )


def test_adapter_and_shared_search_smoke():
    env = FakeVectorEnv(done=True)
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        overcooked_config(0, False, 1, 2, 1.0),
    )

    result = search.run_iteration(0, np.random.default_rng(7))

    assert len(result.actions) == 1
    assert result.terminated is True
    assert len(search.root.children) == 5
    assert env.reset_calls == [None]


def test_shared_search_executes_action_after_terminal_iteration_reset():
    env = FakeVectorEnv(done=True)
    adapter = OvercookedAdapter(env, 0, 0.0, np.random.default_rng(1))
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        overcooked_config(0, False, 2, 2, 1.0),
    )

    first = search.run_iteration(0, np.random.default_rng(7))
    second = search.run_iteration(1, np.random.default_rng(8))

    assert first.terminated is True
    assert second.terminated is True
    assert len(first.actions) == 1
    assert len(second.actions) == 1
    assert len(env.step_calls) == 2
    assert env.reset_calls == [None, None]


def test_scorer_module_imports_without_optional_ml_dependencies():
    module = importlib.import_module("mcts.overcooked.PlanU_mcts")

    assert hasattr(module, "OvercookedActionScorer")
    assert hasattr(module, "normalize_action_scores")


def test_scorer_module_reexports_shared_search_types():
    module = importlib.import_module("mcts.overcooked.PlanU_mcts")

    assert module.ActionNode is ActionNode
    assert module.LanguageNode is LanguageNode
    assert module.PlanUSearch is PlanUSearch


def test_scorer_module_defines_no_search_algorithm_classes():
    source_path = Path("mcts/overcooked/PlanU_mcts.py")
    tree = ast.parse(source_path.read_text())
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    assert "LanguageNode" not in class_names
    assert "ActionNode" not in class_names
    assert "LLMAgent" not in class_names


def test_backend_independent_score_normalization_and_validation():
    module = importlib.import_module("mcts.overcooked.PlanU_mcts")

    token = module.normalize_action_scores(
        [-4.0, -3.0],
        [2, 1],
        ["two words", "one"],
        "token",
        1.0,
    )
    word = module.normalize_action_scores(
        [-4.0, -3.0],
        [2, 1],
        ["two words", "one"],
        "word",
        1.0,
    )
    summed = module.normalize_action_scores(
        [-4.0, -3.0],
        [2, 1],
        ["two words", "one"],
        "sum",
        1.0,
    )

    np.testing.assert_allclose(token, word)
    assert token[0] > summed[0]
    assert np.sum(token) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="normalization_mode"):
        module.normalize_action_scores([0], [1], ["a"], "bad", 1.0)
    with pytest.raises(ValueError, match="temperature"):
        module.normalize_action_scores([0], [1], ["a"], "token", 0.0)


def test_scorer_honors_injected_model_name_and_starts_counters_at_zero():
    module = importlib.import_module("mcts.overcooked.PlanU_mcts")
    scorer = module.OvercookedActionScorer(
        "custom/model",
        tokenizer=object(),
        model=object(),
        device="cpu",
    )

    assert scorer.base_model == "custom/model"
    assert scorer.total_llm_tokenizer_token == 0
    assert scorer.total_llm_tokenizer_call == 0


def test_scorer_teacher_forcing_keeps_tensor_devices_aligned(monkeypatch):
    class FakeTensor:
        def __init__(self, values, device="cpu"):
            self.values = np.asarray(values)
            self.device = device

        @property
        def shape(self):
            return self.values.shape

        def to(self, device):
            return FakeTensor(self.values, device)

        def __sub__(self, other):
            if isinstance(other, FakeTensor):
                if self.device != other.device:
                    raise RuntimeError("tensor device mismatch")
                other = other.values
            return FakeTensor(self.values - other, self.device)

        def __getitem__(self, index):
            return FakeTensor(self.values[index], self.device)

        def squeeze(self, axis):
            return FakeTensor(np.squeeze(self.values, axis=axis), self.device)

        def sum(self):
            return FakeTensor(self.values.sum(), self.device)

        def detach(self):
            return self

        def cpu(self):
            return FakeTensor(self.values, "cpu")

        def item(self):
            return self.values.item()

        def tolist(self):
            return self.values.tolist()

    class FakeTokenizer:
        def __call__(self, values, return_tensors, padding):
            assert return_tensors == "pt"
            assert padding is True
            lengths = [5, 4] if values[0].startswith("prompt ") else [3, 2]
            width = max(lengths)
            mask = [
                [1] * length + [0] * (width - length)
                for length in lengths
            ]
            return {
                "input_ids": FakeTensor(np.ones((2, width), dtype=int)),
                "attention_mask": FakeTensor(mask),
            }

    class FakeModel:
        def __call__(self, input_ids, attention_mask):
            assert input_ids.device == "cuda"
            assert attention_mask.device == "cuda"
            return types.SimpleNamespace(
                logits=FakeTensor(np.zeros((2, 5, 8)), "cuda")
            )

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.sum = lambda tensor, dim: FakeTensor(
        tensor.values.sum(axis=dim),
        tensor.device,
    )
    fake_torch.no_grad = NoGrad
    fake_torch.log_softmax = lambda tensor, dim: tensor
    fake_torch.gather = lambda tensor, dim, index: FakeTensor(
        -np.ones(index.shape),
        tensor.device,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module = importlib.import_module("mcts.overcooked.PlanU_mcts")
    scorer = module.OvercookedActionScorer(
        "custom/model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cuda",
    )
    candidates = [
        ActionCandidate(0, 0, "two words", {"prompt": "prompt"}),
        ActionCandidate(1, 1, "one", {"prompt": "prompt"}),
    ]

    probabilities = scorer.score(np.array([0]), candidates)

    np.testing.assert_allclose(probabilities, [0.5, 0.5])
    assert scorer.total_llm_tokenizer_token == 10
    assert scorer.total_llm_tokenizer_call == 1


def test_cli_parser_imports_without_gym_and_parses_boolean_strings():
    inference = importlib.import_module("mcts.overcooked.PlanU_inference")

    args = inference.parse_args(
        [
            "--num-envs",
            "1",
            "--rnd",
            "False",
            "--init_dist",
            "false",
            "--transpositions",
            "True",
        ]
    )

    assert args.rnd is False
    assert args.init_dist is False
    assert args.transpositions is True


def test_cli_temperature_only_configures_action_scorer(monkeypatch):
    inference = importlib.import_module("mcts.overcooked.PlanU_inference")
    args = inference.parse_args(["--temperature", "0.37"])
    calls = {}

    class RecordingScorer:
        def __init__(self, *args, **kwargs):
            calls["scorer_temperature"] = kwargs["temperature"]

    class RecordingAdapter:
        def __init__(self, *args, **kwargs):
            pass

    class RecordingSearch:
        def __init__(self, adapter, scorer, config, curiosity):
            calls["search"] = (adapter, scorer, config, curiosity)

    def recording_config(**kwargs):
        calls["config_kwargs"] = kwargs
        return types.SimpleNamespace(selection_temperature=1.0)

    monkeypatch.setattr(inference, "OvercookedActionScorer", RecordingScorer)
    monkeypatch.setattr(inference, "OvercookedAdapter", RecordingAdapter)
    monkeypatch.setattr(inference, "PlanUSearch", RecordingSearch)
    monkeypatch.setattr(inference, "overcooked_config", recording_config)

    _, _, config = inference.build_planu_components(
        args,
        envs=object(),
        device="cpu",
        rnd_writer=object(),
    )

    assert calls["scorer_temperature"] == 0.37
    assert "temperature" not in calls["config_kwargs"]
    assert config.selection_temperature == 1.0


def test_cli_rejects_multiple_envs_and_init_distribution():
    inference = importlib.import_module("mcts.overcooked.PlanU_inference")

    with pytest.raises(AssertionError, match="num_envs"):
        inference.validate_args(inference.parse_args(["--num-envs", "2"]))
    with pytest.raises(NotImplementedError, match="init_dist"):
        inference.validate_args(
            inference.parse_args(["--num-envs", "1", "--init_dist", "True"])
        )


def test_existing_direct_script_command_can_parse_without_gym():
    completed = subprocess.run(
        [
            sys.executable,
            "mcts/overcooked/PlanU_inference.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--maxiterations" in completed.stdout
