import ast
import copy
import hashlib
import importlib
import itertools
import json
import logging
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from planu_core.adapters.virtualhome import (
    VirtualHomeAdapter,
    VirtualHomeTask,
    describe_virtualhome_observation,
    virtualhome_config,
)
from planu_core.interfaces import EnvironmentState
from planu_core.provenance import build_effective_config
from planu_core.scorers import (
    ConstantActionScorer,
    HuggingFaceActionScorer,
)
from planu_core.search import PlanUSearch


FOOD_INITIAL = np.array(
    [[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]],
    dtype=np.float32,
)
FOOD_AT_MICROWAVE = np.array(
    [[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]],
    dtype=np.float32,
)
ENTERTAINMENT_INITIAL = np.array(
    [[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]],
    dtype=np.float32,
)
ENTERTAINMENT_AT_CHIPS = np.array(
    [[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float32,
)

# Frozen from independent AST execution of the final active legacy obs2text
# methods at PlanU_v1.py:954 and PlanU_v2.py:1421 on commit 46fbf9a.
FOOD_PARITY_CASES = 512
FOOD_PARITY_SHA256 = (
    "ece4097491bc965e08c3dcfea17adb6eae3286b8baa706aae49d8535230beb59"
)
ENTERTAINMENT_PARITY_CASES = 8456
ENTERTAINMENT_PARITY_SHA256 = (
    "2a701e9d1f1224c99470929f214054e6d4507a4ebb7b5e99845efa539c3facc6"
)


@pytest.mark.parametrize(
    "relative_path",
    [
        "virtual-home/virtual_home/envs/graph_env_v1.py",
        "virtual-home/virtual_home/envs/graph_env_v2.py",
        (
            "virtual-home/virtual_home/simulation/environment/"
            "unity_environment.py"
        ),
    ],
)
def test_graph_environments_do_not_require_ipdb_at_import_time(relative_path):
    tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "ipdb" not in imported_modules


def _food_cases():
    for room in range(4):
        for tail in itertools.product((0, 1), repeat=7):
            observation = [0] * 11
            observation[room] = 1
            observation[4:] = tail
            yield observation


def _entertainment_cases():
    for item_state in itertools.product((0, 1), repeat=8):
        observation = [0] * 21
        observation[0] = 1
        observation[4:12] = item_state
        yield observation

    varying_indices = (4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 18)
    close_indices = (13, 17, 20)
    for state in itertools.product((0, 1), repeat=len(varying_indices)):
        for close_target in range(4):
            observation = [0] * 21
            observation[3] = 1
            for index, value in zip(varying_indices, state):
                observation[index] = value
            observation[12] = observation[16] = observation[19] = 1
            if close_target:
                observation[close_indices[close_target - 1]] = 1
            yield observation

    for room in (1, 2):
        for hold_chips, hold_milk in itertools.product((0, 1), repeat=2):
            observation = [0] * 21
            observation[room] = 1
            observation[6] = hold_chips
            observation[10] = hold_milk
            yield observation


def _digest_cases(task, cases):
    records = []
    for observation in cases:
        prompt, actions = describe_virtualhome_observation(observation, task)
        records.append(
            {
                "observation": observation,
                "prompt": prompt,
                "actions": actions,
            }
        )
    serialized = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return len(records), hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FakeInnerEnv:
    def __init__(self):
        self.steps = 0
        self.env_step = 0
        self._elapsed_steps = 0
        self.first_grab_chips_flag = True
        self.first_grab_milk_flag = True
        self.pre_tv_state = 0

    @property
    def unwrapped(self):
        return self


class FakeVectorEnv:
    def __init__(
        self,
        initial_observation,
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
        inner = self.envs[0]
        inner.steps = 0
        inner.env_step = 0
        inner._elapsed_steps = 0
        inner.first_grab_chips_flag = True
        inner.first_grab_milk_flag = True
        inner.pre_tv_state = 0
        return self.current_observation

    def step(self, action):
        self.step_calls.append(np.array(action, copy=True))
        inner = self.envs[0]
        inner.steps += 1
        inner.env_step += 1
        inner._elapsed_steps += 1
        inner.first_grab_chips_flag = False
        inner.first_grab_milk_flag = False
        inner.pre_tv_state = 1
        return (
            np.array(self.current_observation, copy=True),
            np.array([self.reward]),
            np.array([self.step_done]),
            [{"step": inner.steps}],
        )


class GraphEnvironment:
    __module__ = "virtual_home.envs.graph_environment_v2"

    def __init__(
        self,
        observation,
        reward,
        max_episode_length=50,
        info=None,
    ):
        self.observation = np.asarray(observation).reshape(-1)
        self.reward = reward
        self.steps = max_episode_length - 1
        self.max_episode_length = max_episode_length
        self.info = {} if info is None else dict(info)

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        del action
        self.steps += 1
        done = self.steps >= self.max_episode_length
        return (
            np.array(self.observation, copy=True),
            self.reward,
            done,
            dict(self.info),
        )


class FakeGymWrapper:
    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        return self.env.step(action)


class FakeVirtualHomeHorizonVectorEnv:
    num_envs = 1

    def __init__(self, observation, reward, info=None):
        self.initial_observation = np.array(observation, copy=True)
        self.reward = reward
        self.info = info
        self.envs = []
        self.reset_calls = []
        self.step_calls = []
        self._new_inner()

    def _new_inner(self):
        self.envs = [
            FakeGymWrapper(
                GraphEnvironment(
                    self.initial_observation[0],
                    self.reward,
                    info=self.info,
                )
            )
        ]

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs.get("seed"))
        self._new_inner()
        return np.array(self.initial_observation, copy=True)

    def step(self, action):
        self.step_calls.append(np.array(action, copy=True))
        observation, reward, done, info = self.envs[0].step(action[0])
        return (
            observation[np.newaxis, :],
            np.array([reward]),
            np.array([done]),
            [info],
        )


class AutoResetVirtualHomeHorizonVectorEnv(
    FakeVirtualHomeHorizonVectorEnv
):
    def step(self, action):
        result = super().step(action)
        if bool(result[2][0]):
            self._new_inner()
            self.envs[0].env.steps = 0
        return result


class FixedRng:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def test_task_enum_exact_reference_values():
    assert VirtualHomeTask.FOOD.env_id == "VirtualHome-v1"
    assert VirtualHomeTask.FOOD.stochastic_verb == "open"
    assert VirtualHomeTask.FOOD.uses_llm_prior is False
    assert VirtualHomeTask.FOOD.curiosity_weight == 0.25
    assert VirtualHomeTask.FOOD.obs_shape == 11
    assert VirtualHomeTask.ENTERTAINMENT.env_id == "VirtualHome-v2"
    assert VirtualHomeTask.ENTERTAINMENT.stochastic_verb == "grab"
    assert VirtualHomeTask.ENTERTAINMENT.uses_llm_prior is True
    assert VirtualHomeTask.ENTERTAINMENT.curiosity_weight == 0.1
    assert VirtualHomeTask.ENTERTAINMENT.obs_shape == 21


@pytest.mark.parametrize(
    ("task", "rnd", "expected_curiosity"),
    [
        (VirtualHomeTask.FOOD, True, 0.25),
        (VirtualHomeTask.ENTERTAINMENT, True, 0.1),
        (VirtualHomeTask.FOOD, False, 0.0),
        (VirtualHomeTask.ENTERTAINMENT, False, 0.0),
    ],
)
def test_virtualhome_config_exact_values(task, rnd, expected_curiosity):
    config = virtualhome_config(task, rnd, max_iterations=17, max_depth=9)

    assert config.quantile_learning_rate == 0.7
    assert config.curiosity_weight == expected_curiosity
    assert config.train_curiosity is False
    assert config.include_preview_reward is True
    assert config.selection_temperature == 1.0
    assert config.max_iterations == 17
    assert config.max_depth == 9
    assert config.selection_schedule.always_sample_before == 0
    assert config.selection_schedule.probabilistic_sample_before == 0
    assert config.selection_schedule.sample_probability == 0.0


def test_food_representative_prompt_and_legal_action_ids():
    prompt, actions = describe_virtualhome_observation(
        FOOD_INITIAL[0],
        VirtualHomeTask.FOOD,
    )

    assert prompt == (
        "There are four rooms: the kitchen, bathroom, bedroom, and living "
        "room. You are in the kitchen. You notice pancake and microwave. "
        "Currently, you are not grabbing anything in hand. The pancake is "
        "close to you. The microwave is not opend. In order to heat up the "
        "pancake in the microwave, your next step is to"
    )
    assert actions == [
        (0, "walk to the living room"),
        (2, "walk to the bathroom"),
        (3, "walk to the bedroom"),
        (5, "walk to the microwave"),
        (6, "grab the pancake"),
    ]


def test_entertainment_representative_prompt_and_legal_action_ids():
    prompt, actions = describe_virtualhome_observation(
        ENTERTAINMENT_INITIAL[0],
        VirtualHomeTask.ENTERTAINMENT,
    )

    assert prompt == (
        "There are four rooms: the kitchen, bathroom, bedroom, and living "
        "room. You are in the living room. Earlier in the day, you were in the "
        "bedroom and sowe cookies on the table.and you notice a coffee table, "
        "a TV and a sofa. They are not close to you. Currently, you are not "
        "grabbing anything in hand. Your goal is to enjoy the chips and the "
        "milk while watching TV. Your next step is to"
    )
    assert actions == [
        (1, "walk to the kitchen"),
        (2, "walk to the bathroom"),
        (3, "walk to the bedroom"),
        (6, "walk to the coffee table"),
        (7, "walk to the TV"),
        (8, "walk to the sofa"),
    ]


def test_food_generated_observations_match_active_legacy_digest():
    count, digest = _digest_cases(
        VirtualHomeTask.FOOD,
        _food_cases(),
    )

    assert count == FOOD_PARITY_CASES
    assert digest == FOOD_PARITY_SHA256


def test_entertainment_generated_observations_match_active_legacy_digest():
    count, digest = _digest_cases(
        VirtualHomeTask.ENTERTAINMENT,
        _entertainment_cases(),
    )

    assert count == ENTERTAINMENT_PARITY_CASES
    assert digest == ENTERTAINMENT_PARITY_SHA256


def test_entertainment_prompt_options_are_explicit_and_rng_driven():
    class ReverseRng:
        def __init__(self):
            self.calls = 0

        def shuffle(self, values):
            self.calls += 1
            values.reverse()

    rng = ReverseRng()
    prompt, _ = describe_virtualhome_observation(
        ENTERTAINMENT_INITIAL[0],
        VirtualHomeTask.ENTERTAINMENT,
        prompt_disturb=False,
        prompt_shuffle=True,
        rng=rng,
    )

    assert rng.calls == 1
    assert "sowe cookies" not in prompt
    assert prompt.startswith(
        " Your goal is to enjoy the chips and the milk while watching TV."
    )
    assert prompt.endswith("Your next step is to")


def test_unknown_and_malformed_tasks_have_clear_errors():
    with pytest.raises(ValueError, match="VirtualHomeTask"):
        describe_virtualhome_observation(FOOD_INITIAL[0], "food")
    with pytest.raises(ValueError, match="11 values"):
        describe_virtualhome_observation([1, 0], VirtualHomeTask.FOOD)
    with pytest.raises(ValueError, match="exactly one room"):
        describe_virtualhome_observation([0] * 11, VirtualHomeTask.FOOD)


def test_adapter_rejects_bad_probability_and_vector_count():
    env = FakeVectorEnv(FOOD_INITIAL)
    with pytest.raises(ValueError, match="between 0 and 1"):
        VirtualHomeAdapter(
            env,
            VirtualHomeTask.FOOD,
            1.1,
            np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="exactly one"):
        VirtualHomeAdapter(
            FakeVectorEnv(FOOD_INITIAL, num_envs=2),
            VirtualHomeTask.FOOD,
            0.2,
            np.random.default_rng(1),
        )


def test_actions_use_actual_environment_ids_and_shared_prompt():
    env = FakeVectorEnv(FOOD_INITIAL)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )

    candidates = adapter.actions(adapter.reset())

    assert [candidate.key for candidate in candidates] == [0, 2, 3, 5, 6]
    assert [candidate.payload for candidate in candidates] == [0, 2, 3, 5, 6]
    assert [candidate.text for candidate in candidates] == [
        "walk to the living room",
        "walk to the bathroom",
        "walk to the bedroom",
        "walk to the microwave",
        "grab the pancake",
    ]
    assert len({candidate.metadata["prompt"] for candidate in candidates}) == 1


def test_preview_isolated_clone_uses_raw_reward():
    env = FakeVectorEnv(FOOD_INITIAL, reward=0.0)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        1.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    result = adapter.preview(
        state,
        adapter.actions(state)[0],
        np.random.default_rng(2),
    )

    assert state.runtime is env
    assert state.runtime.envs[0].steps == 0
    assert state.runtime.step_calls == []
    assert result.state.runtime is not env
    assert result.state.runtime.envs[0].steps == 1
    assert result.reward == 0.0


def test_step_preserves_reward_flooring():
    env = FakeVectorEnv(FOOD_INITIAL, reward=0.0)
    rng = FixedRng(0.9)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    result = adapter.step(
        state,
        adapter.actions(state)[0],
        rng,
    )

    assert result.reward == -0.001
    assert result.state.runtime.envs[0].steps == 1
    assert rng.calls == 0


@pytest.mark.parametrize(
    ("task", "observation", "action_id", "verb"),
    [
        (VirtualHomeTask.FOOD, FOOD_AT_MICROWAVE, 8, "open"),
        (
            VirtualHomeTask.ENTERTAINMENT,
            ENTERTAINMENT_AT_CHIPS,
            9,
            "grab",
        ),
    ],
)
def test_task_stochastic_failure_restores_runtime_and_observation(
    task,
    observation,
    action_id,
    verb,
):
    env = FakeVectorEnv(observation, reward=1.0, done=True)
    adapter_rng = FixedRng(1.0)
    step_rng = FixedRng(0.0)
    adapter = VirtualHomeAdapter(env, task, 1.0, adapter_rng)
    state = adapter.reset()
    action = next(
        candidate
        for candidate in adapter.actions(state)
        if candidate.payload == action_id
    )

    result = adapter.step(state, action, step_rng)

    assert verb in action.text
    assert step_rng.calls == 1
    assert adapter_rng.calls == 0
    np.testing.assert_array_equal(result.state.observation, observation)
    assert result.state.runtime.envs[0].steps == 0
    assert result.state.runtime.envs[0].first_grab_chips_flag is True
    assert result.state.runtime.envs[0].first_grab_milk_flag is True
    assert result.state.runtime.envs[0].pre_tv_state == 0
    assert result.reward == -0.001
    assert result.terminated is False
    assert result.truncated is False
    assert result.state.runtime._planu_terminated is False
    assert result.state.runtime._planu_truncated is False
    assert adapter.is_terminal(result.state) is False
    assert adapter.is_truncated(result.state) is False
    assert len(env.step_calls) == 1


def test_only_task_specific_verb_draws_stochastic_rng():
    food_env = FakeVectorEnv(FOOD_INITIAL)
    food_rng = FixedRng(0.0)
    food_adapter = VirtualHomeAdapter(
        food_env,
        VirtualHomeTask.FOOD,
        1.0,
        np.random.default_rng(1),
    )
    food_state = food_adapter.reset()
    grab = next(
        candidate
        for candidate in food_adapter.actions(food_state)
        if "grab" in candidate.text
    )

    food_adapter.step(food_state, grab, food_rng)

    assert food_rng.calls == 0


def test_state_key_includes_all_virtualhome_hidden_runtime_fields():
    env = FakeVectorEnv(ENTERTAINMENT_INITIAL)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.ENTERTAINMENT,
        0.0,
        np.random.default_rng(1),
    )
    baseline = adapter.reset()
    states = [baseline]
    for name, value in [
        ("steps", 1),
        ("env_step", 1),
        ("_elapsed_steps", 1),
        ("first_grab_chips_flag", False),
        ("first_grab_milk_flag", False),
        ("pre_tv_state", 1),
    ]:
        changed = adapter.clone(baseline)
        setattr(changed.runtime.envs[0], name, value)
        states.append(changed)

    keys = {adapter.state_key(state) for state in states}

    assert len(keys) == len(states)
    assert adapter.state_key(baseline) == adapter.state_key(
        adapter.clone(baseline)
    )


def test_state_key_scans_wrappers_inner_env_and_unwrapped():
    class Base:
        def __init__(self):
            self.pre_tv_state = 0

    class Wrapper:
        def __init__(self):
            self.env_step = 0
            self.env = Base()
            self.unwrapped = Base()

    env = FakeVectorEnv(ENTERTAINMENT_INITIAL)
    env.envs = [Wrapper()]
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.ENTERTAINMENT,
        0.0,
        np.random.default_rng(1),
    )
    observation = np.array(ENTERTAINMENT_INITIAL, copy=True)
    baseline = EnvironmentState(observation, copy.deepcopy(env))
    wrapper = EnvironmentState(observation, copy.deepcopy(env))
    wrapper.runtime.envs[0].env_step = 1
    inner = EnvironmentState(observation, copy.deepcopy(env))
    inner.runtime.envs[0].env.pre_tv_state = 1
    unwrapped = EnvironmentState(observation, copy.deepcopy(env))
    unwrapped.runtime.envs[0].unwrapped.pre_tv_state = 1

    assert len(
        {
            adapter.state_key(baseline),
            adapter.state_key(wrapper),
            adapter.state_key(inner),
            adapter.state_key(unwrapped),
        }
    ) == 4


def test_is_truncated_defaults_false_and_reads_explicit_runtime_state():
    adapter = VirtualHomeAdapter(
        FakeVectorEnv(FOOD_INITIAL),
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    assert adapter.is_truncated(state) is False

    state.runtime.envs[0]._truncated = True
    assert adapter.is_truncated(state) is True


def test_virtualhome_reset_clears_both_completion_markers():
    env = FakeVectorEnv(FOOD_INITIAL)
    env._planu_terminated = True
    env._planu_truncated = True
    env.envs[0]._planu_terminated = True
    env.envs[0]._planu_truncated = True
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )

    state = adapter.reset()

    assert state.runtime._planu_terminated is False
    assert state.runtime._planu_truncated is False
    assert state.runtime.envs[0]._planu_terminated is False
    assert state.runtime.envs[0]._planu_truncated is False
    assert adapter.is_terminal(state) is False
    assert adapter.is_truncated(state) is False


@pytest.mark.parametrize("reward", [0.0, 0.1])
@pytest.mark.parametrize("transition_name", ["preview", "step"])
def test_virtualhome_horizon_without_goal_is_truncated(
    reward,
    transition_name,
):
    env = FakeVirtualHomeHorizonVectorEnv(ENTERTAINMENT_INITIAL, reward)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.ENTERTAINMENT,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()
    action = adapter.actions(state)[0]

    result = getattr(adapter, transition_name)(
        state,
        action,
        np.random.default_rng(2),
    )

    assert result.terminated is False
    assert result.truncated is True
    assert result.info["truncation_reason"] == "environment_horizon"
    assert result.state.runtime._planu_terminated is False
    assert result.state.runtime._planu_truncated is True
    assert adapter.is_terminal(result.state) is False
    assert adapter.is_truncated(result.state) is True


@pytest.mark.parametrize("transition_name", ["preview", "step"])
def test_virtualhome_goal_reward_at_horizon_remains_terminal(
    transition_name,
):
    env = FakeVirtualHomeHorizonVectorEnv(FOOD_INITIAL, 1.0)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    result = getattr(adapter, transition_name)(
        state,
        adapter.actions(state)[0],
        np.random.default_rng(2),
    )

    assert result.terminated is True
    assert result.truncated is False
    assert "truncation_reason" not in result.info
    assert result.state.runtime._planu_terminated is True
    assert result.state.runtime._planu_truncated is False
    assert adapter.is_terminal(result.state) is True
    assert adapter.is_truncated(result.state) is False


def test_virtualhome_explicit_time_limit_requires_explicit_goal_override():
    truncated_env = FakeVirtualHomeHorizonVectorEnv(
        FOOD_INITIAL,
        1.0,
        info={"TimeLimit.truncated": True},
    )
    truncated_adapter = VirtualHomeAdapter(
        truncated_env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    truncated_state = truncated_adapter.reset()

    truncated = truncated_adapter.step(
        truncated_state,
        truncated_adapter.actions(truncated_state)[0],
        np.random.default_rng(2),
    )

    assert truncated.terminated is False
    assert truncated.truncated is True

    success_env = FakeVirtualHomeHorizonVectorEnv(
        FOOD_INITIAL,
        1.0,
        info={"TimeLimit.truncated": True, "goal_reached": True},
    )
    success_adapter = VirtualHomeAdapter(
        success_env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    success_state = success_adapter.reset()

    success = success_adapter.step(
        success_state,
        success_adapter.actions(success_state)[0],
        np.random.default_rng(2),
    )

    assert success.terminated is True
    assert success.truncated is False


def test_virtualhome_search_reports_environment_horizon():
    adapter = VirtualHomeAdapter(
        FakeVirtualHomeHorizonVectorEnv(ENTERTAINMENT_INITIAL, 0.1),
        VirtualHomeTask.ENTERTAINMENT,
        0.0,
        np.random.default_rng(1),
    )
    search = PlanUSearch(
        adapter,
        ConstantActionScorer(),
        virtualhome_config(
            VirtualHomeTask.ENTERTAINMENT,
            False,
            max_iterations=1,
            max_depth=2,
        ),
    )

    result = search.run_iteration(0, np.random.default_rng(7))

    assert result.terminated is False
    assert result.truncated is True
    assert result.truncation_reason == "environment_horizon"
    assert result.state_path[-1].truncation_reason == "environment_horizon"


def test_virtualhome_horizon_survives_vector_env_auto_reset():
    env = AutoResetVirtualHomeHorizonVectorEnv(FOOD_INITIAL, 0.0)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    state = adapter.reset()

    result = adapter.step(
        state,
        adapter.actions(state)[0],
        np.random.default_rng(2),
    )

    assert result.state.runtime.envs[0].env.steps == 0
    assert result.terminated is False
    assert result.truncated is True
    assert result.info["truncation_reason"] == "environment_horizon"


@pytest.mark.parametrize(
    ("task", "observation"),
    [
        (VirtualHomeTask.FOOD, FOOD_INITIAL),
        (VirtualHomeTask.ENTERTAINMENT, ENTERTAINMENT_INITIAL),
    ],
)
def test_shared_search_smoke_and_terminal_marker_reset(task, observation):
    env = FakeVectorEnv(observation, reward=1.0, done=True)
    adapter = VirtualHomeAdapter(
        env,
        task,
        0.0,
        np.random.default_rng(1),
    )
    search = PlanUSearch(
        adapter,
        ConstantActionScorer(),
        virtualhome_config(task, False, max_iterations=2, max_depth=2),
    )

    first = search.run_iteration(0, np.random.default_rng(7))
    second = search.run_iteration(1, np.random.default_rng(8))

    assert first.terminated is True
    assert second.terminated is True
    assert len(first.actions) == 1
    assert len(second.actions) == 1
    assert len(env.step_calls) == 2
    assert env.reset_calls == [None, None]


def test_food_zero_scorer_initializes_from_preview_reward_only():
    env = FakeVectorEnv(FOOD_INITIAL, reward=0.25, done=True)
    adapter = VirtualHomeAdapter(
        env,
        VirtualHomeTask.FOOD,
        0.0,
        np.random.default_rng(1),
    )
    search = PlanUSearch(
        adapter,
        ConstantActionScorer(0.0),
        virtualhome_config(
            VirtualHomeTask.FOOD,
            False,
            max_iterations=1,
            max_depth=1,
        ),
    )

    state = adapter.reset()
    root = search._ensure_root(state)
    search.expand(root, state, np.random.default_rng(7))

    values = {
        float(action.distribution.values[0])
        for action in root.children.values()
    }
    assert len(values) == 1
    assert values.pop() == pytest.approx(0.25)


@pytest.mark.parametrize(
    "module_name",
    [
        "mcts.virtualhome.PlanU_v1",
        "mcts.virtualhome.PlanU_v2",
        "mcts.virtualhome.PlanU_inference_food",
        "mcts.virtualhome.PlanU_entertainment",
    ],
)
def test_virtualhome_compatibility_and_runner_modules_import_without_optional_deps(
    module_name,
):
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    "path",
    [
        Path("mcts/virtualhome/PlanU_v1.py"),
        Path("mcts/virtualhome/PlanU_v2.py"),
    ],
)
def test_virtualhome_compatibility_modules_define_no_algorithm_classes(path):
    tree = ast.parse(path.read_text())
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert class_names == set()
    assert not function_names.intersection(
        {"select", "expand", "backup", "mcts_update"}
    )


def test_virtualhome_compatibility_exports_map_to_shared_core():
    food = importlib.import_module("mcts.virtualhome.PlanU_v1")
    entertainment = importlib.import_module("mcts.virtualhome.PlanU_v2")

    assert food.ActionNode is entertainment.ActionNode
    assert food.LanguageNode is entertainment.LanguageNode
    assert food.PlanUSearch is PlanUSearch
    assert entertainment.PlanUSearch is PlanUSearch
    assert food.LLMAgent is ConstantActionScorer
    assert entertainment.LLMAgent is HuggingFaceActionScorer


@pytest.mark.parametrize(
    ("module_name", "extra_args"),
    [
        ("mcts.virtualhome.PlanU_inference_food", []),
        (
            "mcts.virtualhome.PlanU_entertainment",
            ["--temperature", "0.37"],
        ),
    ],
)
def test_runner_parsers_preserve_flags_and_parse_boolean_strings(
    module_name,
    extra_args,
):
    runner = importlib.import_module(module_name)
    args = runner.parse_args(
        [
            "--rnd",
            "False",
            "--transpositions",
            "True",
            "--depth",
            "7",
            "--num-envs",
            "1",
            *extra_args,
        ]
    )

    assert args.rnd is False
    assert args.transpositions is True
    assert args.depth == 7
    assert args.num_envs == 1
    runner.validate_args(args)


@pytest.mark.parametrize(
    "module_name",
    [
        "mcts.virtualhome.PlanU_inference_food",
        "mcts.virtualhome.PlanU_entertainment",
    ],
)
def test_runners_enforce_one_vector_environment(module_name):
    runner = importlib.import_module(module_name)
    args = runner.parse_args(["--num-envs", "2"])

    with pytest.raises(AssertionError, match="num_envs"):
        runner.validate_args(args)


@pytest.mark.parametrize(
    ("task", "obs_shape"),
    [
        (VirtualHomeTask.FOOD, 11),
        (VirtualHomeTask.ENTERTAINMENT, 21),
    ],
)
def test_virtualhome_rnd_settings_match_legacy(task, obs_shape):
    runner = importlib.import_module(
        "mcts.virtualhome.PlanU_inference_food"
        if task is VirtualHomeTask.FOOD
        else "mcts.virtualhome.PlanU_entertainment"
    )

    settings = runner.rnd_settings(task)

    assert settings["learning_rate"] == 1e-5
    assert settings["batch_size"] == 15
    assert settings["hidden_size_list"] == [64, 64, 128]
    assert settings["update_per_collect"] == 20
    assert settings["obs_shape"] == obs_shape


@pytest.mark.parametrize(
    ("module_name", "rewards", "expected"),
    [
        ("mcts.virtualhome.PlanU_inference_food", [1.0, 2.0], 2.98),
        ("mcts.virtualhome.PlanU_entertainment", [1.0, 2.0], 2.98),
    ],
)
def test_runners_report_legacy_discounted_return(
    module_name,
    rewards,
    expected,
):
    runner = importlib.import_module(module_name)

    assert runner.discounted_return(rewards) == pytest.approx(expected)


def test_food_runner_logs_raw_step_rewards_in_action_order(caplog):
    runner = importlib.import_module(
        "mcts.virtualhome.PlanU_inference_food"
    )
    result = types.SimpleNamespace(
        action_path=[
            types.SimpleNamespace(
                action=types.SimpleNamespace(text="open the microwave")
            ),
            types.SimpleNamespace(
                action=types.SimpleNamespace(text="grab the pancake")
            ),
        ],
        rewards=[1.0, 2.0],
    )
    caplog.set_level(logging.INFO)

    runner._log_trajectory_steps(result)

    assert caplog.messages == [
        "action : open the microwave  reward : 1.0",
        "action : grab the pancake  reward : 2.0",
    ]


def test_entertainment_runner_logs_discounted_step_rewards_in_action_order(
    caplog,
):
    runner = importlib.import_module("mcts.virtualhome.PlanU_entertainment")
    result = types.SimpleNamespace(
        action_path=[
            types.SimpleNamespace(
                action=types.SimpleNamespace(text="grab the chips")
            ),
            types.SimpleNamespace(
                action=types.SimpleNamespace(text="walk to the sofa")
            ),
        ],
        rewards=[1.0, 2.0],
    )
    caplog.set_level(logging.INFO)

    runner._log_trajectory_steps(result)

    assert caplog.messages == [
        "action : grab the chips  reward : 1.0",
        "action : walk to the sofa  reward : 1.98",
    ]


def test_food_runner_uses_zero_constant_scorer_and_shared_provenance():
    runner = importlib.import_module(
        "mcts.virtualhome.PlanU_inference_food"
    )
    args = runner.parse_args(["--maxiterations", "3", "--depth", "2"])

    search, scorer, config = runner.build_planu_components(
        args,
        envs=FakeVectorEnv(FOOD_INITIAL),
        device="cpu",
        rnd_writer=object(),
    )

    assert isinstance(search, PlanUSearch)
    assert isinstance(scorer, ConstantActionScorer)
    assert scorer.value == 0.0
    assert config.max_iterations == 3
    assert config.max_depth == 2
    assert runner._build_effective_config is build_effective_config


def test_entertainment_runner_passes_cli_temperature_to_shared_scorer(
    monkeypatch,
):
    runner = importlib.import_module("mcts.virtualhome.PlanU_entertainment")
    args = runner.parse_args(
        ["--maxiterations", "3", "--depth", "2", "--temperature", "0.37"]
    )
    calls = {}

    class RecordingScorer:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

    monkeypatch.setattr(runner, "HuggingFaceActionScorer", RecordingScorer)

    search, scorer, config = runner.build_planu_components(
        args,
        envs=FakeVectorEnv(ENTERTAINMENT_INITIAL),
        device="cpu",
        rnd_writer=object(),
    )

    assert isinstance(search, PlanUSearch)
    assert isinstance(scorer, RecordingScorer)
    assert calls["kwargs"]["temperature"] == 0.37
    assert calls["kwargs"]["device"] == "cpu"
    assert config.max_iterations == 3
    assert config.max_depth == 2
    assert runner._build_effective_config is build_effective_config


def test_entertainment_runner_uses_fp16_only_for_cuda(monkeypatch):
    runner = importlib.import_module("mcts.virtualhome.PlanU_entertainment")
    args = runner.parse_args(["--maxiterations", "1", "--depth", "1"])
    float16 = object()
    calls = []

    class RecordingScorer:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(runner, "HuggingFaceActionScorer", RecordingScorer)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(float16=float16),
    )

    for device in ("cpu", "cuda"):
        runner.build_planu_components(
            args,
            envs=FakeVectorEnv(ENTERTAINMENT_INITIAL),
            device=device,
            rnd_writer=object(),
        )

    assert [call["torch_dtype"] for call in calls] == [None, float16]
    assert [call["device"] for call in calls] == ["cpu", "cuda"]


@pytest.mark.parametrize(
    ("module_name", "return_value", "expected"),
    [
        ("mcts.virtualhome.PlanU_inference_food", 0.001, True),
        ("mcts.virtualhome.PlanU_inference_food", 0.0, False),
        ("mcts.virtualhome.PlanU_entertainment", 1.0, True),
        ("mcts.virtualhome.PlanU_entertainment", 0.999, False),
    ],
)
def test_task_specific_success_thresholds(
    module_name,
    return_value,
    expected,
):
    runner = importlib.import_module(module_name)

    assert runner.is_success(return_value) is expected


@pytest.mark.parametrize(
    ("module_name", "domain"),
    [
        ("mcts.virtualhome.PlanU_inference_food", "fp"),
        ("mcts.virtualhome.PlanU_entertainment", "en"),
    ],
)
def test_token_log_prefix_is_preserved_and_appends_config_hash(
    module_name,
    domain,
):
    runner = importlib.import_module(module_name)
    args = runner.parse_args(
        [
            "--base-model",
            "custom/model",
            "--seed",
            "17",
            "--stochastic",
            "0.4",
            "--rnd",
            "True",
        ]
    )

    path = runner._token_log_path(args, "abc123def456")

    assert path == (
        f"./results_new/Model=custom/model/{domain}/PlanU/"
        "seed=17/stochastic=0.4/rnd=True/config=abc123def456"
    )


def test_virtualhome_shell_runs_both_reference_tasks():
    shell = Path("scripts/PlanU_Virtualhome.sh").read_text()

    assert shell.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "mcts/virtualhome/PlanU_inference_food.py" in shell
    assert "mcts/virtualhome/PlanU_entertainment.py" in shell
    assert "PlanU_inference_entertainment.py" not in shell
    assert shell.count("--maxiterations 1000") == 2
    assert shell.count('--rnd "True"') == 2
    assert "PYTHON=" in shell
    assert "BASE_MODEL=" in shell
