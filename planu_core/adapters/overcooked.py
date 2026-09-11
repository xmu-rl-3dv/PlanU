import copy
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentState,
    TransitionResult,
)


_VALID_TASKS = (0, 3)


def _validate_task(task: int) -> None:
    if task not in _VALID_TASKS:
        raise ValueError("task must be 0 or 3")


def describe_overcooked_observation(
    observation: Any,
    task: int,
    prompt_disturb: bool = True,
    prompt_shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    _validate_task(task)
    obs = np.asarray(observation).reshape(-1).tolist()

    if task == 3:
        action_list = [
            "pick up the tomato",
            "pick up the lettuce",
            "pick up the onion",
            "take the empty bowl",
            "walk to the first cutting board",
            "walk to the second cutting board",
            "serve nothing",
            "chop nothing",
        ]

        ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
        raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
        chopped = [False, False, False]
        ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
        sentences = ["There are two fixed cutting boards in the room."]
        if prompt_disturb:
            sentences = [
                "There are two fixed cutting boards in the room. Earlier this "
                "day, you made a dish with chopped potatoes."
            ]

        item = []
        agent_pos = obs[17:19]
        item_pos = {
            "in_agent": agent_pos,
            "in_first_cutting_board": [1, 0],
            "in_second_cutting_board": [2, 0],
        }
        overlay = {
            "in_agent": [],
            "in_first_cutting_board": [],
            "in_second_cutting_board": [],
        }

        for index in range(4):
            pos = obs[3 * index : 3 * index + 2]
            if pos == ori_pos[index]:
                item.append(ingredient[index])
            if index < 3 and obs[3 * index + 2] == 3:
                chopped[index] = True
            for location in overlay:
                if pos == item_pos[location]:
                    overlay[location].append(index)
                    if len(overlay[location]) > 1:
                        action_list[3] = "take the bowl"

        if len(item) == 1:
            template = "You notice {} on the table."
        elif len(item) == 2:
            template = "You notice {} and {} on the different tables."
        elif len(item) == 3:
            template = "You notice {}, {} and {} on the different tables."
        elif len(item) == 4:
            template = (
                "You notice {}, {}, {} and {} on the different tables."
            )
        if item:
            sentences.append(template.format(*item).capitalize())

        cutting_board_index = ["first", "second"]
        cutting_board_name = [
            "in_first_cutting_board",
            "in_second_cutting_board",
        ]
        for board_index in range(2):
            board_items = overlay[cutting_board_name[board_index]]
            if len(board_items) == 1:
                item_id = board_items[0]
                template = "{} is on the {} cutting board."
                if item_id == 3:
                    sentences.append(
                        template.format(
                            "a bowl",
                            cutting_board_index[board_index],
                        ).capitalize()
                    )
                else:
                    state = (
                        "a chopped " if chopped[item_id] else "an unchopped "
                    )
                    sentences.append(
                        template.format(
                            state + raw_ingredient[item_id],
                            cutting_board_index[board_index],
                        ).capitalize()
                    )
                    if agent_pos == [board_index + 1, 1]:
                        action_list[-1] = (
                            "chop the " + raw_ingredient[item_id]
                        )
            elif len(board_items) > 1:
                in_plate_item = board_items[:-1]
                if len(in_plate_item) == 1:
                    template = (
                        "A bowl containing chopped {} is on the {} cutting "
                        "board."
                    )
                elif len(in_plate_item) == 2:
                    template = (
                        "A bowl containing chopped {} and {} is on the {} "
                        "cutting board."
                    )
                elif len(in_plate_item) == 3:
                    template = (
                        "A bowl containing chopped {}, {} and {} is on the {} "
                        "cutting board."
                    )
                sentences.append(
                    template.format(
                        *[raw_ingredient[item_id] for item_id in in_plate_item],
                        cutting_board_index[board_index],
                    ).capitalize()
                )

        if agent_pos == [1, 1]:
            current_board = 0
        elif agent_pos == [2, 1]:
            current_board = 1
        else:
            current_board = -1

        action_template = "put the {} on the {} cutting board"
        hold_bowl_action = [
            "put the tomato in the bowl",
            "put the lettuce in the bowl",
            "put the onion in the bowl",
        ]
        held_items = overlay["in_agent"]

        if current_board >= 0:
            if len(held_items) == 0:
                template = (
                    "Currently you are standing in front of the {} cutting "
                    "board without anything in hand."
                )
                sentences.append(
                    template.format(
                        cutting_board_index[current_board]
                    ).capitalize()
                )
            elif len(held_items) == 1:
                action_list[6] = "serve the dish"
                item_id = held_items[0]
                template = (
                    "Currently you are standing in front of the {} cutting "
                    "board, carrying {} in hand."
                )
                if item_id == 3:
                    sentences.append(
                        template.format(
                            cutting_board_index[current_board],
                            "a bowl",
                        ).capitalize()
                    )
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
                else:
                    state = (
                        "a chopped " if chopped[item_id] else "an unchopped "
                    )
                    sentences.append(
                        template.format(
                            cutting_board_index[current_board],
                            state + raw_ingredient[item_id],
                        ).capitalize()
                    )
                    if not chopped[item_id]:
                        action_list[4] = action_template.format(
                            raw_ingredient[item_id],
                            "first",
                        )
                        action_list[5] = action_template.format(
                            raw_ingredient[item_id],
                            "second",
                        )
            elif len(held_items) > 1:
                action_list[6] = "serve the dish"
                in_plate_item = held_items[:-1]
                if len(in_plate_item) == 1:
                    template = (
                        "Currently you are standing in front of the {} cutting "
                        "board, carrying a bowl containing chopped {} in hand."
                    )
                elif len(in_plate_item) == 2:
                    template = (
                        "Currently you are standing in front of the {} cutting "
                        "board, carrying a bowl containing chopped {} and {} "
                        "in hand."
                    )
                elif len(in_plate_item) == 3:
                    template = (
                        "Currently you are standing in front of the {} cutting "
                        "board, carrying a bowl containing chopped {}, {} and "
                        "{} in hand."
                    )
                sentences.append(
                    template.format(
                        cutting_board_index[current_board],
                        *[raw_ingredient[item_id] for item_id in in_plate_item],
                    ).capitalize()
                )
                action_list[:3] = hold_bowl_action
                action_list[4] = action_template.format("bowl", "first")
                action_list[5] = action_template.format("bowl", "second")
        else:
            if len(held_items) == 0:
                sentences.append(
                    "Currently you don't have anything in hand.".capitalize()
                )
            elif len(held_items) == 1:
                action_list[6] = "serve the dish"
                item_id = held_items[0]
                template = "Currently you are carrying {} in hand."
                if item_id == 3:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
                else:
                    state = (
                        "a chopped " if chopped[item_id] else "an unchopped "
                    )
                    sentences.append(
                        template.format(
                            state + raw_ingredient[item_id]
                        ).capitalize()
                    )
                    if not chopped[item_id]:
                        action_list[4] = action_template.format(
                            raw_ingredient[item_id],
                            "first",
                        )
                        action_list[5] = action_template.format(
                            raw_ingredient[item_id],
                            "second",
                        )
            elif len(held_items) > 1:
                action_list[6] = "serve the dish"
                in_plate_item = held_items[:-1]
                if len(in_plate_item) == 1:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{}."
                    )
                elif len(in_plate_item) == 2:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{} and {}."
                    )
                elif len(in_plate_item) == 3:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{}, {} and {}."
                    )
                sentences.append(
                    template.format(
                        *[raw_ingredient[item_id] for item_id in in_plate_item]
                    ).capitalize()
                )
                action_list[:3] = hold_bowl_action
                action_list[4] = action_template.format("bowl", "first")
                action_list[5] = action_template.format("bowl", "second")

        sentences.append(
            "Your goal is to serve the dish of a bowl only containing chopped "
            "tomato and lettuce."
        )
        sentences.append("Your next step is to")
    else:
        action_list = [
            "pick up the tomato",
            "take the bowl",
            "walk to the cutting board",
            "serve nothing",
            "chop nothing",
        ]

        ingredient = ["a tomato", "a bowl"]
        raw_ingredient = ["tomato", "bowl"]
        chopped = [False]
        ori_pos = [[0, 5], [6, 5]]
        sentences = ["There is a fixed cutting board in the room."]

        item = []
        agent_pos = obs[9:11]
        item_pos = {
            "in_agent": agent_pos,
            "in_first_cutting_board": [1, 0],
        }
        overlay = {"in_agent": [], "in_first_cutting_board": []}

        for index in range(2):
            pos = obs[3 * index : 3 * index + 2]
            if pos == ori_pos[index]:
                item.append(ingredient[index])
            if index < 1 and obs[3 * index + 2] == 3:
                chopped[index] = True
            for location in overlay:
                if pos == item_pos[location]:
                    overlay[location].append(index)

        if len(item) == 1:
            template = "You notice {} on the table."
        elif len(item) == 2:
            template = "You notice {} and {} on the different tables."
        if item:
            sentences.append(template.format(*item).capitalize())

        board_items = overlay["in_first_cutting_board"]
        if len(board_items) == 1:
            item_id = board_items[0]
            template = "{} is on the cutting board."
            if item_id == 1:
                sentences.append(template.format("a bowl").capitalize())
            else:
                state = "a chopped " if chopped[item_id] else "an unchopped "
                sentences.append(
                    template.format(
                        state + raw_ingredient[item_id]
                    ).capitalize()
                )
                if agent_pos == [1, 1]:
                    action_list[-1] = "chop the " + raw_ingredient[item_id]
        elif len(board_items) > 1:
            sentences.append(
                "a bowl containing a chopped tomato is on the cutting board."
                .capitalize()
            )

        if agent_pos == [1, 1]:
            current_board = 0
        elif agent_pos == [2, 1]:
            current_board = 1
        else:
            current_board = -1

        action_template = "put the {} on the cutting board"
        hold_bowl_action = ["put the tomato in the bowl"]
        held_items = overlay["in_agent"]

        if current_board >= 0:
            if len(held_items) == 0:
                sentences.append(
                    "Currently you are standing in front of the cutting board "
                    "without anything in hand.".capitalize()
                )
            elif len(held_items) == 1:
                action_list[3] = "serve the dish"
                item_id = held_items[0]
                template = (
                    "Currently you are standing in front of the cutting board, "
                    "carrying {} in hand."
                )
                if item_id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")
                else:
                    state = (
                        "a chopped " if chopped[item_id] else "an unchopped "
                    )
                    sentences.append(
                        template.format(
                            state + raw_ingredient[item_id]
                        ).capitalize()
                    )
                    if not chopped[item_id]:
                        action_list[2] = action_template.format(
                            raw_ingredient[item_id]
                        )
            elif len(held_items) > 1:
                action_list[3] = "serve the dish"
                in_plate_item = held_items[:-1]
                template = (
                    "Currently you are standing in front of the cutting board, "
                    "carrying a bowl containing chopped {} in hand."
                )
                sentences.append(
                    template.format(
                        *[raw_ingredient[item_id] for item_id in in_plate_item]
                    ).capitalize()
                )
                action_list[0] = hold_bowl_action[0]
                action_list[2] = action_template.format("bowl")
        else:
            if len(held_items) == 0:
                sentences.append(
                    "Currently you don't have anything in hand.".capitalize()
                )
            elif len(held_items) == 1:
                action_list[3] = "serve the dish"
                item_id = held_items[0]
                template = "Currently you are carrying {} in hand."
                if item_id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")
                else:
                    state = (
                        "a chopped " if chopped[item_id] else "an unchopped "
                    )
                    sentences.append(
                        template.format(
                            state + raw_ingredient[item_id]
                        ).capitalize()
                    )
                    if not chopped[item_id]:
                        action_list[2] = action_template.format(
                            raw_ingredient[item_id]
                        )
            elif len(held_items) > 1:
                action_list[3] = "serve the dish"
                in_plate_item = held_items[:-1]
                if len(in_plate_item) == 1:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{}."
                    )
                elif len(in_plate_item) == 2:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{} and {}."
                    )
                elif len(in_plate_item) == 3:
                    template = (
                        "Currently you are carrying a bowl containing chopped "
                        "{}, {} and {}."
                    )
                sentences.append(
                    template.format(
                        *[raw_ingredient[item_id] for item_id in in_plate_item]
                    ).capitalize()
                )
                action_list[0] = hold_bowl_action[0]
                action_list[2] = action_template.format("bowl")

        sentences.append(
            "Your goal is to serve the dish of a bowl only containing chopped "
            "tomato."
        )
        sentences.append("Your next step is to")

    if prompt_shuffle:
        if rng is None:
            raise ValueError("rng is required when prompt_shuffle is enabled")
        shuffled = sentences[:-1]
        rng.shuffle(shuffled)
        sentences = shuffled + [sentences[-1]]
    return {"prompt": " ".join(sentences), "action": action_list}


def _runtime_objects(runtime: Any) -> Iterable[Tuple[str, Any]]:
    pending = [("runtime", runtime)]
    seen = set()
    while pending:
        path, current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield path, current
        envs = getattr(current, "envs", None)
        if envs is not None:
            pending.extend(
                (f"{path}.envs[{index}]", env)
                for index, env in enumerate(envs)
            )
        inner = getattr(current, "env", None)
        if inner is not None:
            pending.append((f"{path}.env", inner))
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            pending.append((f"{path}.unwrapped", unwrapped))


def _stable_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return tuple(array.reshape(-1).tolist())


def _runtime_step_signature(runtime: Any) -> Tuple[Any, ...]:
    signature = []
    for path, current in _runtime_objects(runtime):
        for name in (
            "steps",
            "step_count",
            "_step_count",
            "env_step",
            "_elapsed_steps",
        ):
            if hasattr(current, name):
                signature.append(
                    (path, name, _stable_value(getattr(current, name)))
                )
    return tuple(signature)


def _scalar_reward(value: Any) -> float:
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ValueError("reward must be scalar")
    reward = float(flattened[0])
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    return reward


def _scalar_bool(value: Any, name: str) -> bool:
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ValueError(f"{name} must be scalar")
    return bool(flattened[0])


class OvercookedAdapter:
    def __init__(
        self,
        envs: Any,
        task: int,
        stochastic_probability: float,
        rng: np.random.Generator,
        prompt_disturb: bool = True,
        prompt_shuffle: bool = False,
    ) -> None:
        _validate_task(task)
        if not 0.0 <= stochastic_probability <= 1.0:
            raise ValueError(
                "stochastic_probability must be between 0 and 1"
            )
        if getattr(envs, "num_envs", None) != 1:
            raise ValueError("OvercookedAdapter requires exactly one vector env")
        self.envs = envs
        self.task = task
        self.stochastic_probability = float(stochastic_probability)
        self.rng = rng
        self.prompt_disturb = prompt_disturb
        self.prompt_shuffle = prompt_shuffle

    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        if seed is None:
            observation = self.envs.reset()
        else:
            try:
                observation = self.envs.reset(seed=seed)
            except TypeError:
                observation = self.envs.reset()
        return EnvironmentState(
            observation=np.asarray(observation).copy(),
            runtime=self.envs,
        )

    def clone(self, state: EnvironmentState) -> EnvironmentState:
        return EnvironmentState(
            observation=np.asarray(state.observation).copy(),
            runtime=copy.deepcopy(state.runtime),
        )

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ):
        del state_visit_count
        observation = np.asarray(state.observation)[0]
        description = describe_overcooked_observation(
            observation,
            self.task,
            prompt_disturb=self.prompt_disturb,
            prompt_shuffle=self.prompt_shuffle,
            rng=self.rng,
        )
        metadata = {"prompt": description["prompt"]}
        return [
            ActionCandidate(
                key=index,
                payload=index,
                text=text,
                metadata=metadata,
            )
            for index, text in enumerate(description["action"])
        ]

    def preview(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        del rng
        preview_state = self.clone(state)
        observation, reward, done, info = preview_state.runtime.step(
            np.array([action.payload])
        )
        terminated = _scalar_bool(done, "done")
        preview_state.runtime._planu_terminated = terminated
        next_state = EnvironmentState(
            observation=np.asarray(observation).copy(),
            runtime=preview_state.runtime,
        )
        return TransitionResult(
            state=next_state,
            reward=_scalar_reward(reward),
            terminated=terminated,
            truncated=False,
            info={"raw_info": info},
        )

    def step(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        before = self.clone(state)
        observation, reward, done, info = state.runtime.step(
            np.array([action.payload])
        )
        terminated = _scalar_bool(done, "done")
        if (
            "chop" in action.text
            and rng.random() < self.stochastic_probability
        ):
            before.runtime._planu_terminated = False
            return TransitionResult(
                state=before,
                reward=-0.001,
                terminated=False,
                truncated=False,
                info={"raw_info": info},
            )
        state.runtime._planu_terminated = terminated
        return TransitionResult(
            state=EnvironmentState(
                observation=np.asarray(observation).copy(),
                runtime=state.runtime,
            ),
            reward=_scalar_reward(reward),
            terminated=terminated,
            truncated=False,
            info={"raw_info": info},
        )

    def state_key(self, state: EnvironmentState):
        observation = tuple(
            np.asarray(state.observation).reshape(-1).tolist()
        )
        return observation, _runtime_step_signature(state.runtime)

    def is_terminal(self, state: EnvironmentState) -> bool:
        for _, runtime in _runtime_objects(state.runtime):
            for name in (
                "_planu_terminated",
                "terminated",
                "_terminated",
                "done",
                "_done",
            ):
                if hasattr(runtime, name):
                    value = np.asarray(getattr(runtime, name)).reshape(-1)
                    if value.size and bool(value[0]):
                        return True
        return False


def overcooked_config(
    task: int,
    rnd: bool,
    max_iterations: int,
    max_depth: int,
    temperature: float = 1.0,
) -> PlanUConfig:
    _validate_task(task)
    del temperature
    if task == 0:
        schedule = SelectionSchedule(
            always_sample_before=50,
            probabilistic_sample_before=100,
            sample_probability=0.2,
        )
    else:
        schedule = SelectionSchedule()
    return PlanUConfig(
        quantile_learning_rate=0.75,
        curiosity_weight=0.5 if rnd else 0.0,
        include_preview_reward=True,
        max_iterations=max_iterations,
        max_depth=max_depth,
        selection_temperature=1.0,
        selection_schedule=schedule,
    )
