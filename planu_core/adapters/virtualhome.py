import copy
from enum import Enum
import math
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentState,
    TransitionResult,
)


class VirtualHomeTask(Enum):
    FOOD = ("VirtualHome-v1", "open", False, 0.25, 11)
    ENTERTAINMENT = ("VirtualHome-v2", "grab", True, 0.1, 21)

    @property
    def env_id(self) -> str:
        return self.value[0]

    @property
    def stochastic_verb(self) -> str:
        return self.value[1]

    @property
    def uses_llm_prior(self) -> bool:
        return self.value[2]

    @property
    def curiosity_weight(self) -> float:
        return self.value[3]

    @property
    def obs_shape(self) -> int:
        return self.value[4]


_FOOD_ACTIONS = (
    "walk to the living room",
    "walk to the kitchen",
    "walk to the bathroom",
    "walk to the bedroom",
    "walk to the pancake",
    "walk to the microwave",
    "grab the pancake",
    "put the pancake in the microwave",
    "open the microwave",
    "close the microwave",
)

_ENTERTAINMENT_ACTIONS = (
    "walk to the living room",
    "walk to the kitchen",
    "walk to the bathroom",
    "walk to the bedroom",
    "walk to the chips",
    "walk to the milk",
    "walk to the coffee table",
    "walk to the TV",
    "walk to the sofa",
    "grab the chips",
    "grab the milk",
    "put the chips on the coffee table",
    "put the milk on the coffee table",
    "turn on the TV",
    "turn off the TV",
    "sit on the sofa",
    "stand up from the sofa",
)


def _validate_task(task: VirtualHomeTask) -> None:
    if not isinstance(task, VirtualHomeTask):
        raise ValueError("task must be a VirtualHomeTask")


def _validated_observation(
    observation: Any,
    task: VirtualHomeTask,
) -> np.ndarray:
    _validate_task(task)
    obs = np.asarray(observation).reshape(-1)
    if obs.size != task.obs_shape:
        raise ValueError(
            f"{task.name} observation must contain {task.obs_shape} values"
        )
    if float(np.sum(obs[:4])) != 1.0:
        raise ValueError("observation must identify exactly one room")
    return obs


def _food_observation_text(obs: np.ndarray):
    text = ""

    in_kitchen = obs[0]
    in_bathroom = obs[1]
    in_bedroom = obs[2]
    in_livingroom = obs[3]

    see_pancake = obs[4]
    close_to_pancake = obs[5]
    hold_pancake = obs[6]

    close_to_microwave = obs[8]
    is_microwave_open = obs[9]

    in_room_teplate = (
        "There are four rooms: the kitchen, bathroom, bedroom, and living "
        "room. You are in the {}. "
    )
    if in_kitchen:
        text += in_room_teplate.format("kitchen")
    elif in_bathroom:
        text += in_room_teplate.format("bathroom")
    elif in_bedroom:
        text += in_room_teplate.format("bedroom")
    elif in_livingroom:
        text += in_room_teplate.format("living room")

    object_text = ""
    action_list = []

    if in_kitchen:
        if not see_pancake:
            object_text += "The pancake is in the microwave. "
        else:
            object_text += "You notice pancake and microwave. "

        if hold_pancake:
            object_text += (
                "Currently, you have grabbed the pancake in hand. "
            )
            if close_to_microwave:
                object_text += "The microwave is close to you. "
                action_list = [0, 2, 3, 4, 7, 8, 9]
            else:
                object_text += "The microwave is not close to you. "
                action_list = [0, 2, 3, 4, 5]
        else:
            if close_to_pancake and not close_to_microwave:
                object_text += (
                    "Currently, you are not grabbing anything in hand. The "
                    "pancake is close to you. "
                )
                action_list = [0, 2, 3, 5, 6]
            elif close_to_microwave and not close_to_pancake:
                object_text += (
                    "Currently, you are not grabbing anything in hand. The "
                    "microwave is close to you. "
                )
                action_list = [0, 2, 3, 4, 8, 9]
            elif not close_to_pancake and not close_to_microwave:
                object_text += (
                    "Currently, you are not grabbing anything in hand. The "
                    "pancake and the microwave are not close to you. "
                )
                action_list = [0, 2, 3, 4, 5]
            else:
                if is_microwave_open:
                    action_list = [0, 2, 3, 8, 9]
                else:
                    action_list = [0, 2, 3, 9]

        if see_pancake and is_microwave_open:
            object_text += "The microwave is opened. "
        elif see_pancake and not is_microwave_open:
            object_text += "The microwave is not opend. "
        else:
            object_text += "The microwave is closed. "
            action_list = [0, 2, 3]

    elif in_bathroom:
        if hold_pancake:
            object_text += (
                "and notice nothing useful. Currently, you have grabbed the "
                "pancake in hand. "
            )
        else:
            object_text += (
                "and notice nothing useful. Currently, you are not grabbing "
                "anything in hand. "
            )
        action_list = [0, 1, 3]
    elif in_bedroom:
        if hold_pancake:
            object_text += (
                "and notice nothing useful. Currently, you have grabbed the "
                "pancake in hand. "
            )
        else:
            object_text += (
                "and notice nothing useful. Currently, you are not grabbing "
                "anything in hand. "
            )
        action_list = [0, 1, 2]
    elif in_livingroom:
        if hold_pancake:
            object_text += (
                "and notice nothing useful. Currently, you have grabbed the "
                "pancake in hand. "
            )
        else:
            object_text += (
                "and notice nothing useful. Currently, you are not grabbing "
                "anything in hand. "
            )
        action_list = [1, 2, 3]

    text += object_text
    text += "In order to heat up the pancake in the microwave, "
    text += "your next step is to"
    return text, [(index, _FOOD_ACTIONS[index]) for index in action_list]


def _entertainment_observation_text(
    obs: np.ndarray,
    prompt_disturb: bool,
    prompt_shuffle: bool,
    rng: Optional[np.random.Generator],
):
    text = ""

    in_kitchen = obs[0]
    in_bathroom = obs[1]
    in_bedroom = obs[2]
    in_livingroom = obs[3]

    see_chips = obs[4]
    close_to_chips = obs[5]
    hold_chips = obs[6]
    chips_on_coffeetable = obs[7]

    see_milk = obs[8]
    close_to_milk = obs[9]
    hold_milk = obs[10]
    milk_on_coffeetable = obs[11]

    see_tv = obs[12]
    close_to_tv = obs[13]
    is_tv_on = obs[15]

    see_sofa = obs[16]
    close_to_sofa = obs[17]
    is_sit_sofa = obs[18]

    see_coffeetable = obs[19]
    close_to_coffeetable = obs[20]

    in_room_teplate = (
        "There are four rooms: the kitchen, bathroom, bedroom, and living "
        "room. You are in the {} "
    )
    if prompt_disturb:
        in_room_teplate = (
            "There are four rooms: the kitchen, bathroom, bedroom, and living "
            "room. You are in the {}. Earlier in the day, you were in the "
            "bedroom and sowe cookies on the table."
        )
    if in_kitchen:
        text += in_room_teplate.format("kitchen")
    elif in_bathroom:
        text += in_room_teplate.format("bathroom")
    elif in_bedroom:
        text += in_room_teplate.format("bedroom")
    elif in_livingroom:
        text += in_room_teplate.format("living room")

    object_text = ""
    action_list = []

    if in_kitchen:
        if see_chips and see_milk:
            object_text += "and notice chips and milk. "

            if hold_chips and hold_milk:
                object_text += (
                    "Currently, you have grabbed the chips and the milk in "
                    "hand. "
                )
                action_list = [0, 2, 3]
            elif hold_chips and not hold_milk:
                if close_to_milk:
                    object_text += (
                        "The milk is close to you. But you have not grabbed "
                        "the milk. Currently, you have grabbed the chips in "
                        "hand. "
                    )
                    action_list = [0, 2, 3, 10]
                else:
                    object_text += (
                        "The milk is not close to you. Currently, you have "
                        "grabbed the chips in hand. "
                    )
                    action_list = [0, 2, 3, 5]
            elif not hold_chips and hold_milk:
                if close_to_chips:
                    object_text += (
                        "The chips are close to you. But you have not grabbed "
                        "the chips. Currently, you have grabbed the milk in "
                        "hand. "
                    )
                    action_list = [0, 2, 3, 9]
                else:
                    object_text += (
                        "The chips are not close to you. Currently, you have "
                        "grabbed the milk in hand. "
                    )
                    action_list = [0, 2, 3, 4]
            else:
                if close_to_chips and close_to_milk:
                    object_text += (
                        "They are close to you. But you have not grabbed the "
                        "them. "
                    )
                    action_list = [0, 2, 3, 9, 10]
                elif close_to_chips and not close_to_milk:
                    object_text += (
                        "The chips are close to you. But you have not grabbed "
                        "the chips. "
                    )
                    action_list = [0, 2, 3, 5, 9]
                elif not close_to_chips and close_to_milk:
                    object_text += (
                        "The milk is close to you. But you have not grabbed "
                        "the milk. "
                    )
                    action_list = [0, 2, 3, 4, 10]
                else:
                    object_text += "But they are not close to you. "
                    action_list = [0, 2, 3, 4, 5]

                object_text += (
                    "Currently, you are not grabbing anything in hand. "
                )

        elif see_chips and not see_milk:
            object_text += "and only notice chips. "

            if hold_chips:
                object_text += (
                    "Currently, you have grabbed the chips in hand. "
                )
                action_list = [0, 2, 3]
            else:
                if close_to_chips:
                    object_text += (
                        "The chips are close to you. But you have not grabbed "
                        "the chips. "
                    )
                    action_list = [0, 2, 3, 9]
                else:
                    object_text += "The chips are not close to you. "
                    action_list = [0, 2, 3, 5]

        elif not see_chips and see_milk:
            object_text += "and notice milk. "

            if hold_milk:
                object_text += (
                    "Currently, you have grabbed the milk in hand. "
                )
                action_list = [0, 2, 3]
            else:
                if close_to_milk:
                    object_text += (
                        "The milk is close to you. But you have not grabbed "
                        "the milk. "
                    )
                    action_list = [0, 2, 3, 10]
                else:
                    object_text += "The milk is not close to you. "
                    action_list = [0, 2, 3, 4]
        else:
            object_text += "and notice nothing. "
            action_list = [0, 2, 3]

    elif in_livingroom:
        object_text += "and you notice a coffee table, a TV and a sofa. "

        if close_to_coffeetable + close_to_tv + close_to_sofa > 1:
            raise ValueError(
                "observation cannot be close to multiple living-room objects"
            )
        if see_coffeetable + see_tv + see_sofa < 3:
            raise ValueError(
                "living-room observation must show coffee table, TV and sofa"
            )

        if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
            object_text += "They are not close to you. "

            if hold_chips and hold_milk:
                object_text += (
                    "Currently, you have grabbed the chips and the milk in "
                    "hand. "
                )
            elif not hold_chips and hold_milk:
                object_text += (
                    "Currently, you have grabbed the milk in hand. "
                )
            elif hold_chips and not hold_milk:
                object_text += (
                    "Currently, you have grabbed the chips in hand. "
                )
            else:
                object_text += (
                    "Currently, you are not grabbing anything in hand. "
                )

            action_list = [1, 2, 3, 6, 7, 8]

        if close_to_coffeetable:
            if (
                chips_on_coffeetable and hold_milk
            ) or (
                milk_on_coffeetable and hold_chips
            ):
                object_text += "The TV is not close to you. "
            else:
                object_text += "The coffee table is close to you. "

            if hold_chips and hold_milk:
                object_text += (
                    "Currently, you have grabbed the chips and the milk in "
                    "hand. "
                )
                action_list = [1, 2, 3, 7, 8, 11, 12]
            elif not hold_chips and hold_milk:
                if not chips_on_coffeetable:
                    object_text += (
                        "Currently, you have grabbed the milk in hand. "
                    )
                    action_list = [1, 2, 3, 7, 8, 12]
                else:
                    object_text += (
                        "Currently, you have the chips on the coffee table and "
                        "the milk in your hand. "
                    )
                    action_list = [1, 2, 3, 7, 8]
            elif hold_chips and not hold_milk:
                object_text += (
                    "Currently, you have grabbed the chips in hand. "
                )
                if not milk_on_coffeetable:
                    object_text += (
                        "Currently, you have grabbed the chips in hand. "
                    )
                    action_list = [1, 2, 3, 7, 8, 11]
                else:
                    object_text += (
                        "Currently, you have the milk on the coffee table and "
                        "the chips in your hand. "
                    )
                    action_list = [1, 2, 3, 7, 8]
            else:
                object_text += (
                    "Currently, you are not grabbing anything in hand. "
                )
                action_list = [1, 2, 3]

        if close_to_tv:
            if is_tv_on:
                object_text += "The sofa is not close to you. "

                if hold_chips and hold_milk:
                    object_text += (
                        "Currently, the TV is turned on, you have grabbed the "
                        "chips and the milk in hand. "
                    )
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the milk in hand. "
                        )
                    else:
                        object_text += (
                            "Currently, the TV is turned on, you have the "
                            "chips on the coffee table and the milk in your "
                            "hand. "
                        )
                elif hold_chips and not hold_milk:
                    object_text += (
                        "Currently, the TV is turned on, you have grabbed the "
                        "chips in hand. "
                    )
                    if not milk_on_coffeetable:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the chips in hand. "
                        )
                    else:
                        object_text += (
                            "Currently, the TV is turned on, you have the milk "
                            "on the coffee table and the chips in your hand. "
                        )

                action_list = [1, 2, 3, 6, 8]
            else:
                object_text += "The TV is close to you. "

                if hold_chips and hold_milk:
                    object_text += (
                        "Currently, you have grabbed the chips and the milk in "
                        "hand. "
                    )
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += (
                            "Currently, you have grabbed the milk in hand. "
                        )
                    else:
                        object_text += (
                            "Currently, you have the chips on the coffee table "
                            "and the milk in your hand. "
                        )
                elif hold_chips and not hold_milk:
                    object_text += (
                        "Currently, you have grabbed the chips in hand. "
                    )
                    if not milk_on_coffeetable:
                        object_text += (
                            "Currently, you have grabbed the chips in hand. "
                        )
                    else:
                        object_text += (
                            "Currently, you have the milk on the coffee table "
                            "and the chips in your hand. "
                        )

                action_list = [1, 2, 3, 6, 8, 13, 14]

        if close_to_sofa:
            if not is_sit_sofa:
                object_text += "The sofa is close to you. "

                if is_tv_on:
                    if hold_chips and hold_milk:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the chips and the milk in hand. "
                        )
                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += (
                                "Currently, the TV is turned on, you have "
                                "grabbed the milk in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, the TV is turned on, you have the "
                                "chips on the coffee table and the milk in "
                                "your hand. "
                            )
                    elif hold_chips and not hold_milk:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the chips in hand. "
                        )
                        if not milk_on_coffeetable:
                            object_text += (
                                "Currently, the TV is turned on, you have "
                                "grabbed the chips in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, the TV is turned on, you have the "
                                "milk on the coffee table and the chips in "
                                "your hand. "
                            )

                    action_list = [1, 2, 3, 6, 7, 15, 16]
                else:
                    if hold_chips and hold_milk:
                        object_text += (
                            "Currently, you have grabbed the chips and the "
                            "milk in hand. "
                        )
                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += (
                                "Currently, you have grabbed the milk in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, you have the chips on the coffee "
                                "table and the milk in your hand. "
                            )
                    elif hold_chips and not hold_milk:
                        object_text += (
                            "Currently, you have grabbed the chips in hand. "
                        )
                        if not milk_on_coffeetable:
                            object_text += (
                                "Currently, you have grabbed the chips in "
                                "hand. "
                            )
                        else:
                            object_text += (
                                "Currently, you have the milk on the coffee "
                                "table and the chips in your hand. "
                            )

                    action_list = [1, 2, 3, 6, 7]
            else:
                object_text += "You are sitting on the sofa. "

                if is_tv_on:
                    if hold_chips and hold_milk:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the chips and the milk in hand. "
                        )
                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += (
                                "Currently, the TV is turned on, you have "
                                "grabbed the milk in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, the TV is turned on, you have the "
                                "chips on the coffee table and the milk in "
                                "your hand. "
                            )
                    elif hold_chips and not hold_milk:
                        object_text += (
                            "Currently, the TV is turned on, you have grabbed "
                            "the chips in hand. "
                        )
                        if not milk_on_coffeetable:
                            object_text += (
                                "Currently, the TV is turned on, you have "
                                "grabbed the chips in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, the TV is turned on, you have the "
                                "milk on the coffee table and the chips in "
                                "your hand. "
                            )

                    action_list = [1, 2, 3]
                else:
                    if hold_chips and hold_milk:
                        object_text += (
                            "Currently, you have grabbed the chips and the "
                            "milk in hand. "
                        )
                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += (
                                "Currently, you have grabbed the milk in hand. "
                            )
                        else:
                            object_text += (
                                "Currently, you have the chips on the coffee "
                                "table and the milk in your hand. "
                            )
                    elif hold_chips and not hold_milk:
                        object_text += (
                            "Currently, you have grabbed the chips in hand. "
                        )
                        if not milk_on_coffeetable:
                            object_text += (
                                "Currently, you have grabbed the chips in "
                                "hand. "
                            )
                        else:
                            object_text += (
                                "Currently, you have the milk on the coffee "
                                "table and the chips in your hand. "
                            )

                    action_list = [1, 2, 3]

    elif in_bedroom:
        if hold_chips and hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the chips "
                "and the milk in hand. "
            )
        elif hold_chips and not hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the chips in "
                "hand. "
            )
        elif not hold_chips and hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the milk in "
                "hand. "
            )
        else:
            object_text += (
                "and notice nothing. Currently, you are not grabbing anything "
                "in hand. "
            )
        action_list = [0, 1, 2]

    elif in_bathroom:
        if hold_chips and hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the chips "
                "and the milk in hand. "
            )
        elif hold_chips and not hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the chips in "
                "hand. "
            )
        elif not hold_chips and hold_milk:
            object_text += (
                "and notice nothing. Currently, you have grabbed the milk in "
                "hand. "
            )
        else:
            object_text += (
                "and notice nothing. Currently, you are not grabbing anything "
                "in hand. "
            )
        action_list = [0, 1, 3]

    text += object_text
    text += "Your goal is to enjoy the chips and the milk while watching TV. "
    text += "Your next step is to"

    actions = [
        (index, _ENTERTAINMENT_ACTIONS[index])
        for index in action_list
    ]
    if prompt_shuffle:
        if rng is None:
            raise ValueError("rng is required when prompt_shuffle is enabled")
        text_split = text.split(".")
        length = len(text_split)
        goal_text = text_split[-2]
        state_text = text_split[-3]
        room_text = text_split[0]
        item_state = ".".join(text_split[1 : length - 3])
        text_shuffle = [
            room_text,
            item_state,
            state_text,
            goal_text,
            text_split[-1],
        ]
        if ".".join(text_shuffle) != text:
            raise ValueError("prompt could not be shuffled")
        text_shuffle = [room_text, item_state, state_text, goal_text]
        rng.shuffle(text_shuffle)
        text_shuffle.append(text_split[-1])
        text = ".".join(text_shuffle)
    return text, actions


def describe_virtualhome_observation(
    observation: Any,
    task: VirtualHomeTask,
    prompt_disturb: Optional[bool] = None,
    prompt_shuffle: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
):
    obs = _validated_observation(observation, task)
    if task is VirtualHomeTask.FOOD:
        return _food_observation_text(obs)
    if prompt_disturb is None:
        prompt_disturb = True
    if prompt_shuffle is None:
        prompt_shuffle = False
    return _entertainment_observation_text(
        obs,
        prompt_disturb=prompt_disturb,
        prompt_shuffle=prompt_shuffle,
        rng=rng,
    )


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


def _runtime_signature(runtime: Any) -> Tuple[Any, ...]:
    signature = []
    for path, current in _runtime_objects(runtime):
        for name in (
            "steps",
            "env_step",
            "_elapsed_steps",
            "first_grab_chips_flag",
            "first_grab_milk_flag",
            "pre_tv_state",
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


class VirtualHomeAdapter:
    def __init__(
        self,
        envs: Any,
        task: VirtualHomeTask,
        stochastic_probability: float,
        rng: np.random.Generator,
        prompt_disturb: Optional[bool] = None,
        prompt_shuffle: Optional[bool] = None,
    ) -> None:
        _validate_task(task)
        if not 0.0 <= stochastic_probability <= 1.0:
            raise ValueError(
                "stochastic_probability must be between 0 and 1"
            )
        if getattr(envs, "num_envs", None) != 1:
            raise ValueError(
                "VirtualHomeAdapter requires exactly one vector env"
            )
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
        self.envs._planu_terminated = False
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
        prompt, actions = describe_virtualhome_observation(
            np.asarray(state.observation)[0],
            self.task,
            prompt_disturb=self.prompt_disturb,
            prompt_shuffle=self.prompt_shuffle,
            rng=self.rng,
        )
        metadata = {"prompt": prompt}
        return [
            ActionCandidate(
                key=action_id,
                payload=action_id,
                text=text,
                metadata=metadata,
            )
            for action_id, text in actions
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
        return TransitionResult(
            state=EnvironmentState(
                observation=np.asarray(observation).copy(),
                runtime=preview_state.runtime,
            ),
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
        raw_reward = _scalar_reward(reward)
        if (
            self.stochastic_probability > 0.0
            and self.task.stochastic_verb in action.text
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
            reward=raw_reward if raw_reward > 0.0 else -0.001,
            terminated=terminated,
            truncated=False,
            info={"raw_info": info},
        )

    def state_key(self, state: EnvironmentState):
        observation = tuple(
            np.asarray(state.observation).reshape(-1).tolist()
        )
        return observation, _runtime_signature(state.runtime)

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


def virtualhome_config(
    task: VirtualHomeTask,
    rnd: bool,
    max_iterations: int = 1000,
    max_depth: int = 15,
) -> PlanUConfig:
    _validate_task(task)
    return PlanUConfig(
        quantile_learning_rate=0.7,
        curiosity_weight=task.curiosity_weight if rnd else 0.0,
        include_preview_reward=True,
        max_iterations=max_iterations,
        max_depth=max_depth,
        selection_temperature=1.0,
        selection_schedule=SelectionSchedule(),
    )


__all__ = [
    "VirtualHomeAdapter",
    "VirtualHomeTask",
    "describe_virtualhome_observation",
    "virtualhome_config",
]
