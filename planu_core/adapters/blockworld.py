import copy
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Hashable, Optional, Sequence, Tuple

import numpy as np

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentState,
    TransitionResult,
)


def _scalar_finite(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("{} must be scalar".format(name))
    try:
        scalar = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be finite".format(name)) from error
    if not math.isfinite(scalar):
        raise ValueError("{} must be finite".format(name))
    return scalar


def _result_pair(value: Any, name: str) -> Tuple[Any, Any]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("{} result must be a pair".format(name))
    return value[0], value[1]


def _copied_mapping(value: Any, name: str):
    if not isinstance(value, Mapping):
        raise ValueError("{} details must be a mapping".format(name))
    return copy.deepcopy(dict(value))


def _type_name(value: Any) -> Tuple[str, str]:
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__


def _freeze_state(value: Any) -> Hashable:
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            tuple(value.shape),
            _freeze_state(value.tolist()),
        )
    if isinstance(value, np.generic):
        return _freeze_state(value.item())
    if (
        isinstance(value, tuple)
        and hasattr(value, "_fields")
        and isinstance(value._fields, tuple)
    ):
        return (
            "namedtuple",
            _type_name(value),
            tuple(
                (name, _freeze_state(getattr(value, name)))
                for name in value._fields
            ),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            _type_name(value),
            tuple(
                (field.name, _freeze_state(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        frozen_items = [
            (_freeze_state(key), _freeze_state(item))
            for key, item in value.items()
        ]
        frozen_items.sort(key=lambda pair: repr(pair[0]))
        return ("mapping", tuple(frozen_items))
    if isinstance(value, list):
        return ("list", tuple(_freeze_state(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_state(item) for item in value))
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    raise TypeError(
        "unsupported state value: {}".format(type(value).__qualname__)
    )


class BlockWorldScorer:
    def score(
        self,
        observation: Any,
        actions: Sequence[ActionCandidate],
    ) -> np.ndarray:
        del observation
        return np.asarray(
            [action.metadata["prior_score"] for action in actions],
            dtype=np.float64,
        )


class BlockWorldAdapter:
    def __init__(self, world_model: Any, search_config: Any) -> None:
        self.world_model = world_model
        self.search_config = search_config

    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        if seed is not None and hasattr(self.world_model, "rng"):
            self.world_model.rng = np.random.default_rng(seed)
        return EnvironmentState(self.world_model.init_state(), None)

    def clone(self, state: EnvironmentState) -> EnvironmentState:
        return EnvironmentState(
            copy.deepcopy(state.observation),
            copy.deepcopy(state.runtime),
        )

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ):
        actions = list(self.search_config.get_actions(state.observation))
        seen = set()
        for action in actions:
            try:
                hash(action)
            except TypeError as error:
                raise ValueError("action key must be hashable") from error
            if action in seen:
                raise ValueError("duplicate action key")
            seen.add(action)

        node_view = SimpleNamespace(
            state=state.observation,
            cum_rewards=[0.0] * max(0, state_visit_count - 1),
        )
        candidates = []
        for action in actions:
            prior, details = _result_pair(
                self.search_config.fast_reward(node_view, action),
                "fast reward",
            )
            metadata = {
                "prior_score": _scalar_finite(prior, "prior score"),
                "fast_reward_details": _copied_mapping(
                    details,
                    "fast reward",
                ),
                "node_view": node_view,
            }
            candidates.append(
                ActionCandidate(
                    key=action,
                    payload=action,
                    text=str(action),
                    metadata=metadata,
                )
            )
        return candidates

    def preview(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        del action, rng
        preview_state = self.clone(state)
        return TransitionResult(
            state=preview_state,
            reward=0.0,
            terminated=bool(
                self.world_model.is_terminal(preview_state.observation)
            ),
            truncated=False,
            info={"record_outcome": False},
        )

    def step(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        if hasattr(self.world_model, "rng"):
            self.world_model.rng = rng
        raw_result = self.world_model.step(
            state.observation,
            action.payload,
        )
        if (
            type(raw_result) is tuple
            and len(raw_result) == 2
            and isinstance(raw_result[1], Mapping)
        ):
            next_observation, raw_aux = raw_result
            aux = copy.deepcopy(dict(raw_aux))
        else:
            next_observation = raw_result
            aux = {}

        reward_kwargs = dict(action.metadata["fast_reward_details"])
        reward_kwargs.update(aux)
        raw_reward = self.search_config.reward(
            action.metadata["node_view"],
            action.payload,
            **reward_kwargs
        )
        reward, raw_details = _result_pair(raw_reward, "reward")
        details = _copied_mapping(raw_details, "reward")
        info = dict(details)
        info.update(aux)
        return TransitionResult(
            state=EnvironmentState(next_observation, None),
            reward=_scalar_finite(reward, "reward"),
            terminated=bool(self.world_model.is_terminal(next_observation)),
            truncated=False,
            info=info,
        )

    def state_key(self, state: EnvironmentState) -> Hashable:
        return _freeze_state(state.observation)

    def is_terminal(self, state: EnvironmentState) -> bool:
        return bool(self.world_model.is_terminal(state.observation))


def blockworld_config(
    n_iters: int,
    depth: int,
    n_atoms: int,
    risk: float,
    lr: float = 0.75,
    vmin: float = -10.0,
    vmax: float = 100.0,
) -> PlanUConfig:
    return PlanUConfig(
        n_quantiles=n_atoms,
        value_min=vmin,
        value_max=vmax,
        quantile_learning_rate=lr,
        discount=1.0,
        include_preview_reward=False,
        max_depth=depth,
        max_iterations=n_iters,
        risk_distortion=risk,
        selection_schedule=SelectionSchedule(),
    )


__all__ = [
    "BlockWorldAdapter",
    "BlockWorldScorer",
    "blockworld_config",
]
