from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    key: Hashable
    payload: Any = field(compare=False, hash=False)
    text: str = field(compare=False, hash=False)
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass
class EnvironmentState:
    observation: Any
    runtime: Any


@dataclass
class TransitionResult:
    state: EnvironmentState
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class EnvironmentAdapter(Protocol):
    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        ...

    def clone(self, state: EnvironmentState) -> EnvironmentState:
        ...

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        ...

    def preview(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        ...

    def step(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
        state_visit_count: int = 0,
    ) -> TransitionResult:
        ...

    def state_key(self, state: EnvironmentState) -> Hashable:
        ...

    def is_terminal(self, state: EnvironmentState) -> bool:
        ...


class ActionScorer(Protocol):
    def score(
        self,
        observation: Any,
        actions: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        ...


class CuriosityProvider(Protocol):
    def score(self, observation: Any) -> float:
        ...

    def observe(self, observation: Any) -> None:
        ...

    def train(self) -> Optional[Mapping[str, float]]:
        ...
