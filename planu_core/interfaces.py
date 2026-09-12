from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Iterator, Mapping, Optional, Protocol, Sequence

import numpy as np


class _FrozenMapping(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        return type(self), (dict(self._values),)

    def __repr__(self) -> str:
        return repr(dict(self._values))


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
            _FrozenMapping(self.metadata),
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


class ActionProvider(Protocol):
    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        ...


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
    ) -> TransitionResult:
        ...

    def state_key(self, state: EnvironmentState) -> Hashable:
        ...

    def is_terminal(self, state: EnvironmentState) -> bool:
        ...

    def is_truncated(self, state: EnvironmentState) -> bool:
        ...


class ActionScorer(Protocol):
    def score(
        self,
        observation: Any,
        actions: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        ...


class DistributionScorer(Protocol):
    def score_distributions(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
        levels: Sequence[float],
    ) -> Sequence[Sequence[float]]:
        ...


class CuriosityProvider(Protocol):
    def score(self, observation: Any) -> float:
        ...

    def observe(self, observation: Any) -> None:
        ...

    def train(self) -> Optional[Mapping[str, float]]:
        ...
