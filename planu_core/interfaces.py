from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ActionCandidate:
    key: Hashable
    payload: Any
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    def reset(self) -> EnvironmentState:
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
    ) -> TransitionResult:
        ...

    def step(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
    ) -> TransitionResult:
        ...

    def state_key(self, observation: Any) -> Hashable:
        ...


class ActionScorer(Protocol):
    def score(
        self,
        observation: Any,
        actions: Sequence[ActionCandidate],
    ):
        ...


class CuriosityProvider(Protocol):
    def score(self, observation: Any) -> float:
        ...

    def observe(self, observation: Any) -> None:
        ...

    def train(self) -> None:
        ...
