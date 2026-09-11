from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

from .distribution import QuantileDistribution
from .interfaces import ActionCandidate


@dataclass(eq=False)
class LanguageNode:
    state: Any
    state_key: Hashable
    parent: Optional["ActionNode"] = None
    children: Dict[Hashable, "ActionNode"] = field(default_factory=dict)
    visit_count: int = 0
    terminated: bool = False
    truncated: bool = False
    outcome_visits: int = 0


@dataclass(eq=False)
class ActionNode:
    parent: LanguageNode
    action: ActionCandidate
    distribution: QuantileDistribution
    children: Dict[Hashable, LanguageNode] = field(default_factory=dict)
    visit_count: int = 0
    cumulative_returns: List[float] = field(default_factory=list)
    preview_state: Any = None

    def get_or_create_outcome(
        self,
        state: Any,
        state_key: Hashable,
        terminated: bool,
        truncated: bool,
        increment_visit: bool = True,
    ) -> LanguageNode:
        child = self.children.get(state_key)
        if child is None:
            child = LanguageNode(
                state=state,
                state_key=state_key,
                parent=self,
                terminated=terminated,
                truncated=truncated,
            )
            self.children[state_key] = child
        elif child.terminated != terminated or child.truncated != truncated:
            raise ValueError(
                "inconsistent terminated/truncated flags for existing outcome"
            )
        if increment_visit:
            child.outcome_visits += 1
        return child
