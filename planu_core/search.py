from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

import numpy as np

from .backup import backup_trajectory
from .config import PlanUConfig
from .distribution import QuantileDistribution
from .interfaces import ActionScorer, EnvironmentAdapter, EnvironmentState
from .nodes import ActionNode, LanguageNode
from .selection import select_action


@dataclass
class TrajectoryResult:
    actions: List[Any]
    rewards: List[float]
    terminated: bool
    truncated: bool
    final_observation: Any
    selection_scores: List[Dict[Hashable, float]]
    action_path: List[ActionNode]
    state_path: List[LanguageNode]


class PlanUSearch:
    def __init__(
        self,
        adapter: EnvironmentAdapter,
        scorer: ActionScorer,
        config: PlanUConfig,
    ):
        self.adapter = adapter
        self.scorer = scorer
        self.config = config
        self.root: Optional[LanguageNode] = None

    def _ensure_root(self, state: EnvironmentState) -> LanguageNode:
        key = self.adapter.state_key(state)
        if self.root is None:
            self.root = LanguageNode(state.observation, key)
        elif self.root.state_key != key:
            raise ValueError("reset state does not match persistent PlanU root")
        return self.root

    def expand(
        self,
        node: LanguageNode,
        state: EnvironmentState,
        rng: np.random.Generator,
    ) -> None:
        if node.terminated or node.truncated or node.children:
            return

        candidates = list(self.adapter.actions(state, node.visit_count))
        if not candidates:
            return
        action_keys = [candidate.key for candidate in candidates]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("duplicate action key")
        try:
            priors = np.asarray(
                self.scorer.score(state.observation, candidates),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("action scores must be finite numbers") from error
        if priors.shape != (len(candidates),):
            raise ValueError("scorer returned the wrong number of action scores")
        if not np.all(np.isfinite(priors)):
            raise ValueError("action scores must be finite")
        for candidate, prior in zip(candidates, priors):
            preview = self.adapter.preview(
                self.adapter.clone(state),
                candidate,
                rng,
            )
            preview_terminated = (
                preview.terminated
                or self.adapter.is_terminal(preview.state)
            )
            initial = float(prior)
            if self.config.include_preview_reward:
                initial += float(preview.reward)
            action = ActionNode(
                parent=node,
                action=candidate,
                distribution=QuantileDistribution.from_scalar(
                    initial,
                    self.config.n_quantiles,
                    self.config.value_min,
                    self.config.value_max,
                ),
                preview_state=preview.state.observation,
            )
            if preview.info.get("record_outcome", True):
                action.get_or_create_outcome(
                    preview.state.observation,
                    self.adapter.state_key(preview.state),
                    preview_terminated,
                    preview.truncated,
                    increment_visit=False,
                )
            node.children[candidate.key] = action

    def run_iteration(
        self,
        iteration: int,
        rng: np.random.Generator,
        reset_seed: Optional[int] = None,
    ) -> TrajectoryResult:
        state = self.adapter.reset(reset_seed)
        node = self._ensure_root(state)
        if self.adapter.is_terminal(state):
            node.terminated = True
            backup_trajectory([], [], self.config)
            return TrajectoryResult(
                actions=[],
                rewards=[],
                terminated=True,
                truncated=False,
                final_observation=state.observation,
                selection_scores=[],
                action_path=[],
                state_path=[node],
            )

        action_path = []
        state_path = [node]
        actions = []
        rewards = []
        score_history = []
        terminated = False
        truncated = False

        for _ in range(self.config.max_depth):
            node.visit_count += 1
            self.expand(node, state, rng)
            if not node.children:
                node.truncated = True
                truncated = True
                break
            action_node, scores = select_action(
                node,
                self.config,
                iteration,
                rng,
            )
            result = self.adapter.step(state, action_node.action, rng)
            terminated = result.terminated or self.adapter.is_terminal(
                result.state
            )
            truncated = result.truncated
            next_key = self.adapter.state_key(result.state)
            existing_outcome = action_node.children.get(next_key)
            outcome_truncated = truncated or (
                existing_outcome is not None and existing_outcome.truncated
            )
            next_node = action_node.get_or_create_outcome(
                result.state.observation,
                next_key,
                terminated,
                outcome_truncated,
            )
            action_path.append(action_node)
            actions.append(action_node.action.key)
            rewards.append(float(result.reward))
            score_history.append(scores)
            state = result.state
            node = next_node
            state_path.append(node)
            if terminated or truncated:
                break
        else:
            node.truncated = True
            truncated = True

        backup_trajectory(action_path, rewards, self.config)
        return TrajectoryResult(
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            final_observation=state.observation,
            selection_scores=score_history,
            action_path=action_path,
            state_path=state_path,
        )
