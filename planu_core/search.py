import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from .backup import backup_trajectory
from .config import PlanUConfig
from .curiosity import NullCuriosity
from .distribution import QuantileDistribution
from .interfaces import (
    ActionScorer,
    CuriosityProvider,
    EnvironmentAdapter,
    EnvironmentState,
)
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


class _MutationJournal:
    def __init__(self) -> None:
        self._language_nodes: Dict[
            int,
            Tuple[
                LanguageNode,
                int,
                bool,
                bool,
                int,
                Dict[Hashable, ActionNode],
            ],
        ] = {}
        self._action_nodes: Dict[
            int,
            Tuple[
                ActionNode,
                Dict[Hashable, LanguageNode],
                List[Tuple[LanguageNode, int, bool, bool]],
            ],
        ] = {}

    def snapshot_language(self, node: LanguageNode) -> None:
        node_id = id(node)
        if node_id not in self._language_nodes:
            self._language_nodes[node_id] = (
                node,
                node.visit_count,
                node.terminated,
                node.truncated,
                node.outcome_visits,
                node.children.copy(),
            )

    def snapshot_action_outcomes(self, node: ActionNode) -> None:
        node_id = id(node)
        if node_id not in self._action_nodes:
            self._action_nodes[node_id] = (
                node,
                node.children.copy(),
                [
                    (
                        child,
                        child.outcome_visits,
                        child.terminated,
                        child.truncated,
                    )
                    for child in node.children.values()
                ],
            )

    def rollback(self) -> None:
        for (
            node,
            visit_count,
            terminated,
            truncated,
            outcome_visits,
            children,
        ) in self._language_nodes.values():
            node.visit_count = visit_count
            node.terminated = terminated
            node.truncated = truncated
            node.outcome_visits = outcome_visits
            node.children.clear()
            node.children.update(children)

        for node, children, outcome_states in self._action_nodes.values():
            node.children.clear()
            node.children.update(children)
            for child, outcome_visits, terminated, truncated in outcome_states:
                child.outcome_visits = outcome_visits
                child.terminated = terminated
                child.truncated = truncated


class PlanUSearch:
    def __init__(
        self,
        adapter: EnvironmentAdapter,
        scorer: ActionScorer,
        config: PlanUConfig,
        curiosity: Optional[CuriosityProvider] = None,
    ):
        self.adapter = adapter
        self.scorer = scorer
        self.config = config
        self.curiosity = NullCuriosity() if curiosity is None else curiosity
        self.root: Optional[LanguageNode] = None

    def _ensure_root(self, state: EnvironmentState) -> LanguageNode:
        key = self.adapter.state_key(state)
        if self.root is None:
            self.root = LanguageNode(copy.deepcopy(state.observation), key)
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
        pending_children: Dict[Hashable, ActionNode] = OrderedDict()
        preview_rng = copy.deepcopy(rng)
        for candidate, prior in zip(candidates, priors):
            preview = self.adapter.preview(
                state,
                candidate,
                copy.deepcopy(preview_rng),
            )
            preview_terminated = (
                preview.terminated
                or self.adapter.is_terminal(preview.state)
            )
            preview_key = self.adapter.state_key(preview.state)
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
                preview_state=copy.deepcopy(preview.state.observation),
            )
            if preview.info.get("record_outcome", True):
                action.get_or_create_outcome(
                    copy.deepcopy(preview.state.observation),
                    preview_key,
                    preview_terminated,
                    preview.truncated,
                    increment_visit=False,
                )
            pending_children[candidate.key] = action
        node.children.update(pending_children)

    def run_iteration(
        self,
        iteration: int,
        rng: np.random.Generator,
        reset_seed: Optional[int] = None,
    ) -> TrajectoryResult:
        root_before = self.root
        journal = _MutationJournal()
        try:
            result, executed_observations = self._run_iteration(
                iteration,
                rng,
                reset_seed,
                journal,
            )
        except BaseException:
            journal.rollback()
            self.root = root_before
            raise
        # Curiosity is external learner state; failures here do not roll back
        # the tree transaction already committed by a successful backup.
        for observation in executed_observations:
            self.curiosity.observe(observation)
        self.curiosity.train()
        return result

    def _run_iteration(
        self,
        iteration: int,
        rng: np.random.Generator,
        reset_seed: Optional[int],
        journal: _MutationJournal,
    ) -> Tuple[TrajectoryResult, List[Any]]:
        state = self.adapter.reset(reset_seed)
        node = self._ensure_root(state)
        if self.adapter.is_terminal(state):
            journal.snapshot_language(node)
            node.terminated = True
            final_observation = copy.deepcopy(state.observation)
            backup_trajectory([], [], self.config)
            return (
                TrajectoryResult(
                    actions=[],
                    rewards=[],
                    terminated=True,
                    truncated=False,
                    final_observation=final_observation,
                    selection_scores=[],
                    action_path=[],
                    state_path=[node],
                ),
                [],
            )

        action_path = []
        state_path = [node]
        actions = []
        rewards = []
        executed_observations = []
        score_history = []
        terminated = False
        truncated = False

        for _ in range(self.config.max_depth):
            journal.snapshot_language(node)
            node.visit_count += 1
            self.expand(node, state, rng)
            if not node.children:
                node.truncated = True
                truncated = True
                break
            novelty = {
                key: self.curiosity.score(action.preview_state)
                for key, action in node.children.items()
            }
            action_node, scores = select_action(
                node,
                self.config,
                iteration,
                rng,
                novelty,
            )
            result = self.adapter.step(
                state,
                action_node.action,
                rng,
                state_visit_count=max(0, node.visit_count - 1),
            )
            try:
                reward = float(result.reward)
            except (TypeError, ValueError) as error:
                raise ValueError("reward must be finite") from error
            if not math.isfinite(reward):
                raise ValueError("reward must be finite")
            terminated = result.terminated or self.adapter.is_terminal(
                result.state
            )
            truncated = result.truncated
            next_key = self.adapter.state_key(result.state)
            next_observation = copy.deepcopy(result.state.observation)
            existing_outcome = action_node.children.get(next_key)
            outcome_truncated = truncated or (
                existing_outcome is not None and existing_outcome.truncated
            )
            journal.snapshot_action_outcomes(action_node)
            if existing_outcome is not None:
                journal.snapshot_language(existing_outcome)
            next_node = action_node.get_or_create_outcome(
                next_observation,
                next_key,
                terminated,
                outcome_truncated,
            )
            action_path.append(action_node)
            actions.append(action_node.action.key)
            rewards.append(reward)
            executed_observations.append(copy.deepcopy(next_observation))
            score_history.append(scores)
            state = result.state
            node = next_node
            state_path.append(node)
            if terminated or truncated:
                break
        else:
            journal.snapshot_language(node)
            node.truncated = True
            truncated = True

        final_observation = copy.deepcopy(state.observation)
        backup_trajectory(action_path, rewards, self.config)
        return (
            TrajectoryResult(
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                final_observation=final_observation,
                selection_scores=score_history,
                action_path=action_path,
                state_path=state_path,
            ),
            executed_observations,
        )
