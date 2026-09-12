import copy
import math
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from .backup import backup_trajectory
from .config import PlanUConfig
from .curiosity import NullCuriosity
from .distribution import QuantileDistribution
from .interfaces import (
    ActionProvider,
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
    truncation_reason: Optional[str]
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
                Optional[str],
                int,
                Dict[Hashable, ActionNode],
            ],
        ] = {}
        self._action_nodes: Dict[
            int,
            Tuple[
                ActionNode,
                Dict[Hashable, LanguageNode],
                List[
                    Tuple[
                        LanguageNode,
                        int,
                        bool,
                        bool,
                        Optional[str],
                    ]
                ],
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
                node.truncation_reason,
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
                        child.truncation_reason,
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
            truncation_reason,
            outcome_visits,
            children,
        ) in self._language_nodes.values():
            node.visit_count = visit_count
            node.terminated = terminated
            node.truncated = truncated
            node.truncation_reason = truncation_reason
            node.outcome_visits = outcome_visits
            node.children.clear()
            node.children.update(children)

        for node, children, outcome_states in self._action_nodes.values():
            node.children.clear()
            node.children.update(children)
            for (
                child,
                outcome_visits,
                terminated,
                truncated,
                truncation_reason,
            ) in outcome_states:
                child.outcome_visits = outcome_visits
                child.terminated = terminated
                child.truncated = truncated
                child.truncation_reason = truncation_reason


class PlanUSearch:
    def __init__(
        self,
        adapter: EnvironmentAdapter,
        scorer: ActionScorer,
        config: PlanUConfig,
        curiosity: Optional[CuriosityProvider] = None,
        action_provider: Optional[ActionProvider] = None,
    ):
        self.adapter = adapter
        self.scorer = scorer
        self.config = config
        self.curiosity = NullCuriosity() if curiosity is None else curiosity
        self.action_provider = action_provider
        self.root: Optional[LanguageNode] = None

    def _normalize_completion(
        self,
        state: EnvironmentState,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[bool, bool]:
        if terminated:
            return True, False
        if truncated or self.adapter.is_truncated(state):
            return False, True
        return bool(self.adapter.is_terminal(state)), False

    def _state_fingerprint(self, state: EnvironmentState) -> Any:
        adapter_fingerprint = getattr(
            self.adapter,
            "state_fingerprint",
            None,
        )
        try:
            if callable(adapter_fingerprint):
                return adapter_fingerprint(state)
            return pickle.dumps(
                state,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception as error:
            raise ValueError(
                "cannot fingerprint full environment state for state key "
                "collision "
                "detection"
            ) from error

    def _check_state_key_collision(
        self,
        existing: LanguageNode,
        incoming: EnvironmentState,
    ) -> Any:
        if not self.config.debug_state_keys:
            return None
        stored_fingerprint = existing.debug_fingerprint
        if stored_fingerprint is None:
            stored = EnvironmentState(existing.state, None)
            stored_fingerprint = self._state_fingerprint(stored)
        incoming_fingerprint = self._state_fingerprint(incoming)
        try:
            equal = stored_fingerprint == incoming_fingerprint
            if isinstance(equal, np.ndarray):
                equal = bool(np.all(equal))
            else:
                equal = bool(equal)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "cannot compare state fingerprints for collision detection"
            ) from error
        if not equal:
            raise ValueError("state key collision")
        return incoming_fingerprint

    def _ensure_root(self, state: EnvironmentState) -> LanguageNode:
        key = self.adapter.state_key(state)
        if self.root is None:
            fingerprint = (
                self._state_fingerprint(state)
                if self.config.debug_state_keys
                else None
            )
            self.root = LanguageNode(
                copy.deepcopy(state.observation),
                key,
                debug_fingerprint=fingerprint,
            )
        elif self.root.state_key != key:
            raise ValueError("reset state does not match persistent PlanU root")
        else:
            self._check_state_key_collision(self.root, state)
        return self.root

    def expand(
        self,
        node: LanguageNode,
        state: EnvironmentState,
        rng: np.random.Generator,
    ) -> None:
        if node.terminated or node.truncated or node.children:
            return

        if self.action_provider is None:
            candidates = list(self.adapter.actions(state, node.visit_count))
        else:
            # run_iteration registers the current visit before expansion.
            candidates = list(
                self.action_provider.actions(
                    state,
                    max(0, node.visit_count - 1),
                )
            )
        if not candidates:
            return
        action_keys = [candidate.key for candidate in candidates]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("duplicate action key")
        distributions = None
        priors = None
        if self.config.categorical_initialization:
            score_distributions = getattr(
                self.scorer,
                "score_distributions",
                None,
            )
            if not callable(score_distributions):
                raise ValueError(
                    "categorical initialization requires score_distributions"
                )
            try:
                raw_rows = list(
                    score_distributions(
                        state.observation,
                        candidates,
                        self.config.categorical_levels,
                    )
                )
                distributions = np.asarray(raw_rows, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "distribution scores must be finite numbers"
                ) from error
            if len(raw_rows) != len(candidates):
                raise ValueError(
                    "distribution scorer returned the wrong row count"
                )
            expected_shape = (
                len(candidates),
                len(self.config.categorical_levels),
            )
            if distributions.shape != expected_shape:
                raise ValueError(
                    "distribution scorer returned the wrong row shape"
                )
            if not np.all(np.isfinite(distributions)):
                raise ValueError("distribution scores must be finite")
            if np.any(distributions < 0.0):
                raise ValueError("distribution scores must be nonnegative")
            if np.any(np.sum(distributions, axis=1) <= 0.0):
                raise ValueError(
                    "distribution score rows must have nonzero mass"
                )
        else:
            try:
                priors = np.asarray(
                    self.scorer.score(state.observation, candidates),
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "action scores must be finite numbers"
                ) from error
            if priors.shape != (len(candidates),):
                raise ValueError(
                    "scorer returned the wrong number of action scores"
                )
            if not np.all(np.isfinite(priors)):
                raise ValueError("action scores must be finite")
        pending_children: Dict[Hashable, ActionNode] = OrderedDict()
        preview_rng = copy.deepcopy(rng)
        for index, candidate in enumerate(candidates):
            preview = self.adapter.preview(
                state,
                candidate,
                copy.deepcopy(preview_rng),
            )
            preview_terminated, preview_truncated = (
                self._normalize_completion(
                    preview.state,
                    preview.terminated,
                    preview.truncated,
                )
            )
            preview_truncation_reason = (
                preview.info.get("truncation_reason")
                if preview_truncated
                else None
            )
            if preview_truncated and preview_truncation_reason is None:
                preview_truncation_reason = "environment_truncated"
            preview_key = self.adapter.state_key(preview.state)
            if distributions is None:
                initial = float(priors[index])
                if self.config.include_preview_reward:
                    initial += float(preview.reward)
                distribution = QuantileDistribution.from_scalar(
                    initial,
                    self.config.n_quantiles,
                    self.config.value_min,
                    self.config.value_max,
                )
            else:
                distribution = QuantileDistribution.from_categorical(
                    self.config.categorical_levels,
                    distributions[index],
                    self.config.n_quantiles,
                    self.config.value_min,
                    self.config.value_max,
                )
                if self.config.include_preview_reward:
                    try:
                        preview_reward = float(preview.reward)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ValueError(
                            "preview reward must be finite"
                        ) from error
                    if not math.isfinite(preview_reward):
                        raise ValueError("preview reward must be finite")
                    distribution.values += preview_reward
                    np.clip(
                        distribution.values,
                        self.config.value_min,
                        self.config.value_max,
                        out=distribution.values,
                    )
            action = ActionNode(
                parent=node,
                action=candidate,
                distribution=distribution,
                preview_state=copy.deepcopy(preview.state.observation),
            )
            if preview.info.get("record_outcome", True):
                preview_fingerprint = (
                    self._state_fingerprint(preview.state)
                    if self.config.debug_state_keys
                    else None
                )
                action.get_or_create_outcome(
                    copy.deepcopy(preview.state.observation),
                    preview_key,
                    preview_terminated,
                    preview_truncated,
                    increment_visit=False,
                    truncation_reason=preview_truncation_reason,
                    debug_fingerprint=preview_fingerprint,
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
        if self.config.train_curiosity:
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
        root_truncated = bool(self.adapter.is_truncated(state))
        root_terminated = (
            not root_truncated and bool(self.adapter.is_terminal(state))
        )
        if root_terminated or root_truncated:
            journal.snapshot_language(node)
            node.terminated = root_terminated
            node.truncated = root_truncated
            node.truncation_reason = (
                "environment_truncated" if root_truncated else None
            )
            final_observation = copy.deepcopy(state.observation)
            backup_trajectory([], [], self.config)
            return (
                TrajectoryResult(
                    actions=[],
                    rewards=[],
                    terminated=root_terminated,
                    truncated=root_truncated,
                    truncation_reason=node.truncation_reason,
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
        truncation_reason = None

        for _ in range(self.config.max_depth):
            journal.snapshot_language(node)
            node.visit_count += 1
            self.expand(node, state, rng)
            if not node.children:
                node.truncated = True
                truncated = True
                truncation_reason = "no_legal_actions"
                node.truncation_reason = truncation_reason
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
            prepare_step = getattr(self.adapter, "prepare_step", None)
            if callable(prepare_step):
                prepare_step(
                    state,
                    action_node.action,
                    max(0, node.visit_count - 1),
                )
            result = self.adapter.step(
                state,
                action_node.action,
                rng,
            )
            try:
                reward = float(result.reward)
            except (TypeError, ValueError) as error:
                raise ValueError("reward must be finite") from error
            if not math.isfinite(reward):
                raise ValueError("reward must be finite")
            terminated, truncated = self._normalize_completion(
                result.state,
                result.terminated,
                result.truncated,
            )
            truncation_reason = (
                result.info.get("truncation_reason")
                if truncated
                else None
            )
            if truncated and truncation_reason is None:
                truncation_reason = "environment_truncated"
            next_key = self.adapter.state_key(result.state)
            next_observation = copy.deepcopy(result.state.observation)
            existing_outcome = action_node.children.get(next_key)
            next_fingerprint = None
            if existing_outcome is not None:
                next_fingerprint = self._check_state_key_collision(
                    existing_outcome,
                    result.state,
                )
            elif self.config.debug_state_keys:
                next_fingerprint = self._state_fingerprint(result.state)
            outcome_terminated = terminated or (
                existing_outcome is not None and existing_outcome.terminated
            )
            outcome_truncated = truncated or (
                existing_outcome is not None and existing_outcome.truncated
            )
            outcome_truncation_reason = (
                truncation_reason
                if truncated
                else (
                    existing_outcome.truncation_reason
                    if existing_outcome is not None
                    else None
                )
            )
            journal.snapshot_action_outcomes(action_node)
            if existing_outcome is not None:
                journal.snapshot_language(existing_outcome)
                existing_outcome.terminated = outcome_terminated
                existing_outcome.truncated = outcome_truncated
                existing_outcome.truncation_reason = (
                    outcome_truncation_reason
                )
            next_node = action_node.get_or_create_outcome(
                next_observation,
                next_key,
                outcome_terminated,
                outcome_truncated,
                truncation_reason=outcome_truncation_reason,
                debug_fingerprint=next_fingerprint,
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
            truncation_reason = "max_depth"
            node.truncation_reason = truncation_reason

        final_observation = copy.deepcopy(state.observation)
        backup_trajectory(action_path, rewards, self.config)
        return (
            TrajectoryResult(
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                truncation_reason=truncation_reason,
                final_observation=final_observation,
                selection_scores=score_history,
                action_path=action_path,
                state_path=state_path,
            ),
            executed_observations,
        )
