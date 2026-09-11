import copy

import numpy as np

from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentState,
    TransitionResult,
)


class FakeAdapter:
    def __init__(self):
        self.actions_taken = []
        self.action_calls = 0
        self.clone_calls = 0
        self.preview_calls = 0
        self.reset_seeds = []
        self.reset_state = None

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.reset_state = EnvironmentState(
            observation=np.array([0]),
            runtime={"position": 0},
        )
        return self.reset_state

    def clone(self, state):
        self.clone_calls += 1
        return copy.deepcopy(state)

    def actions(self, state, state_visit_count=0):
        self.action_calls += 1
        if state.runtime["position"] >= 2:
            return []
        return [ActionCandidate("advance", 0, "advance")]

    def preview(self, state, action, rng):
        self.preview_calls += 1
        return self._transition(copy.deepcopy(state))

    def step(self, state, action, rng):
        self.actions_taken.append(action.key)
        return self._transition(state)

    def state_key(self, state):
        return (
            tuple(np.asarray(state.observation).tolist()),
            state.runtime["position"],
        )

    def is_terminal(self, state):
        return state.runtime["position"] == 2

    @staticmethod
    def _transition(state):
        state.runtime["position"] += 1
        state.observation = np.array([state.runtime["position"]])
        return TransitionResult(
            state=state,
            reward=1.0,
            terminated=state.runtime["position"] == 2,
            truncated=False,
        )


class UniformScorer:
    def __init__(self):
        self.calls = 0

    def score(self, observation, actions):
        self.calls += 1
        return np.ones(len(actions), dtype=np.float64) / len(actions)
