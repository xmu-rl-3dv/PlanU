from typing import Dict, Hashable, Mapping, Optional, Tuple

import numpy as np

from .config import PlanUConfig
from .nodes import ActionNode, LanguageNode


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("softmax values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("softmax values must be finite")

    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("softmax temperature must be finite and positive")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = (values - np.max(values)) / temperature
        exp_values = np.exp(shifted)
        normalization = float(np.sum(exp_values))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("softmax normalization must be finite and nonzero")

    probabilities = exp_values / normalization
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("softmax probabilities must be finite")
    return probabilities


def select_action(
    node: LanguageNode,
    config: PlanUConfig,
    iteration: int,
    rng: np.random.Generator,
    novelty: Optional[Mapping[Hashable, float]] = None,
) -> Tuple[ActionNode, Dict[Hashable, float]]:
    if not node.children:
        raise ValueError("cannot select from an unexpanded state")

    novelty = {} if novelty is None else novelty
    try:
        supplied_novelty = np.asarray(
            [float(value) for value in novelty.values()],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("novelty values must be finite") from error
    if not np.all(np.isfinite(supplied_novelty)):
        raise ValueError("novelty values must be finite")

    child_items = list(node.children.items())
    actions = [action for _, action in child_items]
    raw_novelty = np.asarray(
        [float(novelty.get(key, 0.0)) for key, _ in child_items],
        dtype=np.float64,
    )
    scale = float(np.max(np.abs(raw_novelty)))
    if scale > 0.0:
        scaled_novelty = raw_novelty / scale
        normalized_novelty = scaled_novelty / np.sum(np.abs(scaled_novelty))
    else:
        normalized_novelty = raw_novelty

    base_values = np.asarray(
        [
            action.distribution.distorted_value(config.risk_distortion)
            for action in actions
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(base_values)):
        raise ValueError("base action values must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        final_scores = (
            base_values + config.curiosity_weight * normalized_novelty
        )
    if not np.all(np.isfinite(final_scores)):
        raise ValueError("action scores must be finite")

    scores = {
        action.action.key: float(score)
        for action, score in zip(actions, final_scores)
    }
    sample_probability = config.selection_schedule.sample_probability_at(iteration)
    if sample_probability == 1.0:
        should_sample = True
    elif 0.0 < sample_probability < 1.0:
        should_sample = bool(rng.random() < sample_probability)
    else:
        should_sample = False

    if should_sample:
        probabilities = _stable_softmax(
            final_scores,
            config.selection_temperature,
        )
        selected_index = int(rng.choice(len(actions), p=probabilities))
    else:
        selected_index = int(np.argmax(final_scores))
    return actions[selected_index], scores
