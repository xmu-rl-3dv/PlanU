from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count must be positive")


@dataclass
class QuantileDistribution:
    fractions: np.ndarray
    values: np.ndarray
    value_min: float
    value_max: float

    def __post_init__(self) -> None:
        self.value_min = float(self.value_min)
        self.value_max = float(self.value_max)
        if not np.isfinite(self.value_min):
            raise ValueError("value_min must be finite")
        if not np.isfinite(self.value_max):
            raise ValueError("value_max must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be less than value_max")

        fractions = np.asarray(self.fractions, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if fractions.ndim != 1 or values.ndim != 1:
            raise ValueError("fractions and values must be one-dimensional")
        if fractions.size == 0 or values.size == 0:
            raise ValueError("fractions and values must be nonempty")
        if fractions.size != values.size:
            raise ValueError("fractions and values must have the same length")
        if not np.all(np.isfinite(fractions)) or not np.all(np.isfinite(values)):
            raise ValueError("fractions and values must be finite")

        self.fractions = fractions.copy()
        self.values = values.copy()

    @classmethod
    def from_scalar(
        cls,
        value: float,
        count: int,
        value_min: float,
        value_max: float,
    ) -> "QuantileDistribution":
        _validate_count(count)
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        values = np.full(count, float(value), dtype=np.float64)
        return cls(fractions, values, value_min, value_max)

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        value_min: float,
        value_max: float,
    ) -> "QuantileDistribution":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("values must be a nonempty one-dimensional array")
        fractions = (np.arange(array.size, dtype=np.float64) + 0.5) / array.size
        return cls(fractions, array.copy(), value_min, value_max)

    @classmethod
    def from_categorical(
        cls,
        levels: Sequence[float],
        probabilities: Sequence[float],
        count: int,
        value_min: float,
        value_max: float,
    ) -> "QuantileDistribution":
        _validate_count(count)
        level_array = np.asarray(levels, dtype=np.float64)
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if level_array.ndim != 1 or probability_array.ndim != 1:
            raise ValueError("levels and probabilities must be one-dimensional")
        if level_array.size == 0 or level_array.size != probability_array.size:
            raise ValueError("levels and probabilities must have the same nonzero length")
        if not np.all(np.isfinite(level_array)):
            raise ValueError("levels must be finite")
        if not np.all(np.isfinite(probability_array)):
            raise ValueError("probabilities must be finite")
        if np.any(probability_array < 0.0):
            raise ValueError("probabilities must be nonnegative")

        probability_scale = float(np.max(probability_array))
        if probability_scale <= 0.0:
            raise ValueError("probabilities must have nonzero mass")
        scaled_probabilities = probability_array / probability_scale
        normalized = scaled_probabilities / np.sum(scaled_probabilities)
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        cumulative_probabilities = np.cumsum(normalized)
        cumulative_probabilities[-1] = 1.0
        indices = np.searchsorted(cumulative_probabilities, fractions, side="left")
        values = level_array[np.minimum(indices, level_array.size - 1)]
        return cls(fractions, values, value_min, value_max)

    def distorted_value(self, risk_distortion: float = 0.0) -> float:
        risk_distortion = float(risk_distortion)
        if not np.isfinite(risk_distortion):
            raise ValueError("risk_distortion must be finite")
        if not -1.0 <= risk_distortion <= 1.0:
            raise ValueError("risk_distortion must be between -1 and 1")
        if risk_distortion == 0.0:
            return float(np.mean(self.values))

        if risk_distortion > 0.0:
            mask = self.fractions >= 1.0 - risk_distortion
        else:
            mask = self.fractions <= -risk_distortion
        selected = self.values[mask]
        if selected.size == 0:
            selected = self.values
        return float(np.mean(selected))

    def update_scalar(self, target: float, learning_rate: float) -> None:
        target = float(target)
        learning_rate = float(learning_rate)
        if not np.isfinite(target):
            raise ValueError("quantile target must be finite")
        if not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")

        delta = target - self.values
        weights = np.where(delta > 0.0, self.fractions, self.fractions - 1.0)
        self.values += learning_rate * weights * np.abs(delta)
        np.clip(self.values, self.value_min, self.value_max, out=self.values)
