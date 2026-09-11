import math
from numbers import Integral
from typing import Any


class NullCuriosity:
    ready_to_train = False

    def score(self, observation: Any) -> float:
        return 0.0

    def observe(self, observation: Any) -> None:
        return None

    def train(self) -> None:
        return None


class RndCuriosity:
    def __init__(self, model: Any, minimum_samples: int = 15) -> None:
        if isinstance(minimum_samples, bool) or not isinstance(
            minimum_samples,
            Integral,
        ):
            raise TypeError("minimum_samples must be an integer")
        if minimum_samples < 0:
            raise ValueError("minimum_samples must be nonnegative")
        self.model = model
        self.minimum_samples = int(minimum_samples)
        self.sample_count = 0

    @property
    def ready_to_train(self) -> bool:
        return self.sample_count > self.minimum_samples

    def score(self, observation: Any) -> float:
        value = self.model.estimate(observation)
        for method_name in ("detach", "cpu", "item"):
            method = getattr(value, method_name, None)
            if method is not None:
                value = method()
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("curiosity score must be a finite number") from error
        if not math.isfinite(score):
            raise ValueError("curiosity score must be finite")
        return score

    def observe(self, observation: Any) -> None:
        self.model.collect_data(observation)
        self.sample_count += 1

    def train(self) -> Any:
        if self.ready_to_train:
            return self.model.train()
        return None
