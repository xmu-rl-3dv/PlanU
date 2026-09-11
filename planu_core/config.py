from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionSchedule:
    always_sample_before: int = 0
    probabilistic_sample_before: int = 0
    sample_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.always_sample_before < 0:
            raise ValueError("always_sample_before must be nonnegative")
        if self.probabilistic_sample_before < self.always_sample_before:
            raise ValueError(
                "probabilistic_sample_before cannot precede always_sample_before"
            )
        if not 0.0 <= self.sample_probability <= 1.0:
            raise ValueError("sample_probability must be between 0 and 1")

    def always_samples(self, iteration: int) -> bool:
        return iteration < self.always_sample_before

    def sample_probability_at(self, iteration: int) -> float:
        if iteration < self.probabilistic_sample_before:
            return self.sample_probability
        return 0.0


@dataclass(frozen=True)
class PlanUConfig:
    n_quantiles: int = 51
    value_min: float = -1.0
    value_max: float = 1.0
    quantile_learning_rate: float = 0.75
    discount: float = 1.0
    curiosity_weight: float = 0.0
    include_preview_reward: bool = True
    max_depth: int = 15
    max_iterations: int = 1000
    selection_temperature: float = 1.0
    risk_distortion: float = 0.0
    selection_schedule: SelectionSchedule = field(default_factory=SelectionSchedule)

    def __post_init__(self) -> None:
        if self.n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not self.value_min < self.value_max:
            raise ValueError("value_min must be less than value_max")
        if not 0.0 < self.quantile_learning_rate <= 1.0:
            raise ValueError("quantile_learning_rate must be between 0 and 1")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1")
        if not self.selection_temperature > 0.0:
            raise ValueError("selection_temperature must be positive")
