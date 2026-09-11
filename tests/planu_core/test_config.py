from dataclasses import FrozenInstanceError

import pytest

from planu_core import PlanUConfig


def test_config_defaults_and_immutability():
    from planu_core import PlanUConfig, SelectionSchedule

    config = PlanUConfig()

    assert config.n_quantiles == 51
    assert config.value_min == -1.0
    assert config.value_max == 1.0
    assert config.discount == 1.0
    assert config.quantile_learning_rate == 0.75
    assert config.curiosity_weight == 0.0
    assert config.include_preview_reward is True
    assert config.categorical_initialization is False
    assert config.categorical_levels == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert config.train_curiosity is True
    assert config.debug_state_keys is False
    assert config.max_depth == 15
    assert config.max_iterations == 1000
    assert config.selection_temperature == 1.0
    assert config.risk_distortion == 0.0
    assert config.selection_schedule == SelectionSchedule()

    with pytest.raises(FrozenInstanceError):
        config.discount = 0.9


def test_categorical_levels_are_defensively_frozen():
    levels = [0.1, 0.5, 0.9]

    config = PlanUConfig(categorical_levels=levels)
    levels[0] = 0.2

    assert config.categorical_levels == (0.1, 0.5, 0.9)


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ((), "nonempty"),
        ((0.1, float("nan")), "finite"),
        ((0.1, 0.1), "strictly increasing"),
        ((0.5, 0.3), "strictly increasing"),
        ((-1.1, 0.5), "within value bounds"),
        ((0.5, 1.1), "within value bounds"),
    ],
)
def test_config_rejects_invalid_categorical_levels(levels, message):
    with pytest.raises(ValueError, match=message):
        PlanUConfig(categorical_levels=levels)


def test_selection_schedule_boundaries():
    from planu_core import SelectionSchedule

    schedule = SelectionSchedule(
        always_sample_before=50,
        probabilistic_sample_before=100,
        sample_probability=0.2,
    )

    assert schedule.always_samples(49) is True
    assert schedule.always_samples(50) is False
    assert schedule.sample_probability_at(49) == 1.0
    assert schedule.sample_probability_at(75) == 0.2
    assert schedule.sample_probability_at(100) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"always_sample_before": -1},
        {
            "always_sample_before": 2,
            "probabilistic_sample_before": 1,
        },
        {"sample_probability": -0.01},
        {"sample_probability": 1.01},
    ],
)
def test_selection_schedule_rejects_invalid_values(kwargs):
    from planu_core import SelectionSchedule

    with pytest.raises(ValueError):
        SelectionSchedule(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_quantiles": 0},
        {"max_depth": 0},
        {"max_iterations": 0},
        {"value_min": 1.0, "value_max": 1.0},
        {"value_min": 2.0, "value_max": 1.0},
        {"quantile_learning_rate": 0.0},
        {"quantile_learning_rate": 1.01},
        {"discount": -0.01},
        {"discount": 1.01},
        {"selection_temperature": 0.0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    from planu_core import PlanUConfig

    with pytest.raises(ValueError):
        PlanUConfig(**kwargs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "curiosity_weight must be finite"),
        (float("inf"), "curiosity_weight must be finite"),
        (float("-inf"), "curiosity_weight must be finite"),
        (-0.01, "curiosity_weight must be nonnegative"),
    ],
)
def test_config_rejects_invalid_curiosity_weight(value, message):
    from planu_core import PlanUConfig

    with pytest.raises(ValueError, match=message):
        PlanUConfig(curiosity_weight=value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "risk_distortion must be finite"),
        (float("inf"), "risk_distortion must be finite"),
        (float("-inf"), "risk_distortion must be finite"),
        (-1.01, "risk_distortion must be between -1 and 1"),
        (1.01, "risk_distortion must be between -1 and 1"),
    ],
)
def test_config_rejects_invalid_risk_distortion(value, message):
    from planu_core import PlanUConfig

    with pytest.raises(ValueError, match=message):
        PlanUConfig(risk_distortion=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"value_min": float("nan")}, "value_min must be finite"),
        ({"value_min": float("inf")}, "value_min must be finite"),
        ({"value_min": float("-inf")}, "value_min must be finite"),
        ({"value_max": float("nan")}, "value_max must be finite"),
        ({"value_max": float("inf")}, "value_max must be finite"),
        ({"value_max": float("-inf")}, "value_max must be finite"),
    ],
)
def test_config_rejects_nonfinite_value_bounds(kwargs, message):
    from planu_core import PlanUConfig

    with pytest.raises(ValueError, match=message):
        PlanUConfig(**kwargs)
