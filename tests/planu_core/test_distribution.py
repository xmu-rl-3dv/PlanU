import numpy as np
import pytest

from planu_core.distribution import QuantileDistribution


def test_quantile_distribution_is_exported_from_package():
    from planu_core import QuantileDistribution as ExportedQuantileDistribution

    assert ExportedQuantileDistribution is QuantileDistribution


def test_scalar_initialization_creates_point_mass():
    dist = QuantileDistribution.from_scalar(
        0.4, count=5, value_min=-1.0, value_max=1.0
    )

    np.testing.assert_allclose(dist.fractions, [0.1, 0.3, 0.5, 0.7, 0.9])
    np.testing.assert_allclose(dist.values, [0.4] * 5)


def test_values_initialization_uses_float64_copy_and_midpoint_fractions():
    source = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    dist = QuantileDistribution.from_values(source, value_min=-1.0, value_max=1.0)
    source[0] = 1.0

    assert dist.values.dtype == np.float64
    np.testing.assert_allclose(dist.values, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(dist.fractions, [1 / 6, 1 / 2, 5 / 6])


def test_categorical_initialization_fills_every_quantile():
    dist = QuantileDistribution.from_categorical(
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        probabilities=[0.0, 0.0, 0.0, 0.0, 1.0],
        count=51,
        value_min=-1.0,
        value_max=1.0,
    )

    np.testing.assert_allclose(dist.values, [0.9] * 51)


def test_categorical_initialization_normalizes_probability_mass():
    dist = QuantileDistribution.from_categorical(
        levels=[-1.0, 1.0],
        probabilities=[1.0, 3.0],
        count=4,
        value_min=-1.0,
        value_max=1.0,
    )

    np.testing.assert_allclose(dist.values, [-1.0, 1.0, 1.0, 1.0])


def test_categorical_initialization_normalizes_large_finite_probabilities():
    dist = QuantileDistribution.from_categorical(
        levels=[-1.0, 1.0],
        probabilities=[1e308, 1e308],
        count=2,
        value_min=-1.0,
        value_max=1.0,
    )

    np.testing.assert_allclose(dist.values, [-1.0, 1.0])


def test_risk_neutral_value_is_arithmetic_mean():
    dist = QuantileDistribution.from_values([-1.0, 0.0, 1.0], -1.0, 1.0)

    assert dist.distorted_value() == 0.0


def test_positive_and_negative_distortion_select_upper_and_lower_tails():
    dist = QuantileDistribution.from_values([-1.0, 0.0, 1.0], -1.0, 1.0)

    assert dist.distorted_value(0.5) == 0.5
    assert dist.distorted_value(-0.5) == -0.5


@pytest.mark.parametrize("risk_distortion", [-0.01, 0.01])
def test_distortion_falls_back_to_all_values_when_tail_is_empty(risk_distortion):
    dist = QuantileDistribution.from_values([-1.0, 0.0, 1.0], -1.0, 1.0)

    assert dist.distorted_value(risk_distortion) == 0.0


@pytest.mark.parametrize(
    "risk_distortion",
    [float("-inf"), -1.01, 1.01, float("inf"), float("nan")],
)
def test_distortion_rejects_nonfinite_or_out_of_range_values(risk_distortion):
    dist = QuantileDistribution.from_scalar(0.0, 3, -1.0, 1.0)

    with pytest.raises(ValueError):
        dist.distorted_value(risk_distortion)


def test_scalar_update_matches_reference_formula():
    dist = QuantileDistribution.from_scalar(0.5, count=3, value_min=-1.0, value_max=1.0)

    dist.update_scalar(target=1.0, learning_rate=0.75)

    expected = 0.5 + 0.75 * np.array([1 / 6, 1 / 2, 5 / 6]) * 0.5
    np.testing.assert_allclose(dist.values, expected)


def test_scalar_update_clips_values_to_bounds():
    upper = QuantileDistribution.from_scalar(0.9, count=1, value_min=-1.0, value_max=1.0)
    lower = QuantileDistribution.from_scalar(-0.9, count=1, value_min=-1.0, value_max=1.0)

    upper.update_scalar(target=10.0, learning_rate=1.0)
    lower.update_scalar(target=-10.0, learning_rate=1.0)

    np.testing.assert_allclose(upper.values, [1.0])
    np.testing.assert_allclose(lower.values, [-1.0])


@pytest.mark.parametrize("target", [float("-inf"), float("inf"), float("nan")])
def test_update_rejects_nonfinite_target(target):
    dist = QuantileDistribution.from_scalar(0.0, count=5, value_min=-1.0, value_max=1.0)

    with pytest.raises(ValueError, match="finite"):
        dist.update_scalar(target, learning_rate=0.75)


@pytest.mark.parametrize(
    "learning_rate",
    [float("-inf"), -0.1, 0.0, 1.01, float("inf"), float("nan")],
)
def test_update_rejects_invalid_learning_rate(learning_rate):
    dist = QuantileDistribution.from_scalar(0.0, count=5, value_min=-1.0, value_max=1.0)

    with pytest.raises(ValueError):
        dist.update_scalar(target=1.0, learning_rate=learning_rate)


@pytest.mark.parametrize("count", [0, -1])
def test_initializers_reject_nonpositive_count(count):
    with pytest.raises(ValueError):
        QuantileDistribution.from_scalar(0.0, count, -1.0, 1.0)

    with pytest.raises(ValueError):
        QuantileDistribution.from_categorical(
            [0.0], [1.0], count, value_min=-1.0, value_max=1.0
        )


@pytest.mark.parametrize(
    ("value_min", "value_max"),
    [
        (float("-inf"), 1.0),
        (float("nan"), 1.0),
        (-1.0, float("inf")),
        (-1.0, float("nan")),
        (1.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_distribution_rejects_invalid_bounds(value_min, value_max):
    with pytest.raises(ValueError):
        QuantileDistribution.from_scalar(0.0, 3, value_min, value_max)


@pytest.mark.parametrize(
    ("fractions", "values"),
    [
        ([], []),
        ([0.5], []),
        ([0.25, 0.75], [0.0]),
        ([float("nan")], [0.0]),
        ([0.5], [float("inf")]),
    ],
)
def test_direct_construction_rejects_invalid_arrays(fractions, values):
    with pytest.raises(ValueError):
        QuantileDistribution(
            np.asarray(fractions),
            np.asarray(values),
            value_min=-1.0,
            value_max=1.0,
        )


@pytest.mark.parametrize(
    ("levels", "probabilities"),
    [
        ([], []),
        ([0.0], []),
        ([0.0, 1.0], [1.0]),
        ([0.0], [-1.0]),
        ([0.0], [float("nan")]),
        ([0.0], [float("inf")]),
        ([0.0], [0.0]),
        ([float("nan")], [1.0]),
    ],
)
def test_categorical_initialization_rejects_invalid_inputs(levels, probabilities):
    with pytest.raises(ValueError):
        QuantileDistribution.from_categorical(
            levels,
            probabilities,
            count=3,
            value_min=-1.0,
            value_max=1.0,
        )
