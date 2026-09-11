import importlib
import math

import numpy as np
import pytest

from planu_core.interfaces import ActionCandidate


def test_shared_scorer_module_imports_without_optional_ml_dependencies():
    module = importlib.import_module("planu_core.scorers")

    assert hasattr(module, "HuggingFaceActionScorer")
    assert hasattr(module, "ConstantActionScorer")
    assert hasattr(module, "normalize_action_scores")


def test_constant_scorer_returns_one_per_action_by_default():
    from planu_core.scorers import ConstantActionScorer

    candidates = [
        ActionCandidate(3, 3, "first"),
        ActionCandidate(7, 7, "second"),
    ]

    assert ConstantActionScorer().score(object(), candidates) == [1.0, 1.0]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_constant_scorer_rejects_nonfinite_value(value):
    from planu_core.scorers import ConstantActionScorer

    with pytest.raises(ValueError, match="finite"):
        ConstantActionScorer(value)


def test_constant_scorer_preserves_finite_configured_value():
    from planu_core.scorers import ConstantActionScorer

    candidates = [ActionCandidate(0, 0, "only")]

    assert ConstantActionScorer(-0.25).score(None, candidates) == [-0.25]


def test_overcooked_scorer_is_compatibility_alias():
    overcooked = importlib.import_module("mcts.overcooked.PlanU_mcts")
    shared = importlib.import_module("planu_core.scorers")

    assert shared.OvercookedActionScorer is shared.HuggingFaceActionScorer
    assert overcooked.OvercookedActionScorer is shared.HuggingFaceActionScorer
    assert overcooked.normalize_action_scores is shared.normalize_action_scores
    assert overcooked._device_map_for is shared._device_map_for


def test_shared_normalization_matches_reference_math():
    from planu_core.scorers import normalize_action_scores

    actual = normalize_action_scores(
        log_likelihoods=[-4.0, -3.0],
        token_lengths=[2, 1],
        actions=["two words", "one"],
        normalization_mode="token",
        temperature=1.0,
    )

    expected_logits = np.array([-2.0, -3.0])
    expected = np.exp(expected_logits - expected_logits.max())
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected)
