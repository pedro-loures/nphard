from __future__ import annotations

import numpy as np

from tp2.algorithms import (
    FarthestFirstConfig,
    IntervalRefinementConfig,
    farthest_first_k_center,
    interval_refinement_k_center,
)
from tp2.distances.minkowski import MinkowskiParams, pairwise_minkowski
from tp2.distances.function import MinkowskiDistanceFunction


def _line_points(n: int = 4) -> np.ndarray:
    return np.arange(n, dtype=float).reshape(-1, 1)


def test_farthest_first_radius_with_matrix() -> None:
    X = _line_points()
    D = pairwise_minkowski(X, params=MinkowskiParams(p=2))
    centers, labels, radius = farthest_first_k_center(D, k=2, config=FarthestFirstConfig(random_state=0))

    assert sorted(centers.tolist()) == [0, 3]
    assert set(labels.tolist()) == {0, 3}
    assert np.isclose(radius, 1.0)


def test_interval_refinement_matches_farthest_first_radius() -> None:
    X = _line_points()
    distance_func = MinkowskiDistanceFunction(X, p=2)
    cfg = IntervalRefinementConfig(width_fraction=0.05)

    centers, labels, radius = interval_refinement_k_center(distance_func, k=2, config=cfg)
    assert len(centers) == 2
    assert set(labels.tolist()) == set(centers.tolist())
    assert radius <= 1.05  # should approximate the 2-approx radius bound

