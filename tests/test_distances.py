from __future__ import annotations

import numpy as np

from tp2.distances.mahalanobis import MahalanobisParams, covariance_inverse, pairwise_mahalanobis
from tp2.distances.minkowski import MinkowskiParams, pairwise_minkowski
from tp2.distances.function import MahalanobisDistanceFunction, MinkowskiDistanceFunction


def test_pairwise_minkowski_matches_manual_norm() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=float)
    params = MinkowskiParams(p=2)
    D = pairwise_minkowski(X, params=params)

    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, np.sqrt(5.0)],
            [2.0, np.sqrt(5.0), 0.0],
        ]
    )
    assert np.allclose(D, expected)

    dist_func = MinkowskiDistanceFunction(X, p=2)
    assert np.allclose(dist_func.dist_row(0), expected[0])
    assert np.isclose(dist_func.max_distance(), np.max(expected))


def test_mahalanobis_distance_function_matches_pairwise() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    params = MahalanobisParams(regularization=1e-3)
    cov_inv = covariance_inverse(X, params=params)
    D = pairwise_mahalanobis(X, cov_inv=cov_inv)

    dist_func = MahalanobisDistanceFunction(X, cov_inv=cov_inv)
    # Each row from the function must match the explicit matrix.
    for idx in range(X.shape[0]):
        assert np.allclose(dist_func.dist_row(idx), D[idx])

    assert np.isclose(dist_func.max_distance(), float(np.max(D)))

