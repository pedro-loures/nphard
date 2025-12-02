from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


PType = float | int


@dataclass(frozen=True)
class MinkowskiParams:
    """Parameters for the Minkowski distance.

    Attributes
    ----------
    p:
        Norm order (p >= 1). Common choices:
        - p=1: Manhattan
        - p=2: Euclidean
    """

    p: PType = 2.0

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("Minkowski parameter p must satisfy p >= 1.")


def pairwise_minkowski(
    x: np.ndarray, y: np.ndarray | None = None, params: MinkowskiParams | None = None
) -> np.ndarray:
    """Compute the pairwise Minkowski distance matrix between rows of `x` and `y`.

    The implementation is fully vectorized using NumPy primitives and does not
    rely on pre-built distance helpers.

    Parameters
    ----------
    x:
        Array of shape (n_samples_x, n_features).
    y:
        Optional array of shape (n_samples_y, n_features). If ``None``,
        distances are computed between all pairs of rows in `x`.
    params:
        MinkowskiParams instance controlling the value of ``p``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples_x, n_samples_y).
    """
    if params is None:
        params = MinkowskiParams()

    x = np.asarray(x, dtype=float)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=float)

    # Broadcasting to shape (n_samples_x, n_samples_y, n_features)
    diff = x[:, None, :] - y[None, :, :]

    if params.p == 1:
        # Manhattan distance
        return np.sum(np.abs(diff), axis=-1)
    if params.p == 2:
        # Euclidean distance
        return np.sqrt(np.sum(diff * diff, axis=-1))

    # General Minkowski distance
    abs_p = np.abs(diff) ** params.p
    return np.sum(abs_p, axis=-1) ** (1.0 / params.p)


def minkowski_radius_from_centers(
    X: np.ndarray,
    center_indices: np.ndarray,
    params: MinkowskiParams | None = None,
) -> float:
    """Compute the k-center radius (maximum distance to nearest center).

    Parameters
    ----------
    X:
        Data array of shape (n_samples, n_features).
    center_indices:
        Indices of the selected centers (1D array).
    params:
        Minkowski parameters.
    """
    if params is None:
        params = MinkowskiParams()

    centers = X[center_indices]
    dists = pairwise_minkowski(X, centers, params=params)
    closest = np.min(dists, axis=1)
    return float(np.max(closest))


MetricName = Literal["minkowski"]


