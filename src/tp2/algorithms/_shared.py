from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from tp2.distances.function import DistanceFunction

DistanceLike = Union[np.ndarray, DistanceFunction]


@dataclass(frozen=True)
class DistanceInput:
    """Unified view over either a distance matrix or a DistanceFunction."""

    n: int
    matrix: np.ndarray | None
    function: DistanceFunction | None

    def row(self, idx: int) -> np.ndarray:
        """Return the row of distances from point idx to all points."""
        if self.function is not None:
            return self.function.dist_row(idx)
        assert self.matrix is not None
        return self.matrix[idx]


def normalize_distance_input(D: DistanceLike) -> DistanceInput:
    """Wrap a distance matrix or DistanceFunction into a DistanceInput."""
    if hasattr(D, "dist_row") and hasattr(D, "dist"):
        func = D  # type: ignore[assignment]
        n = func.shape()[0]
        return DistanceInput(n=n, matrix=None, function=func)

    matrix = np.asarray(D, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    return DistanceInput(n=matrix.shape[0], matrix=matrix, function=None)


def validate_k(k: int, n: int) -> None:
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n.")


def initialize_min_distances(distance: DistanceInput, center: int) -> np.ndarray:
    """Return the initial min-distance vector for a chosen center."""
    return distance.row(center).copy()


def update_min_distances(distance: DistanceInput, mins: np.ndarray, center: int) -> None:
    """Tighten the running min-distance vector with a new center."""
    np.minimum(mins, distance.row(center), out=mins)


def assign_points_to_centers(distance: DistanceInput, centers: np.ndarray) -> np.ndarray:
    """Assign each point to its nearest center (returns actual center indices)."""
    if centers.size == 0:
        raise ValueError("At least one center is required.")

    if distance.function is not None:
        labels = np.zeros(distance.n, dtype=int)
        for i in range(distance.n):
            dists = np.array([distance.function.dist(i, c) for c in centers], dtype=float)
            labels[i] = centers[int(np.argmin(dists))]
        return labels

    assert distance.matrix is not None
    nearest_idx = np.argmin(distance.matrix[:, centers], axis=1)
    return centers[nearest_idx]


def compute_radius(
    distance: DistanceInput, centers: np.ndarray, labels: np.ndarray | None = None
) -> float:
    """Compute the max distance from any point to its assigned center."""
    if labels is None:
        labels = assign_points_to_centers(distance, centers)

    if distance.function is not None:
        radius = 0.0
        for i in range(distance.n):
            radius = max(radius, float(distance.function.dist(i, int(labels[i]))))
        return radius

    assert distance.matrix is not None
    row_idx = np.arange(distance.n)
    return float(np.max(distance.matrix[row_idx, labels]))


def max_distance_upper_bound(distance: DistanceInput) -> float:
    """Return a conservative upper bound for the k-center radius."""
    if distance.function is not None:
        return float(distance.function.max_distance())
    assert distance.matrix is not None
    return float(np.max(distance.matrix))


