from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from tp2.distances.function import DistanceFunction

from ._shared import (
    assign_points_to_centers,
    compute_radius,
    initialize_min_distances,
    normalize_distance_input,
    update_min_distances,
    validate_k,
)

@dataclass(frozen=True)
class FarthestFirstConfig:
    """Configuration for the farthest-first traversal 2-approximation."""

    random_state: int | None = None


def farthest_first_k_center(
    D: Union[np.ndarray, DistanceFunction], k: int, config: FarthestFirstConfig | None = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the farthest-first 2-approximation for k-center.

    Parameters
    ----------
    D:
        Pre-computed distance matrix of shape (n_samples, n_samples) or a DistanceFunction.
    k:
        Number of centers to select.
    config:
        Optional configuration (for random initialization).

    Returns
    -------
    centers:
        Indices of selected centers (shape: (k,)).
    labels:
        Cluster assignment for each point (shape: (n_samples,)).
    radius:
        Achieved k-center radius (maximum distance to the nearest center).
    """
    if config is None:
        config = FarthestFirstConfig()

    distance = normalize_distance_input(D)
    validate_k(k, distance.n)

    rng = np.random.default_rng(config.random_state)
    first_center = int(rng.integers(0, distance.n))

    centers = [first_center]
    min_dists = initialize_min_distances(distance, first_center)

    while len(centers) < k:
        next_center = int(np.argmax(min_dists))
        centers.append(next_center)
        update_min_distances(distance, min_dists, next_center)

    centers_arr = np.array(centers, dtype=int)
    labels = assign_points_to_centers(distance, centers_arr)
    radius = compute_radius(distance, centers_arr, labels)

    return centers_arr, labels, radius


