from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from tp2.distances.function import DistanceFunction

from ._shared import (
    DistanceInput,
    assign_points_to_centers,
    max_distance_upper_bound,
    normalize_distance_input,
    validate_k,
)

@dataclass(frozen=True)
class IntervalRefinementConfig:
    """Configuration for the interval-refinement 2-approximation.

    Attributes
    ----------
    width_fraction:
        Target fraction of the initial [lo, hi] interval length. The algorithm
        refines the radius bounds until (hi - lo) <= width_fraction * (hi0 - lo0).
    """

    width_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not (0 < self.width_fraction < 1):
            raise ValueError("width_fraction must be in (0, 1).")


def _decision_greedy_cover(
    distance: DistanceInput, k: int, radius: float
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Greedy decision procedure: can we cover with k balls of radius `radius`?

    Returns
    -------
    feasible:
        True if the procedure uses at most k centers.
    centers:
        Indices of centers chosen.
    labels:
        Assigned center index for each point.
    """
    uncovered = np.ones(distance.n, dtype=bool)
    centers: list[int] = []

    if distance.function is not None:
        within_cache: dict[int, np.ndarray] = {}
    else:
        assert distance.matrix is not None
        within = distance.matrix <= radius

    while np.any(uncovered) and len(centers) < k:
        idx = int(np.argmax(uncovered))
        centers.append(idx)
        if distance.function is not None:
            if idx not in within_cache:
                dists_from_center = distance.function.dist_row(idx)
                within_cache[idx] = dists_from_center <= radius
            covered_by_center = within_cache[idx]
        else:
            covered_by_center = within[idx]
        
        uncovered[covered_by_center] = False

    feasible = not np.any(uncovered)

    centers_arr = np.array(centers, dtype=int)
    if centers_arr.size == 0:
        centers_arr = np.array([0], dtype=int)

    labels = assign_points_to_centers(distance, centers_arr)
    return feasible, centers_arr, labels


def interval_refinement_k_center(
    D: Union[np.ndarray, DistanceFunction], k: int, config: IntervalRefinementConfig
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Interval-refinement 2-approximation for k-center.

    Parameters
    ----------
    D:
        Pre-computed distance matrix of shape (n_samples, n_samples) or a DistanceFunction.
    k:
        Number of centers.
    config:
        Configuration specifying the final interval width fraction.
    """
    distance = normalize_distance_input(D)
    validate_k(k, distance.n)

    lo = 0.0
    hi = max_distance_upper_bound(distance)
    lo0, hi0 = lo, hi

    best_centers = None
    best_labels = None
    best_radius = hi

    while (hi - lo) > config.width_fraction * (hi0 - lo0):
        mid = 0.5 * (lo + hi)
        feasible, centers, labels = _decision_greedy_cover(distance, k, radius=mid)
        if feasible:
            # mid is an upper bound on the optimal radius
            hi = mid
            best_centers = centers
            best_labels = labels
            best_radius = mid
        else:
            # mid is too small -> increase lower bound
            lo = mid

    if best_centers is None or best_labels is None:
        # Fallback: use greedy cover at hi
        _, best_centers, best_labels = _decision_greedy_cover(distance, k, radius=hi)
        best_radius = hi

    return best_centers, best_labels, float(best_radius)


