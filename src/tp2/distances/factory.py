from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np

from .function import (
    DistanceFunction,
    MahalanobisDistanceFunction,
    MinkowskiDistanceFunction,
    estimate_distance_matrix_memory,
)
from .mahalanobis import MahalanobisParams, covariance_inverse, pairwise_mahalanobis
from .minkowski import MinkowskiParams, pairwise_minkowski


MetricName = Literal["minkowski", "mahalanobis"]


@dataclass(frozen=True)
class MetricConfig:
    name: MetricName
    params: Dict[str, Any] | None = None


class DistanceFactory:
    """Factory for computing and caching distance matrices.

    The factory is responsible for:
    - instantiating metric parameters;
    - computing pairwise distances using the appropriate implementation;
    - optionally caching distance matrices keyed by (dataset_id, metric_config);
    - estimating memory requirements for distance matrices.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    @staticmethod
    def estimate_memory(n_samples: int, dtype_size: int = 8) -> float:
        """Estimate memory required for a distance matrix in GB."""
        return estimate_distance_matrix_memory(n_samples, dtype_size)

    @staticmethod
    def _metric_key(metric: MetricConfig) -> str:
        return f"{metric.name}:{metric.params!r}"

    def get_distance_matrix(
        self,
        dataset_id: str,
        X: np.ndarray,
        metric: MetricConfig,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Return (and optionally cache) the distance matrix for a dataset/metric."""
        key = (dataset_id, self._metric_key(metric))
        if use_cache and key in self._cache:
            return self._cache[key]

        if metric.name == "minkowski":
            p = 2.0
            if metric.params and "p" in metric.params:
                p = float(metric.params["p"])
            params = MinkowskiParams(p=p)
            D = pairwise_minkowski(X, params=params)
        elif metric.name == "mahalanobis":
            reg = 1e-6
            if metric.params and "regularization" in metric.params:
                reg = float(metric.params["regularization"])
            params = MahalanobisParams(regularization=reg)
            cov_inv = covariance_inverse(X, params=params)
            D = pairwise_mahalanobis(X, cov_inv=cov_inv)
        else:
            raise ValueError(f"Unknown metric: {metric.name!r}")

        if use_cache:
            self._cache[key] = D
        return D

    def get_distance_function(
        self,
        X: np.ndarray,
        metric: MetricConfig,
    ) -> DistanceFunction:
        """Return a distance function for on-the-fly distance computation.
        
        This method creates a DistanceFunction object that computes distances
        on-demand without storing the full distance matrix in memory.
        This is useful for memory-efficient processing of large datasets.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            metric: Metric configuration
            
        Returns:
            DistanceFunction object that can compute distances on-the-fly
        """
        if metric.name == "minkowski":
            p = 2.0
            if metric.params and "p" in metric.params:
                p = float(metric.params["p"])
            return MinkowskiDistanceFunction(X, p=p)
        elif metric.name == "mahalanobis":
            reg = 1e-6
            if metric.params and "regularization" in metric.params:
                reg = float(metric.params["regularization"])
            return MahalanobisDistanceFunction(X, regularization=reg)
        else:
            raise ValueError(f"Unknown metric: {metric.name!r}")


