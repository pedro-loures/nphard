from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


def estimate_distance_matrix_memory(n_samples: int, dtype_size: int = 8) -> float:
    """Return the memory footprint (in GB) of an n×n distance matrix."""
    return (n_samples * n_samples * dtype_size) / (1024 ** 3)


def _max_distance_over_dataset(
    n_samples: int,
    dist_fn: Callable[[int, int], float],
    row_fn: Callable[[int], np.ndarray],
) -> float:
    """Shared helper to compute (or approximate) the max inter-point distance."""
    if n_samples > 10_000:
        sample_pairs = min(10_000, n_samples * 10)
        rng = np.random.default_rng()
        idx_i = rng.integers(0, n_samples, size=sample_pairs)
        idx_j = rng.integers(0, n_samples, size=sample_pairs)
        max_dist = 0.0
        for i_idx, j_idx in zip(idx_i, idx_j):
            max_dist = max(max_dist, float(dist_fn(int(i_idx), int(j_idx))))
        return max_dist

    max_dist = 0.0
    for i in range(n_samples):
        row_dists = row_fn(i)
        max_dist = max(max_dist, float(np.max(row_dists)))
    return max_dist


class DistanceFunction(Protocol):
    """Protocol for on-the-fly distance computation.
    
    This interface allows algorithms to compute distances without
    storing the full distance matrix in memory.
    """

    def dist(self, i: int, j: int) -> float:
        """Compute distance from point i to point j.
        
        Args:
            i: Index of first point
            j: Index of second point
            
        Returns:
            Distance between points i and j
        """
        ...

    def dist_row(self, i: int) -> np.ndarray:
        """Compute distances from point i to all points.
        
        Args:
            i: Index of source point
            
        Returns:
            Array of shape (n_samples,) with distances from point i to all points
        """
        ...

    def dist_col(self, j: int) -> np.ndarray:
        """Compute distances from all points to point j.
        
        Args:
            j: Index of target point
            
        Returns:
            Array of shape (n_samples,) with distances from all points to point j
        """
        ...

    def max_distance(self) -> float:
        """Compute the maximum distance between any two points.
        
        Returns:
            Maximum distance in the dataset
        """
        ...

    def shape(self) -> tuple[int, int]:
        """Return the shape of the distance matrix (n_samples, n_samples).
        
        Returns:
            Tuple (n_samples, n_samples)
        """
        ...


class MinkowskiDistanceFunction:
    """On-the-fly Minkowski distance computation."""

    def __init__(self, X: np.ndarray, p: float = 2.0):
        """Initialize Minkowski distance function.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            p: Minkowski parameter (p >= 1)
        """
        self.X = np.asarray(X, dtype=float)
        self.p = float(p)
        self.n = self.X.shape[0]
        if self.p < 1:
            raise ValueError("Minkowski parameter p must satisfy p >= 1.")

    def dist(self, i: int, j: int) -> float:
        """Compute Minkowski distance from point i to point j."""
        diff = self.X[i] - self.X[j]
        if self.p == 1:
            return float(np.sum(np.abs(diff)))
        if self.p == 2:
            return float(np.sqrt(np.sum(diff * diff)))
        abs_p = np.abs(diff) ** self.p
        return float(np.sum(abs_p) ** (1.0 / self.p))

    def dist_row(self, i: int) -> np.ndarray:
        """Compute distances from point i to all points."""
        x_i = self.X[i]
        diff = self.X - x_i  # Shape: (n_samples, n_features)
        
        if self.p == 1:
            return np.sum(np.abs(diff), axis=1)
        if self.p == 2:
            return np.sqrt(np.sum(diff * diff, axis=1))
        
        abs_p = np.abs(diff) ** self.p
        return np.sum(abs_p, axis=1) ** (1.0 / self.p)

    def dist_col(self, j: int) -> np.ndarray:
        """Compute distances from all points to point j (same as dist_row for symmetric metrics)."""
        return self.dist_row(j)

    def max_distance(self) -> float:
        """Compute maximum distance by sampling pairs (memory-efficient approximation)."""
        return _max_distance_over_dataset(self.n, self.dist, self.dist_row)

    def shape(self) -> tuple[int, int]:
        """Return shape of distance matrix."""
        return (self.n, self.n)


class MahalanobisDistanceFunction:
    """On-the-fly Mahalanobis distance computation."""

    def __init__(self, X: np.ndarray, cov_inv: np.ndarray | None = None, regularization: float = 1e-6):
        """Initialize Mahalanobis distance function.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            cov_inv: Pre-computed inverse covariance matrix (if None, computed from X)
            regularization: Regularization term for covariance estimation
        """
        self.X = np.asarray(X, dtype=float)
        self.n = self.X.shape[0]
        
        if cov_inv is None:
            from .mahalanobis import covariance_inverse, MahalanobisParams
            params = MahalanobisParams(regularization=regularization)
            cov_inv = covariance_inverse(X, params=params)
        
        self.cov_inv = np.asarray(cov_inv, dtype=float)
        # Precompute Cholesky factor for efficiency
        self.L = np.linalg.cholesky(self.cov_inv)
        # Pre-transform all data points
        self.X_transformed = self.X @ self.L.T

    def dist(self, i: int, j: int) -> float:
        """Compute Mahalanobis distance from point i to point j."""
        diff = self.X_transformed[i] - self.X_transformed[j]
        return float(np.sqrt(np.sum(diff * diff)))

    def dist_row(self, i: int) -> np.ndarray:
        """Compute distances from point i to all points."""
        x_i_transformed = self.X_transformed[i]
        diff = self.X_transformed - x_i_transformed  # Shape: (n_samples, n_features)
        return np.sqrt(np.sum(diff * diff, axis=1))

    def dist_col(self, j: int) -> np.ndarray:
        """Compute distances from all points to point j."""
        return self.dist_row(j)

    def max_distance(self) -> float:
        """Compute maximum distance by sampling pairs."""
        return _max_distance_over_dataset(self.n, self.dist, self.dist_row)

    def shape(self) -> tuple[int, int]:
        """Return shape of distance matrix."""
        return (self.n, self.n)


class MatrixDistanceFunction:
    """Wrapper to make a precomputed distance matrix look like a DistanceFunction.
    
    This allows algorithms to work with either matrices or functions.
    """

    def __init__(self, D: np.ndarray):
        """Initialize with precomputed distance matrix.
        
        Args:
            D: Distance matrix of shape (n_samples, n_samples)
        """
        self.D = np.asarray(D, dtype=float)
        if self.D.shape[0] != self.D.shape[1]:
            raise ValueError("Distance matrix must be square.")
        self.n = self.D.shape[0]

    def dist(self, i: int, j: int) -> float:
        """Get distance from matrix."""
        return float(self.D[i, j])

    def dist_row(self, i: int) -> np.ndarray:
        """Get row from matrix."""
        return self.D[i].copy()

    def dist_col(self, j: int) -> np.ndarray:
        """Get column from matrix."""
        return self.D[:, j].copy()

    def max_distance(self) -> float:
        """Get maximum from matrix."""
        return float(np.max(self.D))

    def shape(self) -> tuple[int, int]:
        """Return shape of matrix."""
        return self.D.shape

