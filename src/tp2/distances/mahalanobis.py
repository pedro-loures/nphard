from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MahalanobisParams:
    """Parameters for the Mahalanobis distance.

    Attributes
    ----------
    regularization:
        Non-negative ridge term added to the diagonal of the covariance
        matrix before inversion to improve numerical stability.
    """

    regularization: float = 1e-6

    def __post_init__(self) -> None:
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative.")


def covariance_inverse(X: np.ndarray, params: MahalanobisParams | None = None) -> np.ndarray:
    """Estimate the inverse covariance matrix from data."""
    if params is None:
        params = MahalanobisParams()

    X = np.asarray(X, dtype=float)
    # Sample covariance with rows as observations
    cov = np.cov(X, rowvar=False, bias=False)
    cov = np.asarray(cov, dtype=float)

    # Regularization on the diagonal
    if params.regularization > 0:
        cov = cov + params.regularization * np.eye(cov.shape[0], dtype=float)

    return np.linalg.inv(cov)


def pairwise_mahalanobis(
    x: np.ndarray,
    y: np.ndarray | None = None,
    cov_inv: np.ndarray | None = None,
    params: MahalanobisParams | None = None,
) -> np.ndarray:
    """Compute pairwise Mahalanobis distances using an explicit covariance inverse.

    Parameters
    ----------
    x, y:
        Arrays of shape (n_samples, n_features). If `y` is ``None``, distances
        are computed between all pairs of rows in `x`.
    cov_inv:
        Pre-computed inverse covariance matrix of shape (n_features, n_features).
        If ``None``, it is estimated from `x`.
    params:
        Mahalanobis parameters controlling regularization when estimating
        the covariance.
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        y = x
    else:
        y = np.asarray(y, dtype=float)

    if cov_inv is None:
        cov_inv = covariance_inverse(x, params=params)
    else:
        cov_inv = np.asarray(cov_inv, dtype=float)

    # Transform the data by the Cholesky factor of cov_inv:
    # d_M(x, y) = || L (x - y) ||_2 where L^T L = cov_inv
    L = np.linalg.cholesky(cov_inv)
    x_t = x @ L.T
    y_t = y @ L.T

    diff = x_t[:, None, :] - y_t[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


