from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm.auto import tqdm

from tp2.algorithms import (
    FarthestFirstConfig,
    IntervalRefinementConfig,
    farthest_first_k_center,
    interval_refinement_k_center,
)
from tp2.distances import DistanceFactory, MetricConfig
from tp2.distances.function import DistanceFunction, estimate_distance_matrix_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _iter_datasets(roots: Iterable[Path]) -> Iterable[Tuple[str, Path, Path]]:
    """Yield (dataset_id, features_path, labels_path) over all dataset folders."""
    for root in roots:
        for sub in root.rglob("features.parquet"):
            labels = sub.with_name("labels.parquet")
            if not labels.exists():
                continue
            rel_id = sub.parent.relative_to(root)
            dataset_id = f"{root.name}/{rel_id.as_posix()}"
            yield dataset_id, sub, labels


def _load_dataset(features_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = pd.read_parquet(features_path).to_numpy(dtype=float)
    y = pd.read_parquet(labels_path)["label"].to_numpy()
    return X, y


def _safe_dataset_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "_").replace("\\", "_")


def _append_results(
    output_root: Path, dataset_id: str, rows: List[Dict], label: str
) -> None:
    if not rows:
        return
    written = len(rows)
    df = pd.DataFrame(rows)
    safe_name = _safe_dataset_name(dataset_id)
    output_path = output_root / f"{safe_name}.parquet"
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"  Saved {written} results for {label} to {output_path}")
    rows.clear()


def _save_partial_results(output_root: Path, dataset_id: str, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    safe_name = _safe_dataset_name(dataset_id)
    output_path = output_root / f"{safe_name}_partial.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"  Saved {len(rows)} partial results to {output_path}")
    rows.clear()


def _cluster_labels_from_centers(center_assignments: np.ndarray) -> np.ndarray:
    _, inv = np.unique(center_assignments, return_inverse=True)
    return inv


def _silhouette_from_representation(
    representation: Union[np.ndarray, DistanceFunction],
    metric: MetricConfig,
    dataset_id: str,
    X: np.ndarray,
    cluster_labels: np.ndarray,
    factory: DistanceFactory,
    memory_efficient: bool,
) -> float:
    if memory_efficient and hasattr(representation, "dist_row"):
        if metric.name == "minkowski":
            p_val = metric.params.get("p", 2.0) if metric.params else 2.0
            if p_val == 1.0:
                sil_metric = "manhattan"
            elif p_val == 2.0:
                sil_metric = "euclidean"
            else:
                sil_metric = "euclidean"
                logger.warning(
                    f"  Using euclidean metric for silhouette (p={p_val} not directly supported)"
                )
            return float(silhouette_score(X, cluster_labels, metric=sil_metric))

        if metric.name == "mahalanobis":
            logger.debug("  Computing silhouette score with Mahalanobis (temporary matrix)")
            D_sil = factory.get_distance_matrix(dataset_id, X, metric, use_cache=False)
            try:
                return float(silhouette_score(D_sil, cluster_labels, metric="precomputed"))
            finally:
                del D_sil

        return float("nan")

    return float(silhouette_score(representation, cluster_labels, metric="precomputed"))


def _distance_representation(
    factory: DistanceFactory,
    dataset_id: str,
    X: np.ndarray,
    metric: MetricConfig,
    memory_efficient: bool,
    mem_gb: float,
) -> Union[np.ndarray, DistanceFunction] | None:
    if memory_efficient:
        logger.info(f"  Creating distance function for {metric.name} (memory-efficient mode)...")
        metric_start_time = time.perf_counter()
        try:
            dist_func = factory.get_distance_function(X, metric)
        except Exception as exc:
            logger.error(f"  ✗ Error creating distance function for {metric.name}: {exc}")
            logger.error(traceback.format_exc())
            return None
        metric_time = time.perf_counter() - metric_start_time
        logger.info(
            f"  ✓ Distance function created for {metric.name}: "
            f"shape={dist_func.shape()}, time={metric_time:.2f}s"
        )
        return dist_func

    logger.info(f"  Computing distance matrix for {metric.name}...")
    metric_start_time = time.perf_counter()
    try:
        D = factory.get_distance_matrix(dataset_id, X, metric, use_cache=True)
    except MemoryError as exc:
        logger.error(f"  ✗ Memory error computing distance matrix for {metric.name}: {exc}")
        logger.error(f"    Dataset {dataset_id} requires ~{mem_gb:.2f} GB per matrix")
        return None
    except Exception as exc:
        logger.error(f"  ✗ Error computing distance matrix for {metric.name}: {exc}")
        logger.error(traceback.format_exc())
        return None

    metric_time = time.perf_counter() - metric_start_time
    logger.info(
        f"  ✓ Distance matrix computed for {metric.name}: "
        f"shape={D.shape}, time={metric_time:.2f}s, memory={mem_gb:.2f}GB"
    )
    return D


def _dataset_size_check(n_samples: int, max_samples: int | None) -> Tuple[bool, float]:
    mem_gb = estimate_distance_matrix_memory(n_samples)
    if max_samples is not None and n_samples > max_samples:
        return True, mem_gb
    return False, mem_gb


def _run_kcenter_suite(
    representation: Union[np.ndarray, DistanceFunction],
    metric: MetricConfig,
    dataset_id: str,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    interval_widths: List[float],
    repetitions: int,
    factory: DistanceFactory,
    memory_efficient: bool,
) -> List[Dict]:
    rows: List[Dict] = []
    metric_payload = json.dumps(metric.params or {})

    for rep in tqdm(range(repetitions), desc="Repetitions", leave=False):
        seed = rep
        try:
            ff_config = FarthestFirstConfig(random_state=seed)
            t0 = time.perf_counter()
            centers, labels_centers, radius = farthest_first_k_center(representation, k, ff_config)
            t1 = time.perf_counter()

            cluster_labels = _cluster_labels_from_centers(labels_centers)
            sil = _silhouette_from_representation(
                representation, metric, dataset_id, X, cluster_labels, factory, memory_efficient
            )
            ari = adjusted_rand_score(y, cluster_labels)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "algorithm": "kcenter_farthest_first",
                    "metric_name": metric.name,
                    "metric_params": metric_payload,
                    "interval_width": None,
                    "repetition": rep,
                    "seed": seed,
                    "radius": float(radius),
                    "silhouette": float(sil),
                    "ari": float(ari),
                    "runtime_sec": float(t1 - t0),
                }
            )

            for width in interval_widths:
                try:
                    cfg = IntervalRefinementConfig(width_fraction=width)
                    t0 = time.perf_counter()
                    centers_ir, labels_ir, radius_ir = interval_refinement_k_center(
                        representation, k, cfg
                    )
                    t1 = time.perf_counter()

                    cluster_labels_ir = _cluster_labels_from_centers(labels_ir)
                    sil_ir = _silhouette_from_representation(
                        representation,
                        metric,
                        dataset_id,
                        X,
                        cluster_labels_ir,
                        factory,
                        memory_efficient,
                    )
                    ari_ir = adjusted_rand_score(y, cluster_labels_ir)

                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "algorithm": "kcenter_interval_refinement",
                            "metric_name": metric.name,
                            "metric_params": metric_payload,
                            "interval_width": float(width),
                            "repetition": rep,
                            "seed": seed,
                            "radius": float(radius_ir),
                            "silhouette": float(sil_ir),
                            "ari": float(ari_ir),
                            "runtime_sec": float(t1 - t0),
                        }
                    )
                except Exception as exc:
                    logger.warning(f"  Error in interval-refinement (width={width}, rep={rep}): {exc}")
                    continue

        except Exception as exc:
            logger.warning(f"  Error in farthest-first (rep={rep}): {exc}")
            continue

    return rows


def _run_kmeans_suite(
    dataset_id: str, X: np.ndarray, y: np.ndarray, k: int, repetitions: int
) -> List[Dict]:
    rows: List[Dict] = []

    for rep in tqdm(range(repetitions), desc="KMeans reps", leave=False):
        try:
            seed = rep
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
            t0 = time.perf_counter()
            labels_km = km.fit_predict(X)
            t1 = time.perf_counter()

            sil_km = silhouette_score(X, labels_km, metric="euclidean")
            ari_km = adjusted_rand_score(y, labels_km)

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "algorithm": "kmeans",
                    "metric_name": "euclidean",
                    "metric_params": json.dumps({"p": 2}),
                    "interval_width": None,
                    "repetition": rep,
                    "seed": seed,
                    "radius": np.nan,
                    "silhouette": float(sil_km),
                    "ari": float(ari_km),
                    "runtime_sec": float(t1 - t0),
                }
            )
        except Exception as exc:
            logger.warning(f"  Error in K-Means (rep={rep}): {exc}")
            continue

    return rows

def run_experiments(
    dataset_roots: List[Path],
    output_root: Path,
    repetitions: int = 15,
    max_samples: int | None = 10000,
    test_mode: bool = False,
    verbose: bool = False,
    memory_efficient: bool = False,
) -> None:
    """Run experiments on datasets.
    
    Args:
        dataset_roots: Root directories containing dataset subfolders
        output_root: Directory to save result Parquet files
        repetitions: Number of repetitions per configuration
        max_samples: Maximum number of samples per dataset (None = no limit, 0 = no limit)
        test_mode: If True, process only first 3 datasets
        verbose: If True, enable DEBUG logging
        memory_efficient: If True, use memory-efficient mode (no caching, process all datasets)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_root.mkdir(parents=True, exist_ok=True)
    factory = DistanceFactory()
    
    # Memory-efficient mode: disable filtering and caching
    if memory_efficient:
        logger.info("Memory-efficient mode enabled: processing all datasets without caching")
        max_samples = None

    metric_configs = [
        MetricConfig(name="minkowski", params={"p": 1}),
        MetricConfig(name="minkowski", params={"p": 2}),
        MetricConfig(name="mahalanobis", params={"regularization": 1e-3}),
    ]

    interval_widths = [0.25, 0.18, 0.12, 0.08, 0.04]

    dataset_iter = list(_iter_datasets(dataset_roots))
    logger.info(f"Found {len(dataset_iter)} datasets to process")
    
    if test_mode:
        dataset_iter = dataset_iter[:3]
        logger.info(f"Test mode: Processing only first {len(dataset_iter)} datasets")
    
    if max_samples is not None:
        logger.info(f"Filtering datasets: max_samples={max_samples}")
    
    skipped_count = 0
    processed_count = 0
    
    for dataset_id, feat_path, lab_path in tqdm(
        dataset_iter, desc="Datasets", unit="dataset"
    ):
        pending_rows: List[Dict] = []
        try:
            logger.info(f"Processing dataset: {dataset_id}")
            X, y = _load_dataset(feat_path, lab_path)
            n_samples = X.shape[0]
            k = int(np.unique(y).size)
            logger.info(f"  Dataset shape: {X.shape}, k={k}")

            effective_max = None if memory_efficient else max_samples
            skip_dataset, mem_gb = _dataset_size_check(n_samples, effective_max)
            if skip_dataset:
                logger.warning(
                    f"  SKIPPING {dataset_id}: {n_samples} samples exceeds max_samples={max_samples} "
                    f"(would require ~{mem_gb:.2f} GB for distance matrix)"
                )
                skipped_count += 1
                continue

            logger.info(f"  Estimated memory per distance matrix: ~{mem_gb:.2f} GB")
            if mem_gb > 10:
                if memory_efficient:
                    logger.warning(
                        f"  WARNING: Large distance matrix ({mem_gb:.2f} GB) - "
                        f"memory-efficient mode enabled (no caching, may be slow)"
                    )
                else:
                    logger.warning(
                        f"  WARNING: Large distance matrix ({mem_gb:.2f} GB) - computation may be slow or fail"
                    )

            for metric in tqdm(metric_configs, desc="Metrics", leave=False):
                representation = _distance_representation(
                    factory, dataset_id, X, metric, memory_efficient, mem_gb
                )
                if representation is None:
                    continue

                try:
                    metric_rows = _run_kcenter_suite(
                        representation,
                        metric,
                        dataset_id,
                        X,
                        y,
                        k,
                        interval_widths,
                        repetitions,
                        factory,
                        memory_efficient,
                    )
                except Exception as exc:
                    logger.error(f"  Error processing metric {metric.name} for {dataset_id}: {exc}")
                    logger.error(traceback.format_exc())
                    continue

                pending_rows = metric_rows
                _append_results(output_root, dataset_id, pending_rows, metric.name)

            kmeans_rows = _run_kmeans_suite(dataset_id, X, y, k, repetitions)
            pending_rows = kmeans_rows
            _append_results(output_root, dataset_id, pending_rows, "kmeans")

            safe_name = _safe_dataset_name(dataset_id)
            output_path = output_root / f"{safe_name}.parquet"
            if output_path.exists():
                df_check = pd.read_parquet(output_path)
                logger.info(f"  ✓ Dataset {dataset_id} complete: {len(df_check)} total results saved")
                processed_count += 1
            else:
                logger.warning(f"  ✗ No result file created for {dataset_id}")

            if memory_efficient:
                factory._cache.clear()
                logger.debug(f"  Cleared distance matrix cache after {dataset_id}")

        except Exception as exc:
            logger.error(f"Error processing dataset {dataset_id}: {exc}")
            logger.error(traceback.format_exc())
            _save_partial_results(output_root, dataset_id, pending_rows)
            continue

    logger.info("=" * 60)
    logger.info("Experiment summary:")
    logger.info(f"  Processed: {processed_count} datasets")
    logger.info(f"  Skipped: {skipped_count} datasets (size limit)")
    logger.info(f"  Result files in: {output_root}")
    logger.info("=" * 60)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TP2 k-center experiments.")
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="One or more root folders containing dataset subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where raw result Parquet files will be stored.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=15,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples per dataset (skip larger ones). Use 0 for no limit.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 3 datasets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Memory-efficient mode: process all datasets without caching distance matrices.",
    )

    args = parser.parse_args(argv)
    max_samples = None if args.max_samples == 0 else args.max_samples
    run_experiments(
        args.datasets,
        args.output,
        repetitions=args.repetitions,
        max_samples=max_samples,
        test_mode=args.test,
        verbose=args.verbose,
        memory_efficient=args.memory_efficient,
    )


if __name__ == "__main__":
    main()


