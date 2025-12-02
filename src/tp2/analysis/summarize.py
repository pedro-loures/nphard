from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean_or_none(series: pd.Series) -> float | None:
    clean = series.dropna()
    return None if clean.empty else float(clean.mean())


def _format_mean_std(mean_val: float | None, std_val: float | None, precision: int = 3) -> str:
    if mean_val is None or std_val is None:
        return "N/A"
    fmt = f"{{:.{precision}f}} ± {{:.{precision}f}}"
    return fmt.format(mean_val, std_val)


def _load_raw(raw_root: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    parquet_files = list(raw_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under {raw_root}. "
            f"Make sure experiments completed successfully and generated result files."
        )
    print(f"Loading {len(parquet_files)} result files from {raw_root}")
    for p in parquet_files:
        try:
            parts.append(pd.read_parquet(p))
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
            continue
    if not parts:
        raise FileNotFoundError(f"Could not load any Parquet files from {raw_root}")
    return pd.concat(parts, ignore_index=True)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_id", "algorithm", "metric_name", "metric_params", "interval_width"]
    metrics = ["radius", "silhouette", "ari", "runtime_sec"]

    agg = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"])
    # Flatten MultiIndex columns
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def _create_algorithm_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create table comparing algorithms aggregated over all datasets."""
    # Map algorithm names to readable format
    algorithm_map = {
        "kcenter_farthest_first": "Farthest-First",
        "kcenter_interval_refinement": "Interval-Refinement",
        "kmeans": "K-Means"
    }
    
    # Aggregate: average over datasets, metrics, and interval widths
    agg_data = []
    for alg in sorted(summary["algorithm"].unique()):
        alg_data = summary[summary["algorithm"] == alg]
        
        row = {
            "Algorithm": algorithm_map.get(alg, alg),
            "Radius": _format_mean_std(
                _mean_or_none(alg_data["radius_mean"]),
                _mean_or_none(alg_data["radius_std"]),
            ),
            "Silhouette": _format_mean_std(
                float(alg_data["silhouette_mean"].mean()), float(alg_data["silhouette_std"].mean())
            ),
            "ARI": _format_mean_std(
                float(alg_data["ari_mean"].mean()), float(alg_data["ari_std"].mean())
            ),
            "Runtime (s)": _format_mean_std(
                float(alg_data["runtime_sec_mean"].mean()),
                float(alg_data["runtime_sec_std"].mean()),
                precision=4,
            ),
        }
        agg_data.append(row)
    
    return pd.DataFrame(agg_data)


def _create_metric_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create table comparing distance metrics aggregated over all datasets."""
    metric_rows = []
    
    # Process minkowski metrics (p=1 and p=2)
    for p_val in [1, 2]:
        p_data = summary[
            (summary["metric_name"] == "minkowski") & 
            (summary["metric_params"].str.contains(f'"p": {p_val}', na=False))
        ]
        if len(p_data) > 0:
            label = "Manhattan" if p_val == 1 else "Euclidean"
            metric_rows.append({
                "Metric": label,
                "Radius": _format_mean_std(
                    _mean_or_none(p_data["radius_mean"]),
                    _mean_or_none(p_data["radius_std"]),
                ),
                "Silhouette": _format_mean_std(
                    float(p_data["silhouette_mean"].mean()),
                    float(p_data["silhouette_std"].mean()),
                ),
                "ARI": _format_mean_std(
                    float(p_data["ari_mean"].mean()),
                    float(p_data["ari_std"].mean()),
                ),
                "Runtime (s)": _format_mean_std(
                    float(p_data["runtime_sec_mean"].mean()),
                    float(p_data["runtime_sec_std"].mean()),
                    precision=4,
                ),
            })
    
    # Process mahalanobis
    mahal_data = summary[summary["metric_name"] == "mahalanobis"]
    if len(mahal_data) > 0:
        metric_rows.append({
            "Metric": "Mahalanobis",
            "Radius": _format_mean_std(
                _mean_or_none(mahal_data["radius_mean"]),
                _mean_or_none(mahal_data["radius_std"]),
            ),
            "Silhouette": _format_mean_std(
                float(mahal_data["silhouette_mean"].mean()),
                float(mahal_data["silhouette_std"].mean()),
            ),
            "ARI": _format_mean_std(
                float(mahal_data["ari_mean"].mean()),
                float(mahal_data["ari_std"].mean()),
            ),
            "Runtime (s)": _format_mean_std(
                float(mahal_data["runtime_sec_mean"].mean()),
                float(mahal_data["runtime_sec_std"].mean()),
                precision=4,
            ),
        })
    
    return pd.DataFrame(metric_rows)


def _create_interval_width_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create table showing interval width analysis for interval-refinement algorithm."""
    # Filter to interval_refinement only
    interval_data = summary[summary["algorithm"] == "kcenter_interval_refinement"].copy()
    
    if len(interval_data) == 0:
        return pd.DataFrame()
    
    # Group by interval width
    width_rows = []
    for width in sorted(interval_data["interval_width"].dropna().unique()):
        width_data = interval_data[interval_data["interval_width"] == width]
        width_pct = int(width * 100)
        width_rows.append({
            "Width": f"{width_pct}%",
            "Radius": _format_mean_std(
                float(width_data["radius_mean"].mean()), float(width_data["radius_std"].mean())
            ),
            "Silhouette": _format_mean_std(
                float(width_data["silhouette_mean"].mean()),
                float(width_data["silhouette_std"].mean()),
            ),
            "ARI": _format_mean_std(
                float(width_data["ari_mean"].mean()), float(width_data["ari_std"].mean())
            ),
            "Runtime (s)": _format_mean_std(
                float(width_data["runtime_sec_mean"].mean()),
                float(width_data["runtime_sec_std"].mean()),
                precision=4,
            ),
        })
    
    return pd.DataFrame(width_rows)


def _save_table_artifacts(summary: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # Save full detailed summary (for reference)
    summary.to_parquet(output_root / "summary.parquet", index=False)
    summary.to_csv(output_root / "summary.csv", index=False)

    # Create concise aggregated tables
    algorithm_table = _create_algorithm_comparison_table(summary)
    metric_table = _create_metric_comparison_table(summary)
    interval_width_table = _create_interval_width_table(summary)

    # Save LaTeX tables with proper math notation for ± symbol
    latex_alg = algorithm_table.to_latex(index=False, escape=False, float_format=None)
    latex_alg = latex_alg.replace(" ± ", " $\\pm$ ")
    (output_root / "table_algorithm_comparison.tex").write_text(latex_alg, encoding="utf-8")

    latex_metric = metric_table.to_latex(index=False, escape=False, float_format=None)
    latex_metric = latex_metric.replace(" ± ", " $\\pm$ ")
    (output_root / "table_metric_comparison.tex").write_text(latex_metric, encoding="utf-8")

    latex_width = interval_width_table.to_latex(index=False, escape=False, float_format=None)
    latex_width = latex_width.replace(" ± ", " $\\pm$ ")
    (output_root / "table_interval_width.tex").write_text(latex_width, encoding="utf-8")

    # Save CSV versions for reference
    algorithm_table.to_csv(output_root / "table_algorithm_comparison.csv", index=False)
    metric_table.to_csv(output_root / "table_metric_comparison.csv", index=False)
    interval_width_table.to_csv(output_root / "table_interval_width.csv", index=False)

    # Human-readable description
    txt_lines = [
        "Concise summary tables for TP2 k-center study.\n",
        "Tables:\n",
        "- table_algorithm_comparison: Comparison of algorithms aggregated over all datasets\n",
        "- table_metric_comparison: Comparison of distance metrics aggregated over all datasets\n",
        "- table_interval_width: Analysis of interval width effects for interval-refinement algorithm\n",
    ]
    (output_root / "summary.txt").write_text("".join(txt_lines), encoding="utf-8")

    # JSON metadata
    meta: Dict = {
        "tables": [
            "table_algorithm_comparison",
            "table_metric_comparison",
            "table_interval_width",
        ],
        "description": "Concise aggregated experimental results for TP2 k-center study.",
    }
    (output_root / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _plot_and_describe(
    summary: pd.DataFrame,
    output_root: Path,
    metric: str,
    ylabel: str,
) -> None:
    """Create bar plot per algorithm for a given metric and save sidecar text/JSON."""
    # Aggregate further over datasets for global comparison
    plot_df = (
        summary.groupby(["algorithm", "metric_name"], dropna=False)[f"{metric}_mean"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    algorithms = plot_df["algorithm"].unique()
    x = np.arange(len(algorithms))
    width = 0.25

    # Separate by metric_name (minkowski/mahalanobis/euclidean)
    metric_names = plot_df["metric_name"].unique()
    for i, mname in enumerate(metric_names):
        sub = plot_df[plot_df["metric_name"] == mname]
        heights = [sub[sub["algorithm"] == alg][f"{metric}_mean"].mean() for alg in algorithms]
        ax.bar(x + i * width, heights, width=width, label=mname)

    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.capitalize()} by algorithm and metric")
    ax.legend()
    fig.tight_layout()

    fname = f"{metric}_by_algorithm"
    img_path = output_root / f"{fname}.png"
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    # Text description
    description = (
        f"Bar chart of {metric} (averaged over datasets) comparing algorithms "
        f"for each distance metric. Higher bars indicate better performance "
        f"for {metric} (except runtime, where lower is better)."
    )
    (output_root / f"{fname}.txt").write_text(description, encoding="utf-8")

    # JSON metadata
    meta = {
        "figure": img_path.name,
        "metric": metric,
        "ylabel": ylabel,
        "group_by": ["algorithm", "metric_name"],
        "description": description,
    }
    (output_root / f"{fname}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate TP2 k-center experiment results.")
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Directory containing raw Parquet logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for summary tables and plots.",
    )

    args = parser.parse_args(argv)

    raw_df = _load_raw(args.raw)
    summary = _aggregate(raw_df)
    _save_table_artifacts(summary, args.output)

    # Plots with sidecar text/JSON
    _plot_and_describe(summary, args.output, metric="silhouette", ylabel="Silhouette score")
    _plot_and_describe(summary, args.output, metric="ari", ylabel="Adjusted Rand index")
    _plot_and_describe(summary, args.output, metric="runtime_sec", ylabel="Runtime (s)")


if __name__ == "__main__":
    main()


