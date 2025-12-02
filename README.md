# TP2 — 2-Approximate $k$-Center Study

Implementation of the DCC207/Algoritmos 2 practical work on approximation algorithms. The repository ships end-to-end tooling to:
[ NOTA DO ESTUDANTE - O README FOI ESCRITO POR UM AGENTE DE PROGRAMAÇÃO APENAS PARA FACILITAR REPRODUÇÃO, FAVOR NÃO CONSIDERAR O README PARA O TRABALHO ] 

- implement Minkowski ($p \ge 1$) and Mahalanobis distances without using prebuilt distance helpers;
- run the two 2-approximation $k$-center algorithms (farthest-first traversal and the interval-refinement variant with configurable final width);
- acquire 10 numeric UCI datasets (via OpenML) and generate 50 synthetic datasets mirroring the scikit-learn gallery plus custom Gaussian mixtures;
- execute the full experimental protocol (15 repetitions per configuration), log quality/radius/runtime metrics, and compare against the scikit-learn K-Means baseline;
- aggregate the results into publication-ready tables/plots and compile an IEEE-style report located in `report/`. For every generated image and key table, sidecar `.txt` and `.json` files describe the contents and store structured metadata.

## Project Layout

```
.
├── configs/                 # Dataset selection metadata
├── data/                    # External datasets (generated/downloaded)
├── docs/                    # Additional documentation
├── report/                  # IEEE LaTeX article + figures
├── results/                 # Raw logs + summary tables/plots
├── src/tp2/                 # Python package with algorithms & pipelines
└── tests/                   # (Add unit tests here as needed)
```

Key entry points:

| Purpose | Script |
| --- | --- |
| Download UCI datasets | `python -m tp2.data.uci --config configs/datasets.yaml --output data/uci` |
| Generate synthetic datasets | `python -m tp2.data.synthetic --output data/synthetic` |
| Run experiments | `python -m tp2.experiments.run --datasets data/uci data/synthetic --output results/raw` |
| Summarize & plot results (tables + images + sidecar .txt/.json) | `python -m tp2.analysis.summarize --raw results/raw --output results/summary` |

### Source code highlights

- `src/tp2/algorithms/_shared.py` collects distance adapters, assignment helpers, and radius calculations so the approximation algorithms stay focused on their control flow.
- `tests/` hosts smoke tests for Minkowski/Mahalanobis distances plus tiny k-center instances—extend this suite whenever you touch numerical routines.

## Environment Setup

1. **Python**: 3.11+ is recommended (matching the assignment guidelines).
2. **Virtual environment**:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate # Linux/macOS
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   # Optional dev tools
   pip install -e .[dev]
   ```

## Development workflow

- **Testing**: run `pytest` from the repository root to execute the smoke tests referenced above before pushing any refactor.
- **Linting (optional)**: `ruff check .` catches common issues quickly; substitute your preferred linter if needed.
- **Style**: keep inline comments concise and explanatory—clarify why code exists, not conversational messages to the reader.
- **Contributing**: see `CONTRIBUTING.md` for expectations around tests, shared helpers, and script verbosity.

## Data Acquisition Workflow

1. **UCI datasets**  
   - Edit `configs/datasets.yaml` if you want to swap datasets or tweak metadata.  
   - Execute:

     ```bash
     python -m tp2.data.uci --config configs/datasets.yaml --output data/uci
     ```

   - Each dataset folder will contain `features.parquet`, `labels.parquet`, and `metadata.json`.

2. **Synthetic datasets**  
   - The generator mirrors six scikit-learn gallery scenarios (5 datasets each) plus 10 custom Gaussian mixtures.  
   - Run:

     ```bash
     python -m tp2.data.synthetic --output data/synthetic --seed 0
     ```

   - Metadata for every dataset is stored beside the Parquet files to guarantee reproducibility.

> **Tip:** Large downloads may need OpenML credentials or caching; configure via `SKLEARN_DATA` env var if desired.

## Running Experiments

1. Ensure the `data/uci/*` and `data/synthetic/*` folders exist (each containing dataset subfolders).
2. Launch the orchestrator:

   ```bash
   python -m tp2.experiments.run \
       --datasets data/uci data/synthetic \
       --output results/raw \
       --repetitions 15
   ```

   - Distance matrices are cached per dataset/metric.
   - Algorithms executed: farthest-first traversal and interval refinement with widths `[25%, 18%, 12%, 8%, 4%]`.
   - Baseline: scikit-learn `KMeans`.
   - Metrics: solution radius, silhouette (computed with the same distance matrix), adjusted Rand index, runtime.

3. Raw logs are saved as Parquet files under `results/raw/`.

### Summaries and Visualizations

Aggregate logs and recreate the figures referenced in the report:

```bash
python -m tp2.analysis.summarize --raw results/raw --output results/summary
```

Artifacts generated in `results/summary/`:

- `summary.parquet` / `summary.csv`: Full detailed summary with mean ± std grouped by dataset, algorithm, metric, and interval width.
- `table_algorithm_comparison.tex` / `.csv`: Concise table comparing algorithms aggregated over all datasets.
- `table_metric_comparison.tex` / `.csv`: Concise table comparing distance metrics aggregated over all datasets.
- `table_interval_width.tex` / `.csv`: Analysis of interval width effects for interval-refinement algorithm.
- `summary.txt` / `summary.json`: Human-readable description and structured metadata for the summary tables.
- `silhouette_by_algorithm.png`, `ari_by_algorithm.png`, `runtime_sec_by_algorithm.png`: bar charts summarizing quality and runtime.
- For each PNG above there is a `*.txt` file describing the figure and a `*.json` file with metadata (figure name, metric, axes, grouping keys).


## Reproducibility Notes

- Each experiment repetition uses an explicit random seed logged in the raw tables.
- Synthetic datasets store their parameters (`seed`, number of centers, transformation matrices) in JSON metadata.
- Distance matrices are recomputed only when metric parameters change; see `tp2/distances/factory.py`.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `MemoryError` while computing distance matrices | Use the `chunk_size` parameter in metric configs (see `DistanceFactory`), or split datasets. |
| Slow Mahalanobis computations | Increase regularization (`--metric '{"name": "mahalanobis", "params": {"regularization": 1e-2}}'`) or precompute inverse covariances. |
| Missing OpenML dataset | Update `configs/datasets.yaml` with a reachable `openml_id` or download manually and place Parquet files in `data/uci/<dataset>` |

## Next Steps

- Add automated unit tests under `tests/` for distance functions and small clustering examples.
- Experiment with acceleration structures (cover trees, triangle inequality pruning) to handle datasets with tens of thousands of instances.
- Extend the analysis notebook to include statistical tests (e.g., Wilcoxon signed-rank) comparing algorithms.

---

For questions or clarifications, refer to `tp2.txt` (assignment brief) or contact the course staff.


