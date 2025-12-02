#!/bin/bash

# Abort immediately so partial outputs do not flow into later stages.
set -e

cd /scratch/pedro.loures/nphard

# Ensure the virtual environment provides every runtime dependency.
source .venv/bin/activate

echo "Step 1: Running experiments..."
python -m tp2.experiments.run --datasets data/uci data/synthetic --output results/raw --repetitions 15

echo "Step 2: Summarizing results and generating plots..."
python -m tp2.analysis.summarize --raw results/raw --output results/summary

# Copy summary artifacts so the LaTeX project has everything it needs.
echo "Step 3: Copying all artifacts to report/ folder..."
mkdir -p report/results/summary
cp -v results/summary/* report/results/summary/

echo "Step 4: Compiling LaTeX report..."
cd report

# LaTeX tools are noisy; run quietly and ignore non-critical warnings.
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true
bibtex ieee > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true

echo "Pipeline complete! Check report/ieee.pdf"

