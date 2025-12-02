#!/bin/bash
set -e

cd /scratch/pedro.loures/nphard
source .venv/bin/activate

# Poll until all experiment processes finish.
while pgrep -f "tp2.experiments.run" > /dev/null; do
    sleep 30
done

# Summarize once distance matrices exist.
python -m tp2.analysis.summarize --raw results/raw --output results/summary

# Copy summary artifacts into the LaTeX tree.
mkdir -p report/results/summary
cp -v results/summary/* report/results/summary/

# Compile the paper quietly; ignore non-critical warnings.
cd report
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true
bibtex ieee > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode ieee.tex > /dev/null 2>&1 || true

echo "Pipeline complete! Check report/ieee.pdf"

