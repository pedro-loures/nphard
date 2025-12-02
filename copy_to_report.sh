#!/bin/bash
# Copy latest summary artifacts into report/results/summary for LaTeX usage.

set -e

REPORT_DIR="/scratch/pedro.loures/nphard/report"
RESULTS_DIR="/scratch/pedro.loures/nphard/results/summary"

# Ensure the destination exists even if report/ was just cloned.
mkdir -p "${REPORT_DIR}/results/summary"

if [ -d "${RESULTS_DIR}" ]; then
    cp -v "${RESULTS_DIR}"/* "${REPORT_DIR}/results/summary/" 2>/dev/null || true
    echo "Summary artifacts copied to ${REPORT_DIR}/results/summary/"
else
    echo "Warning: ${RESULTS_DIR} does not exist yet. Run analysis first."
fi

