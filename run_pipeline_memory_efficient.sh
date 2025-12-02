#!/bin/bash
# Memory-efficient pipeline that processes ALL datasets without filtering
# Run this in tmux to persist after logout

set -e

cd /scratch/pedro.loures/nphard
source .venv/bin/activate

LOG_FILE="/scratch/pedro.loures/nphard/pipeline_memory_efficient.log"
COMPLETE_FLAG="/scratch/pedro.loures/nphard/pipeline_memory_efficient_complete.flag"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting Memory-Efficient TP2 Pipeline ==="

# Check available memory
if command -v free >/dev/null 2>&1; then
    AVAILABLE_MEM_GB=$(free -g | awk '/^Mem:/ {print $7}')
    log "  Available system memory: ${AVAILABLE_MEM_GB}GB"
fi

# Step 1: Run experiments in memory-efficient mode.
log "Step 1/4: Running experiments in memory-efficient mode..."
python -m tp2.experiments.run --datasets data/uci data/synthetic --output results/raw_memory_efficient --repetitions 15 --memory-efficient 2>&1 | tee -a "$LOG_FILE"
EXPERIMENT_EXIT_CODE=${PIPESTATUS[0]}
log "Step 1 complete: Experiments finished (exit code: $EXPERIMENT_EXIT_CODE)"

# Validate that results were generated
RESULT_COUNT=$(find results/raw_memory_efficient -name "*.parquet" 2>/dev/null | wc -l)
log "  Found $RESULT_COUNT result files"

if [ "$RESULT_COUNT" -eq 0 ]; then
    log "ERROR: No result files generated! Experiments may have failed."
    log "  Check the log above for errors."
    exit 1
fi

if [ "$EXPERIMENT_EXIT_CODE" -ne 0 ]; then
    log "WARNING: Experiments exited with code $EXPERIMENT_EXIT_CODE, but $RESULT_COUNT files were found."
    log "  Proceeding with analysis, but check logs for errors."
fi

# Step 2: Summarize results and generate plots
log "Step 2/4: Summarizing results and generating plots..."
python -m tp2.analysis.summarize --raw results/raw_memory_efficient --output results/summary_memory_efficient 2>&1 | tee -a "$LOG_FILE"
ANALYSIS_EXIT_CODE=${PIPESTATUS[0]}

if [ "$ANALYSIS_EXIT_CODE" -ne 0 ]; then
    log "ERROR: Analysis failed with exit code $ANALYSIS_EXIT_CODE"
    exit 1
fi

# Validate that summary files were created
if [ ! -f "results/summary_memory_efficient/summary.parquet" ] && [ ! -f "results/summary_memory_efficient/summary.csv" ]; then
    log "ERROR: Analysis did not generate summary files!"
    exit 1
fi

log "Step 2 complete: Analysis finished"

# Step 3: Copy all artifacts to report/ folder.
log "Step 3/4: Copying artifacts to report/ ..."
mkdir -p report/results/summary_memory_efficient

if [ -d "results/summary_memory_efficient" ] && [ "$(ls -A results/summary_memory_efficient 2>/dev/null)" ]; then
    cp -v results/summary_memory_efficient/* report/results/summary_memory_efficient/ 2>&1 | tee -a "$LOG_FILE"
    log "Step 3 complete: Artifacts copied to report/"
else
    log "ERROR: No files to copy from results/summary_memory_efficient/"
    exit 1
fi

# Step 4: Compile LaTeX report (if pdflatex is available)
log "Step 4/4: Compiling LaTeX report..."
if command -v pdflatex >/dev/null 2>&1; then
    cd report
    pdflatex -interaction=nonstopmode ieee.tex >> "$LOG_FILE" 2>&1 || log "Warning: First pdflatex had issues"
    if command -v bibtex >/dev/null 2>&1; then
        bibtex ieee >> "$LOG_FILE" 2>&1 || log "Warning: bibtex had issues (this is OK if no .bib file)"
    fi
    pdflatex -interaction=nonstopmode ieee.tex >> "$LOG_FILE" 2>&1 || log "Warning: Second pdflatex had issues"
    pdflatex -interaction=nonstopmode ieee.tex >> "$LOG_FILE" 2>&1 || log "Warning: Third pdflatex had issues"
    cd ..
    if [ -f "report/ieee.pdf" ]; then
        log "Step 4 complete: LaTeX compilation finished - PDF generated"
    else
        log "Step 4 complete: LaTeX compilation attempted but PDF not found"
    fi
else
    log "Step 4 skipped: pdflatex not available (install texlive or similar)"
    log "  You can compile manually later with: cd report && pdflatex ieee.tex"
fi

# Create completion flag
touch "$COMPLETE_FLAG"
log "=== Memory-Efficient Pipeline Complete! ==="
log "Check report/ieee.pdf for the final report"
log "All artifacts are in report/results/summary_memory_efficient/"


