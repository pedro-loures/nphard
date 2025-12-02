#!/bin/bash
# Quick status check for the background pipeline

LOG_FILE="/scratch/pedro.loures/nphard/pipeline_background.log"
COMPLETE_FLAG="/scratch/pedro.loures/nphard/pipeline_complete.flag"

echo "=== Pipeline Status ==="

# Check if tmux session exists
if tmux has-session -t tp2_background 2>/dev/null; then
    echo "✓ Pipeline is RUNNING in tmux session 'tp2_background'"
    echo ""
    echo "Recent log output:"
    tail -5 "$LOG_FILE" 2>/dev/null || echo "  (log file not created yet)"
else
    if [ -f "$COMPLETE_FLAG" ]; then
        echo "✓ Pipeline has COMPLETED"
        echo ""
        echo "Final log output:"
        tail -10 "$LOG_FILE" 2>/dev/null
        echo ""
        echo "Check report/ieee.pdf for the final report"
    else
        echo "✗ Pipeline session not found and no completion flag"
        echo "  It may have crashed. Check: tail -f $LOG_FILE"
    fi
fi

echo ""
echo "To monitor live: tmux attach -t tp2_background"
echo "To view log: tail -f $LOG_FILE"


