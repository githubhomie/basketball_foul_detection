#!/bin/bash
# sync_checkpoints.sh - Backup checkpoints and results to S3
#
# Usage:
#   bash aws_training/sync_checkpoints.sh                        # Sync all checkpoints
#   bash aws_training/sync_checkpoints.sh experiment_name        # Sync specific experiment
#   bash aws_training/sync_checkpoints.sh --results-only exp_xxx # Sync results only (faster)

S3_BUCKET="s3://nba-foul-checkpoints-oh"
LOCAL_DIR="/data/checkpoints"

# Check for --results-only flag
RESULTS_ONLY=false
if [ "$1" = "--results-only" ]; then
    RESULTS_ONLY=true
    shift
fi

echo "========================================"
if [ "$RESULTS_ONLY" = true ]; then
    echo "Syncing Results to S3"
else
    echo "Syncing Checkpoints + Results to S3"
fi
echo "========================================"
echo ""

if [ -n "$1" ]; then
    # Sync specific experiment
    EXPERIMENT=$1
    if [ ! -d "$LOCAL_DIR/$EXPERIMENT" ]; then
        echo "Error: Experiment '$EXPERIMENT' not found in $LOCAL_DIR"
        exit 1
    fi

    if [ "$RESULTS_ONLY" = true ]; then
        # Only sync results subdirectory
        if [ ! -d "$LOCAL_DIR/$EXPERIMENT/results" ]; then
            echo "Error: No results directory in $EXPERIMENT"
            exit 1
        fi
        echo "Syncing results only: $EXPERIMENT/results/"
        echo ""
        aws s3 sync "$LOCAL_DIR/$EXPERIMENT/results" "$S3_BUCKET/$EXPERIMENT/results" \
            --only-show-errors
        echo "Done!"
        echo "View at: $S3_BUCKET/$EXPERIMENT/results/"
    else
        # Sync entire experiment
        echo "Syncing experiment: $EXPERIMENT"
        echo ""
        aws s3 sync "$LOCAL_DIR/$EXPERIMENT" "$S3_BUCKET/$EXPERIMENT" \
            --only-show-errors
        echo "Done!"
        echo "View at: $S3_BUCKET/$EXPERIMENT"
    fi
else
    if [ "$RESULTS_ONLY" = true ]; then
        echo "Error: --results-only requires an experiment name"
        echo "Usage: bash sync_checkpoints.sh --results-only experiment_name"
        exit 1
    fi

    # Sync all experiments
    echo "Syncing all checkpoints..."
    echo ""
    aws s3 sync "$LOCAL_DIR" "$S3_BUCKET" \
        --only-show-errors
    echo "Done!"
    echo "View at: $S3_BUCKET/"
fi

echo ""
echo "To download checkpoints later:"
echo "  aws s3 sync $S3_BUCKET/ ./checkpoints/"
