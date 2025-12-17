#!/bin/bash
# sync_checkpoints.sh - Backup checkpoints to S3
#
# Usage:
#   bash aws_training/sync_checkpoints.sh              # Sync all checkpoints
#   bash aws_training/sync_checkpoints.sh experiment_name  # Sync specific experiment

S3_BUCKET="s3://nba-foul-checkpoints-oh"
LOCAL_DIR="/data/checkpoints"

echo "========================================"
echo "Syncing Checkpoints to S3"
echo "========================================"
echo ""

if [ -n "$1" ]; then
    # Sync specific experiment
    EXPERIMENT=$1
    if [ ! -d "$LOCAL_DIR/$EXPERIMENT" ]; then
        echo "Error: Experiment '$EXPERIMENT' not found in $LOCAL_DIR"
        exit 1
    fi
    echo "Syncing experiment: $EXPERIMENT"
    echo ""
    aws s3 sync "$LOCAL_DIR/$EXPERIMENT" "$S3_BUCKET/checkpoints/$EXPERIMENT" \
        --only-show-errors
    echo "Done!"
    echo "View at: $S3_BUCKET/checkpoints/$EXPERIMENT"
else
    # Sync all experiments
    echo "Syncing all checkpoints..."
    echo ""
    aws s3 sync "$LOCAL_DIR" "$S3_BUCKET/checkpoints" \
        --only-show-errors
    echo "Done!"
    echo "View at: $S3_BUCKET/checkpoints/"
fi

echo ""
echo "To download checkpoints later:"
echo "  aws s3 sync $S3_BUCKET/checkpoints/ ./checkpoints/"
