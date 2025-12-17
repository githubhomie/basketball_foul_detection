#!/bin/bash
# download_frames.sh - Download training frames from S3 to local storage
#
# This downloads ~22GB of frame data from your S3 bucket to the local
# EC2 instance storage for fast training I/O.
#
# Usage: bash aws_training/download_frames.sh

set -e

S3_BUCKET="s3://nba-foul-dataset-oh"
LOCAL_DIR="/data/frames"

echo "========================================"
echo "Downloading Training Frames from S3"
echo "========================================"
echo ""
echo "Source: $S3_BUCKET/frames/"
echo "Destination: $LOCAL_DIR"
echo ""

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found. Install with: pip install awscli"
    exit 1
fi

# Check S3 access
echo "Checking S3 access..."
if ! aws s3 ls $S3_BUCKET &> /dev/null; then
    echo "Error: Cannot access S3 bucket."
    echo "Make sure your EC2 instance has the IAM role attached,"
    echo "or configure AWS credentials with: aws configure"
    exit 1
fi
echo "S3 access confirmed!"
echo ""

# Check existing downloads
EXISTING_CLIPS=$(find $LOCAL_DIR -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$EXISTING_CLIPS" -gt 2000 ]; then
    echo "Found $EXISTING_CLIPS clips already downloaded."
    read -p "Re-download/sync? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Skipping download. Existing frames will be used."
        exit 0
    fi
fi

# Download frames
echo "Starting download..."
echo "This will take approximately 10-15 minutes for 22GB."
echo ""

# Use aws s3 sync for resumable transfers
# --only-show-errors reduces output noise
aws s3 sync $S3_BUCKET/frames/ $LOCAL_DIR/ \
    --only-show-errors

# Verify download
echo ""
echo "Download complete!"
echo ""

CLIP_COUNT=$(find $LOCAL_DIR -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
FRAME_COUNT=$(find $LOCAL_DIR -name "*.jpg" 2>/dev/null | wc -l)

echo "Statistics:"
echo "  Clips downloaded: $CLIP_COUNT"
echo "  Total frames: $FRAME_COUNT"
echo "  Expected: ~2,357 clips, ~70,710 frames"
echo ""

if [ "$CLIP_COUNT" -lt 2000 ]; then
    echo "WARNING: Fewer clips than expected!"
    echo "Try running this script again to resume download."
else
    echo "Download verified successfully!"
fi

# Show disk usage
echo ""
echo "Disk usage:"
du -sh $LOCAL_DIR
df -h /data
