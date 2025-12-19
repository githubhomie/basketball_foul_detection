#!/bin/bash
#
# Run foul detection on a game video from S3.
#
# Usage:
#   ./scripts/inference/run_game_analysis.sh <s3_video_path> <game_id> [threshold]
#
# Examples:
#   ./scripts/inference/run_game_analysis.sh s3://nba-foul-dataset-oh/game_analysis/game001/input/game.mp4 game001
#   ./scripts/inference/run_game_analysis.sh s3://nba-foul-dataset-oh/game_analysis/game001/input/game.mp4 game001 0.20
#
# The script will:
#   1. Download video from S3
#   2. Run foul detection inference
#   3. Upload results (clips + detections.json) to S3
#   4. Clean up local files
#

set -e

# Configuration
S3_BUCKET="s3://nba-foul-dataset-oh"
CHECKPOINT_DIR="/data/checkpoints/basketball_v2_allin_20251130_044506"
WORK_DIR="/tmp/game_analysis"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <s3_video_path> <game_id> [threshold]"
    echo ""
    echo "Arguments:"
    echo "  s3_video_path  S3 path to the video file"
    echo "  game_id        Unique identifier for this game"
    echo "  threshold      Optional confidence threshold (default: 0.15)"
    exit 1
fi

S3_VIDEO_PATH="$1"
GAME_ID="$2"
THRESHOLD="${3:-0.15}"

# Derive paths
VIDEO_FILENAME=$(basename "$S3_VIDEO_PATH")
LOCAL_VIDEO="$WORK_DIR/input/$VIDEO_FILENAME"
OUTPUT_DIR="$WORK_DIR/output"
S3_OUTPUT_PATH="$S3_BUCKET/game_analysis/$GAME_ID/output/"

echo "============================================================"
echo "GAME ANALYSIS: $GAME_ID"
echo "============================================================"
echo "Video:      $S3_VIDEO_PATH"
echo "Threshold:  $THRESHOLD"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output:     $S3_OUTPUT_PATH"
echo ""

# Get repo root (script is in scripts/inference/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create work directories
mkdir -p "$WORK_DIR/input"
mkdir -p "$OUTPUT_DIR"

# Step 1: Download video from S3
echo "[1/4] Downloading video from S3..."
aws s3 cp "$S3_VIDEO_PATH" "$LOCAL_VIDEO"
echo "  Downloaded: $LOCAL_VIDEO"
echo ""

# Step 2: Run inference
echo "[2/4] Running foul detection..."
cd "$REPO_ROOT"

python scripts/inference/full_game_inference.py \
    --video "$LOCAL_VIDEO" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_DIR" \
    --threshold "$THRESHOLD" \
    --game-id "$GAME_ID"

echo ""

# Step 3: Upload results to S3
echo "[3/4] Uploading results to S3..."
aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT_PATH" --only-show-errors
echo "  Uploaded to: $S3_OUTPUT_PATH"
echo ""

# Step 4: Cleanup
echo "[4/4] Cleaning up..."
rm -rf "$WORK_DIR"
echo "  Done!"
echo ""

echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results available at:"
echo "  $S3_OUTPUT_PATH"
echo ""
echo "To download results:"
echo "  aws s3 sync $S3_OUTPUT_PATH ./game_analysis_$GAME_ID/"
echo ""
echo "To view detections:"
echo "  aws s3 cp ${S3_OUTPUT_PATH}detections.json - | jq ."
echo "============================================================"
