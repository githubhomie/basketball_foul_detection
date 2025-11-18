#!/bin/bash
#
# Training script for NBA basketball foul detection
# Uses E2E-Spot architecture trained from ImageNet initialization
#

# Configuration
DATASET="basketball"
FRAME_DIR="${FRAME_DIR:-/home/ubuntu/frames}"  # Default EC2 path, override with: export FRAME_DIR=/your/path
MODEL_ARCH="rny002_gsm"  # RegNet-Y 200MF + Gated Shift Module
TEMPORAL_ARCH="gru"  # Bidirectional GRU
SAVE_DIR="./checkpoints/basketball_$(date +%Y%m%d_%H%M%S)"

# Model parameters
CLIP_LEN=30  # Our clips are 30 frames
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.001
CROP_DIM=224

# Training options
MIXUP=True
CRITERION="map"  # Optimize for mAP
DILATE_LEN=0  # No label dilation
WARM_UP_EPOCHS=3

# Create save directory
mkdir -p "$SAVE_DIR"

# Print configuration
echo "=================================================================================================="
echo "NBA BASKETBALL FOUL DETECTION TRAINING"
echo "=================================================================================================="
echo "Dataset:        $DATASET"
echo "Frame dir:      $FRAME_DIR"
echo "Model:          $MODEL_ARCH + $TEMPORAL_ARCH"
echo "Clip length:    $CLIP_LEN frames"
echo "Batch size:     $BATCH_SIZE"
echo "Epochs:         $NUM_EPOCHS"
echo "Learning rate:  $LEARNING_RATE"
echo "Save dir:       $SAVE_DIR"
echo "=================================================================================================="
echo ""

# Run training
python3 train_e2e.py "$DATASET" "$FRAME_DIR" \
    -m "$MODEL_ARCH" \
    -t "$TEMPORAL_ARCH" \
    -s "$SAVE_DIR" \
    --clip_len "$CLIP_LEN" \
    --crop_dim "$CROP_DIM" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --mixup "$MIXUP" \
    --criterion "$CRITERION" \
    --dilate_len "$DILATE_LEN" \
    --warm_up_epochs "$WARM_UP_EPOCHS" \
    "$@"  # Pass any additional arguments

echo ""
echo "=================================================================================================="
echo "Training complete! Results saved to: $SAVE_DIR"
echo "=================================================================================================="
