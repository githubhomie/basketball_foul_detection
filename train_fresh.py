#!/usr/bin/env python3
"""Fresh training start with locked-in config"""

import os
import sys
import subprocess
from datetime import datetime

print("="*80)
print("CHECKING PREREQUISITES")
print("="*80)

# Check 1: Repository exists
REPO_DIR = "/content/basketball_foul_detection"
if not os.path.exists(REPO_DIR):
    print(f"ERROR: Repository not found at {REPO_DIR}")
    print("Run the 'Clone Repository' cell first")
    sys.exit(1)
print(f"✓ Repository exists: {REPO_DIR}")

# Check 2: Change to repo directory
try:
    os.chdir(REPO_DIR)
    print(f"✓ Changed to: {os.getcwd()}")
except Exception as e:
    print(f"ERROR: Cannot change to repo directory: {e}")
    sys.exit(1)

# Check 3: Update repo to get mixup fix
print("\nUpdating repository...")
result = subprocess.run(["git", "pull", "origin", "main"],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Repository updated")
else:
    print(f"WARNING: Git pull failed (maybe already up to date)")

# Check 4: Frames directory exists
TRAINING_FRAME_DIR = "/content/frames_training"
if not os.path.exists(TRAINING_FRAME_DIR):
    print(f"\nERROR: Frames directory not found: {TRAINING_FRAME_DIR}")
    print("Run the 'Copy Frames to Local Storage' cell first")
    sys.exit(1)

# Count frames
try:
    frame_dirs = [d for d in os.listdir(TRAINING_FRAME_DIR)
                  if os.path.isdir(os.path.join(TRAINING_FRAME_DIR, d))]
    print(f"✓ Frames directory exists: {len(frame_dirs)} clips")
    if len(frame_dirs) < 1000:
        print(f"WARNING: Only {len(frame_dirs)} clips found (expected ~2200)")
except Exception as e:
    print(f"ERROR: Cannot read frames directory: {e}")
    sys.exit(1)

# Check 5: train_e2e.py exists
if not os.path.exists("train_e2e.py"):
    print("\nERROR: train_e2e.py not found")
    sys.exit(1)
print("✓ train_e2e.py exists")

# Check 6: Checkpoint directory
CHECKPOINT_BASE = "/content/drive/MyDrive/nba_foul_training/checkpoints"
if not os.path.exists(CHECKPOINT_BASE):
    print(f"\nERROR: Checkpoint directory not found: {CHECKPOINT_BASE}")
    print("Make sure Google Drive is mounted")
    sys.exit(1)
print(f"✓ Checkpoint directory exists")

print("\n" + "="*80)
print("ALL CHECKS PASSED - STARTING TRAINING")
print("="*80)
print()

# Configuration
DATASET = "basketball"
MODEL_ARCH = "rny002_gsm"
TEMPORAL_ARCH = "gru"
BATCH_SIZE = 24
CLIP_LEN = 30
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
CROP_DIM = 224

# Create new checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"{CHECKPOINT_BASE}/basketball_colab_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Configuration:")
print(f"  Dataset:        {DATASET}")
print(f"  Frame dir:      {TRAINING_FRAME_DIR}")
print(f"  Model:          {MODEL_ARCH} + {TEMPORAL_ARCH}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Epochs:         {NUM_EPOCHS}")
print(f"  Mixup:          False")
print(f"  Dilate len:     1")
print(f"  FG upsample:    0.5")
print(f"  Start val:      Epoch 5")
print(f"  Save dir:       {SAVE_DIR}")
print()
print("="*80)
print()

# Run training (without capture so output streams live)
cmd = [
    "python3", "train_e2e.py",
    DATASET,
    TRAINING_FRAME_DIR,
    "-m", MODEL_ARCH,
    "-t", TEMPORAL_ARCH,
    "-s", SAVE_DIR,
    "--clip_len", str(CLIP_LEN),
    "--crop_dim", str(CROP_DIM),
    "--batch_size", str(BATCH_SIZE),
    "--num_epochs", str(NUM_EPOCHS),
    "--learning_rate", str(LEARNING_RATE),
    "--mixup", "False",
    "--criterion", "map",
    "--dilate_len", "1",
    "--fg_upsample", "0.5",
    "--start_val_epoch", "5",
    "--warm_up_epochs", "3"
]

result = subprocess.run(cmd)
sys.exit(result.returncode)
