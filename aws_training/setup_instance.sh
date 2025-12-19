#!/bin/bash
# setup_instance.sh - One-time EC2 instance setup for NBA foul detection training
# Run this after first SSH into a new EC2 instance
#
# Usage: bash aws_training/setup_instance.sh

set -e  # Exit on any error

echo "========================================"
echo "NBA Foul Detection - AWS Instance Setup"
echo "========================================"
echo ""

# Check if we're on EC2
if [ ! -f /sys/hypervisor/uuid ] || [ "$(head -c 3 /sys/hypervisor/uuid)" != "ec2" ]; then
    echo "Warning: This doesn't appear to be an EC2 instance."
    echo "Some steps may not work as expected."
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# 1. Install system packages
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop tree ffmpeg

# Install nvtop for GPU monitoring (if available)
if command -v nvtop &> /dev/null; then
    echo "  nvtop already installed"
else
    sudo apt-get install -y -qq nvtop 2>/dev/null || echo "  nvtop not available, skipping"
fi

echo "  Done!"

# 2. Set up data directories
echo ""
echo "[2/6] Setting up data directories..."
sudo mkdir -p /data/frames
sudo mkdir -p /data/checkpoints
sudo chown -R ubuntu:ubuntu /data
echo "  Created /data/frames and /data/checkpoints"

# 3. Set up Python environment (venv preferred, conda fallback)
echo ""
echo "[3/6] Setting up Python environment..."

# Check for existing venv first (EC2 instance may already have one)
if [ -d ~/venv ]; then
    echo "  Found existing venv at ~/venv"
    source ~/venv/bin/activate
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ] || [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    # Fall back to conda if available
    if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    else
        source ~/miniconda3/etc/profile.d/conda.sh
    fi
    if conda env list | grep -q "foul_detection"; then
        echo "  Environment 'foul_detection' already exists"
        conda activate foul_detection
    else
        echo "  Creating conda environment 'foul_detection'..."
        conda create -n foul_detection python=3.10 -y
        conda activate foul_detection
    fi
else
    # Create new venv
    echo "  Creating venv at ~/venv..."
    python3 -m venv ~/venv
    source ~/venv/bin/activate
fi

echo "  Using Python: $(which python)"
echo "  Done!"

# 4. Install PyTorch and dependencies
echo ""
echo "[4/6] Installing PyTorch and dependencies..."

# Check if PyTorch is already installed
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch with CUDA..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Install project dependencies
echo "  Installing project dependencies..."
pip install -q \
    timm>=0.9.0 \
    numpy \
    pillow \
    matplotlib \
    tqdm \
    tabulate \
    opencv-python \
    wandb \
    boto3 \
    pyyaml \
    seaborn \
    pandas

echo "  Done!"

# 5. Verify GPU
echo ""
echo "[5/6] Verifying GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected!')
"

# 6. Configure tmux
echo ""
echo "[6/6] Configuring tmux..."
cat > ~/.tmux.conf << 'EOF'
# Enable mouse support (for scrolling)
set -g mouse on

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1

# Status bar
set -g status-bg black
set -g status-fg white
set -g status-left '#[fg=green][#S] '
set -g status-right '#[fg=yellow]GPU: #(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")% | #[fg=cyan]%H:%M'
set -g status-interval 5
set -g status-right-length 50

# Increase scrollback buffer
set -g history-limit 50000

# Don't rename windows automatically
set-option -g allow-rename off
EOF
echo "  Done!"

# Final summary
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Log in to Weights & Biases:"
echo "   wandb login"
echo "   (paste your API key from https://wandb.ai/authorize)"
echo ""
echo "2. Download training frames from S3:"
echo "   bash aws_training/download_frames.sh"
echo "   (takes ~10-15 minutes for 22GB)"
echo ""
echo "3. Start training:"
echo "   tmux new -s train"
echo "   source ~/venv/bin/activate  # or: conda activate foul_detection"
echo "   python aws_training/train_wrapper.py --config aws_training/configs/v2_baseline.yaml"
echo ""
echo "4. To detach from tmux (training keeps running):"
echo "   Press Ctrl+B, then D"
echo ""
echo "5. To reconnect later:"
echo "   tmux attach -t train"
echo ""
