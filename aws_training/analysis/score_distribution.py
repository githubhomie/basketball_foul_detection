#!/usr/bin/env python3
"""
Analyze prediction score distributions.

Extracted from TRAIN4.ipynb Cell 27.

Shows:
- Score distribution for foul vs non-foul clips
- Histogram and CDF plots
- Percentile breakdown

Usage:
    python aws_training/analysis/score_distribution.py --checkpoint /data/checkpoints/experiment_xxx
    python aws_training/analysis/score_distribution.py --checkpoint /data/checkpoints/experiment_xxx --output scores.png
"""

import os
import sys
import json
import gzip
import argparse
from pathlib import Path
from glob import glob
from collections import Counter

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "basketball_foul_detection"))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def load_predictions(checkpoint_dir: str, split: str = "test"):
    """Load prediction files."""
    patterns = [
        f"{checkpoint_dir}/pred-{split}.*.recall.json.gz",
        f"{checkpoint_dir}/pred-{split}.*.json.gz",
    ]

    pred_files = []
    for pattern in patterns:
        pred_files.extend(glob(pattern))

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found for '{split}'")

    pred_path = sorted(pred_files)[-1]

    if pred_path.endswith('.gz'):
        with gzip.open(pred_path, 'rt') as f:
            return json.load(f), pred_path
    else:
        with open(pred_path) as f:
            return json.load(f), pred_path


def main():
    parser = argparse.ArgumentParser(description="Score Distribution Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", help="Save plot to file (optional)")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    args = parser.parse_args()

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "basketball_foul_detection" / "data" / "basketball"
        if not data_dir.exists():
            data_dir = PROJECT_ROOT / "data" / "basketball"

    # Load data
    gt_data = load_json(str(data_dir / f"{args.split}.json"))
    gt_by_video = {e['video']: bool(e.get('events')) for e in gt_data}

    pred_data, pred_path = load_predictions(args.checkpoint, args.split)

    print("=" * 60)
    print("PREDICTION SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Predictions: {Path(pred_path).name}")
    print()

    # Collect scores by clip type
    foul_clip_max_scores = []
    nonfoul_clip_max_scores = []
    all_scores = []

    for entry in pred_data:
        video = entry['video']
        events = entry.get('events', [])
        is_foul_clip = gt_by_video.get(video, False)

        # Get max score for this clip
        if events:
            max_score = max(e['score'] for e in events)
            all_scores.extend([e['score'] for e in events])
        else:
            max_score = 0

        if is_foul_clip:
            foul_clip_max_scores.append(max_score)
        else:
            nonfoul_clip_max_scores.append(max_score)

    # Statistics
    print(f"Total clips: {len(pred_data)}")
    print(f"  Foul clips:     {len(foul_clip_max_scores)}")
    print(f"  Non-foul clips: {len(nonfoul_clip_max_scores)}")
    print(f"Total predictions: {len(all_scores)}")
    print()

    # Foul clip statistics
    if foul_clip_max_scores:
        foul_scores = np.array(foul_clip_max_scores)
        print("FOUL CLIPS (max score per clip):")
        print(f"  Mean:   {foul_scores.mean():.4f}")
        print(f"  Median: {np.median(foul_scores):.4f}")
        print(f"  Min:    {foul_scores.min():.4f}")
        print(f"  Max:    {foul_scores.max():.4f}")
        print(f"  Std:    {foul_scores.std():.4f}")
        print()

    # Non-foul clip statistics
    if nonfoul_clip_max_scores:
        nonfoul_scores = np.array(nonfoul_clip_max_scores)
        print("NON-FOUL CLIPS (max score per clip):")
        print(f"  Mean:   {nonfoul_scores.mean():.4f}")
        print(f"  Median: {np.median(nonfoul_scores):.4f}")
        print(f"  Min:    {nonfoul_scores.min():.4f}")
        print(f"  Max:    {nonfoul_scores.max():.4f}")
        print(f"  Std:    {nonfoul_scores.std():.4f}")
        print()

    # All scores percentiles
    if all_scores:
        all_arr = np.array(all_scores)
        print("ALL PREDICTION SCORES (percentiles):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(all_arr, p):.4f}")
        print()

    # Score bins
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]

    print("SCORE DISTRIBUTION (max score per clip):")
    print(f"{'Bin':<12} | {'Foul Clips':>12} | {'Non-Foul':>12} | {'Separation'}")
    print("-" * 55)

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]

        foul_in_bin = sum(1 for s in foul_clip_max_scores if low <= s < high)
        nonfoul_in_bin = sum(1 for s in nonfoul_clip_max_scores if low <= s < high)

        foul_pct = 100 * foul_in_bin / len(foul_clip_max_scores) if foul_clip_max_scores else 0
        nonfoul_pct = 100 * nonfoul_in_bin / len(nonfoul_clip_max_scores) if nonfoul_clip_max_scores else 0

        # Visual indicator of separation
        if foul_pct > nonfoul_pct + 10:
            sep = "++ (foul)"
        elif nonfoul_pct > foul_pct + 10:
            sep = "-- (non-foul)"
        else:
            sep = "~~ (overlap)"

        print(f"[{low:.2f}, {high:.2f}) | {foul_in_bin:>5} ({foul_pct:>5.1f}%) | {nonfoul_in_bin:>5} ({nonfoul_pct:>5.1f}%) | {sep}")

    print()

    # Threshold recommendation
    print("=" * 60)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 60)

    # Find threshold where foul clips mostly have higher scores
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        foul_above = sum(1 for s in foul_clip_max_scores if s >= thresh)
        nonfoul_above = sum(1 for s in nonfoul_clip_max_scores if s >= thresh)

        foul_recall = foul_above / len(foul_clip_max_scores) if foul_clip_max_scores else 0
        nonfoul_fp = nonfoul_above / len(nonfoul_clip_max_scores) if nonfoul_clip_max_scores else 0

        print(f"threshold={thresh:.2f}: {foul_recall:.1%} foul recall, {nonfoul_fp:.1%} FPR")

    print()

    # Generate plot
    if MATPLOTLIB_AVAILABLE and (args.output or True):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1 = axes[0]
        if foul_clip_max_scores:
            ax1.hist(foul_clip_max_scores, bins=20, alpha=0.7, label='Foul clips', color='red', density=True)
        if nonfoul_clip_max_scores:
            ax1.hist(nonfoul_clip_max_scores, bins=20, alpha=0.7, label='Non-foul clips', color='blue', density=True)
        ax1.set_xlabel('Max Prediction Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution: Foul vs Non-Foul Clips')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add threshold line
        ax1.axvline(x=0.1, color='green', linestyle='--', label='threshold=0.1')

        # CDF
        ax2 = axes[1]
        for scores, label, color in [(foul_clip_max_scores, 'Foul', 'red'),
                                      (nonfoul_clip_max_scores, 'Non-Foul', 'blue')]:
            if scores:
                sorted_scores = np.sort(scores)
                cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                ax2.plot(sorted_scores, cdf, label=label, color=color, linewidth=2)

        ax2.set_xlabel('Max Prediction Score')
        ax2.set_ylabel('Cumulative Proportion')
        ax2.set_title('CDF of Max Scores by Clip Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {args.output}")
        else:
            # Save to checkpoint directory
            default_output = Path(args.checkpoint) / "score_distribution.png"
            plt.savefig(str(default_output), dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {default_output}")

        plt.close()


if __name__ == "__main__":
    main()
