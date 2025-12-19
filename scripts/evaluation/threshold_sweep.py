#!/usr/bin/env python3
"""
Threshold sweep to find optimal operating point.

Extracted from TRAIN4.ipynb Cells 29, 33.

Tests multiple confidence thresholds and reports:
- Recall, Precision, F1, FPR at each threshold
- Best F1 threshold
- Best recall at 70%+ precision

Usage:
    python scripts/evaluation/threshold_sweep.py --checkpoint /data/checkpoints/experiment_xxx
    python scripts/evaluation/threshold_sweep.py --checkpoint /data/checkpoints/experiment_xxx --split val
"""

import os
import sys
import json
import gzip
import argparse
import csv
from pathlib import Path
from glob import glob
from collections import defaultdict
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def load_predictions(checkpoint_dir: str, split: str = "test"):
    """Load prediction files from checkpoint directory."""
    patterns = [
        f"{checkpoint_dir}/pred-{split}.*.recall.json.gz",
        f"{checkpoint_dir}/pred-{split}.*.json.gz",
    ]

    pred_path = None
    for pattern in patterns:
        matches = glob(pattern)
        # Exclude .score. files (different format - raw scores, not events)
        matches = [f for f in matches if '.score.' not in f]
        if matches:
            pred_path = sorted(matches)[-1]  # Latest epoch
            break

    if pred_path is None:
        raise FileNotFoundError(f"No prediction files found for '{split}' in {checkpoint_dir}")

    if pred_path.endswith('.gz'):
        with gzip.open(pred_path, 'rt') as f:
            return json.load(f), pred_path
    else:
        with open(pred_path) as f:
            return json.load(f), pred_path


def apply_nms(events, window=3):
    """Apply non-maximum suppression."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda x: x['score'], reverse=True)
    kept = []
    suppressed_frames = set()

    for event in sorted_events:
        frame = event['frame']
        if not any(abs(frame - sf) <= window for sf in suppressed_frames):
            kept.append(event)
            suppressed_frames.add(frame)

    return kept


def evaluate_at_threshold(gt_by_video, pred_by_video, threshold, nms_window=3):
    """Evaluate detection at a specific threshold."""
    tp, fp, fn, tn = 0, 0, 0, 0

    for video, gt_entry in gt_by_video.items():
        pred_entry = pred_by_video.get(video, {'events': []})

        filtered = [e for e in pred_entry.get('events', []) if e['score'] >= threshold]
        filtered = apply_nms(filtered, nms_window)

        has_gt = len(gt_entry.get('events', [])) > 0
        has_pred = len(filtered) > 0

        if has_gt:
            if has_pred:
                tp += 1
            else:
                fn += 1
        else:
            if has_pred:
                fp += 1
            else:
                tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'recall': recall,
        'precision': precision,
        'fpr': fpr,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description="Threshold Sweep Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--nms-window", type=int, default=3, help="NMS window size")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON/CSV in checkpoint/results/")
    args = parser.parse_args()

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "data" / "basketball"

    # Load ground truth
    gt_path = data_dir / f"{args.split}.json"
    gt_data = load_json(str(gt_path))
    gt_by_video = {e['video']: e for e in gt_data}

    # Load predictions
    pred_data, pred_path = load_predictions(args.checkpoint, args.split)
    pred_by_video = {e['video']: e for e in pred_data}

    # Count clips
    foul_clips = sum(1 for e in gt_data if len(e.get('events', [])) > 0)
    nonfoul_clips = len(gt_data) - foul_clips

    print("=" * 80)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Predictions: {Path(pred_path).name}")
    print(f"NMS window:  {args.nms_window} frames")
    print(f"Test set:    {foul_clips} foul clips, {nonfoul_clips} non-foul clips")
    print()

    # Thresholds to test
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

    print(f"{'Thresh':<8} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'FPR':>8} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-" * 80)

    results = []
    best_f1 = {'f1': 0}
    best_balanced = {'f1': 0}  # Best F1 with FPR < 10%
    best_recall_at_precision = {'recall': 0, 'precision': 0}

    for thresh in thresholds:
        result = evaluate_at_threshold(gt_by_video, pred_by_video, thresh, args.nms_window)
        results.append(result)

        # Mark best results
        marker = ""
        if result['f1'] > best_f1['f1']:
            best_f1 = result
        if result['f1'] > best_balanced['f1'] and result['fpr'] < 0.10:
            best_balanced = result
            marker = " <- BEST BALANCED"
        if result['precision'] >= 0.70 and result['recall'] > best_recall_at_precision['recall']:
            best_recall_at_precision = result

        print(f"{thresh:<8.2f} | {result['recall']:>7.1%} | {result['precision']:>9.1%} | "
              f"{result['f1']:>7.1%} | {result['fpr']:>7.1%} | "
              f"{result['tp']:>4} {result['fp']:>4} {result['fn']:>4} {result['tn']:>4}{marker}")

    print("-" * 80)
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print(f"BEST F1 SCORE: threshold = {best_f1['threshold']:.2f}")
    print(f"  Recall:    {best_f1['recall']:.1%} ({best_f1['tp']}/{best_f1['tp']+best_f1['fn']} fouls detected)")
    print(f"  Precision: {best_f1['precision']:.1%}")
    print(f"  F1 Score:  {best_f1['f1']:.1%}")
    print(f"  FPR:       {best_f1['fpr']:.1%}")
    print()

    if best_balanced['f1'] > 0:
        print(f"BEST BALANCED (F1 with FPR < 10%): threshold = {best_balanced['threshold']:.2f}")
        print(f"  Recall:    {best_balanced['recall']:.1%}")
        print(f"  Precision: {best_balanced['precision']:.1%}")
        print(f"  F1 Score:  {best_balanced['f1']:.1%}")
        print(f"  FPR:       {best_balanced['fpr']:.1%}")
        print()

    if best_recall_at_precision['precision'] >= 0.70:
        print(f"BEST RECALL @ 70%+ PRECISION: threshold = {best_recall_at_precision['threshold']:.2f}")
        print(f"  Recall:    {best_recall_at_precision['recall']:.1%}")
        print(f"  Precision: {best_recall_at_precision['precision']:.1%}")
        print()

    # Usage guidance
    print("=" * 80)
    print("USAGE GUIDANCE")
    print("=" * 80)
    print()
    print("For PRODUCTION (minimize false alarms):")
    print(f"  Use threshold = {best_balanced['threshold']:.2f}" if best_balanced['f1'] > 0 else "  Use threshold = 0.20+")
    print()
    print("For MAXIMUM RECALL (catch all fouls):")
    print(f"  Use threshold = 0.05 or lower")
    print()
    print("For BALANCED (default):")
    print(f"  Use threshold = {best_f1['threshold']:.2f}")

    # Save results if requested
    if args.save_results:
        results_dir = Path(args.checkpoint) / "results"
        results_dir.mkdir(exist_ok=True)

        # Save CSV (easy for plotting)
        csv_file = results_dir / f"threshold_sweep_{args.split}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['threshold', 'recall', 'precision', 'f1', 'fpr', 'tp', 'fp', 'fn', 'tn'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        # Save JSON (complete results + recommendations)
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "nms_window": args.nms_window,
            "foul_clips": foul_clips,
            "nonfoul_clips": nonfoul_clips,
            "sweep_results": results,
            "recommendations": {
                "best_f1": {
                    "threshold": best_f1['threshold'],
                    "recall": best_f1['recall'],
                    "precision": best_f1['precision'],
                    "f1": best_f1['f1'],
                    "fpr": best_f1['fpr'],
                },
                "best_balanced": {
                    "threshold": best_balanced['threshold'],
                    "recall": best_balanced['recall'],
                    "precision": best_balanced['precision'],
                    "f1": best_balanced['f1'],
                    "fpr": best_balanced['fpr'],
                } if best_balanced['f1'] > 0 else None,
                "best_recall_at_70_precision": {
                    "threshold": best_recall_at_precision['threshold'],
                    "recall": best_recall_at_precision['recall'],
                    "precision": best_recall_at_precision['precision'],
                } if best_recall_at_precision['precision'] >= 0.70 else None,
            }
        }

        json_file = results_dir / f"threshold_sweep_{args.split}.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save best threshold for easy loading
        best_file = results_dir / "best_threshold.json"
        with open(best_file, 'w') as f:
            json.dump({
                "threshold": best_f1['threshold'],
                "f1": best_f1['f1'],
                "recall": best_f1['recall'],
                "precision": best_f1['precision'],
            }, f, indent=2)

        print()
        print(f"Results saved to:")
        print(f"  {csv_file}")
        print(f"  {json_file}")
        print(f"  {best_file}")


if __name__ == "__main__":
    main()
