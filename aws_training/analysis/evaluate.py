#!/usr/bin/env python3
"""
Comprehensive model evaluation beyond mAP.

Extracted from TRAIN4.ipynb Cell 25.

This script evaluates:
1. Binary detection (foul vs no-foul clip-level)
2. Multi-class classification accuracy
3. Temporal accuracy (how close predictions are to ground truth)
4. Per-class breakdown

Usage:
    python aws_training/analysis/evaluate.py --checkpoint /data/checkpoints/experiment_xxx
    python aws_training/analysis/evaluate.py --checkpoint /data/checkpoints/experiment_xxx --split val
    python aws_training/analysis/evaluate.py --checkpoint /data/checkpoints/experiment_xxx --threshold 0.15
"""

import os
import sys
import json
import gzip
import argparse
from pathlib import Path
from glob import glob
from collections import defaultdict, Counter

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "basketball_foul_detection"))


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def load_predictions(checkpoint_dir: str, split: str = "test"):
    """Load prediction files from checkpoint directory."""
    # Find prediction files
    patterns = [
        f"{checkpoint_dir}/pred-{split}.*.recall.json.gz",
        f"{checkpoint_dir}/pred-{split}.*.json.gz",
        f"{checkpoint_dir}/pred-{split}.*.json",
    ]

    pred_files = []
    for pattern in patterns:
        pred_files.extend(glob(pattern))

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found for split '{split}' in {checkpoint_dir}\n"
            f"Looked for patterns like: pred-{split}.*.json.gz"
        )

    # Sort and get latest
    pred_files = sorted(pred_files)
    pred_path = pred_files[-1]

    # Extract epoch number from filename
    try:
        epoch = int(Path(pred_path).name.split('.')[1])
    except (IndexError, ValueError):
        epoch = -1

    # Load predictions
    if pred_path.endswith('.gz'):
        with gzip.open(pred_path, 'rt') as f:
            pred_data = json.load(f)
    else:
        with open(pred_path) as f:
            pred_data = json.load(f)

    return pred_data, epoch, pred_path


def apply_nms(events, window=3):
    """Apply non-maximum suppression to merge nearby predictions."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda x: x['score'], reverse=True)
    kept = []
    suppressed_frames = set()

    for event in sorted_events:
        frame = event['frame']
        is_suppressed = any(abs(frame - sf) <= window for sf in suppressed_frames)
        if not is_suppressed:
            kept.append(event)
            suppressed_frames.add(frame)

    return kept


def evaluate_detection(gt_by_video, pred_by_video, threshold=0.1, nms_window=3):
    """Evaluate binary detection (foul vs no-foul at clip level)."""
    results = {
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'missed_fouls': [],
        'false_alarms': []
    }

    for video, gt_entry in gt_by_video.items():
        pred_entry = pred_by_video.get(video, {'events': []})

        # Filter and apply NMS
        filtered = [e for e in pred_entry.get('events', []) if e['score'] >= threshold]
        filtered = apply_nms(filtered, nms_window)

        has_gt_foul = len(gt_entry.get('events', [])) > 0
        has_pred_foul = len(filtered) > 0

        if has_gt_foul:
            if has_pred_foul:
                results['tp'] += 1
            else:
                results['fn'] += 1
                results['missed_fouls'].append({
                    'video': video,
                    'gt_events': gt_entry['events'],
                    'all_preds': pred_entry.get('events', [])
                })
        else:
            if has_pred_foul:
                results['fp'] += 1
                results['false_alarms'].append({
                    'video': video,
                    'pred_events': filtered
                })
            else:
                results['tn'] += 1

    # Calculate metrics
    tp, fp, fn, tn = results['tp'], results['fp'], results['fn'], results['tn']

    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0

    return results


def evaluate_classification(gt_by_video, pred_by_video, threshold=0.1, tolerance=2, nms_window=3):
    """Evaluate multi-class classification accuracy."""
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    frame_errors = []

    for video, gt_entry in gt_by_video.items():
        gt_events = gt_entry.get('events', [])
        if not gt_events:
            continue

        pred_entry = pred_by_video.get(video, {'events': []})
        filtered = [e for e in pred_entry.get('events', []) if e['score'] >= threshold]
        filtered = apply_nms(filtered, nms_window)

        if not filtered:
            continue

        for gt_event in gt_events:
            gt_frame = gt_event['frame']
            gt_label = gt_event['label']

            # Find best matching prediction within tolerance
            best_pred = None
            best_dist = float('inf')

            for pred in filtered:
                dist = abs(pred['frame'] - gt_frame)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_pred = pred

            if best_pred:
                pred_label = best_pred['label']
                confusion[gt_label][pred_label] += 1
                total += 1

                if gt_label == pred_label:
                    correct += 1

                frame_errors.append(best_dist)

    return {
        'confusion': dict(confusion),
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'frame_errors': frame_errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--tolerance", type=int, default=2, help="Frame tolerance for matching")
    parser.add_argument("--nms-window", type=int, default=3, help="NMS window size")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    args = parser.parse_args()

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "basketball_foul_detection" / "data" / "basketball"
        if not data_dir.exists():
            data_dir = PROJECT_ROOT / "data" / "basketball"

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Split:         {args.split}")
    print(f"Threshold:     {args.threshold}")
    print(f"Tolerance:     ±{args.tolerance} frames")
    print(f"NMS window:    {args.nms_window} frames")
    print()

    # Load ground truth
    gt_path = data_dir / f"{args.split}.json"
    if not gt_path.exists():
        print(f"Error: Ground truth not found at {gt_path}")
        sys.exit(1)

    gt_data = load_json(str(gt_path))
    gt_by_video = {e['video']: e for e in gt_data}

    # Load predictions
    try:
        pred_data, epoch, pred_path = load_predictions(args.checkpoint, args.split)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    pred_by_video = {e['video']: e for e in pred_data}

    print(f"Predictions:   epoch {epoch}")
    print(f"               {Path(pred_path).name}")
    print(f"GT clips:      {len(gt_data)}")
    print(f"Pred clips:    {len(pred_data)}")
    print()

    # 1. Binary detection evaluation
    print("-" * 70)
    print("1. BINARY DETECTION (Foul vs No-Foul at Clip Level)")
    print("-" * 70)

    det_results = evaluate_detection(gt_by_video, pred_by_video, args.threshold, args.nms_window)

    foul_total = det_results['tp'] + det_results['fn']
    nonfoul_total = det_results['fp'] + det_results['tn']

    print(f"Foul clips:     {foul_total}")
    print(f"  Detected (TP): {det_results['tp']}")
    print(f"  Missed (FN):   {det_results['fn']}")
    print()
    print(f"Non-foul clips: {nonfoul_total}")
    print(f"  Correct (TN):  {det_results['tn']}")
    print(f"  False alarm:   {det_results['fp']}")
    print()
    print(f"Detection Recall:    {det_results['recall']:.1%} ({det_results['tp']}/{foul_total})")
    print(f"Detection Precision: {det_results['precision']:.1%}")
    print(f"Detection F1:        {det_results['f1']:.1%}")
    print(f"False Positive Rate: {det_results['fpr']:.1%} ({det_results['fp']}/{nonfoul_total})")
    print()

    # 2. Classification evaluation
    print("-" * 70)
    print("2. MULTI-CLASS CLASSIFICATION (When Detected)")
    print("-" * 70)

    class_results = evaluate_classification(gt_by_video, pred_by_video, args.threshold, args.tolerance, args.nms_window)

    print(f"Classification Accuracy: {class_results['accuracy']:.1%}")
    print(f"  Correct: {class_results['correct']} / {class_results['total']}")
    print()

    # Get all classes
    all_classes = sorted(set(class_results['confusion'].keys()))

    # Confusion matrix
    print("Confusion Matrix (rows=GT, cols=Pred):")
    header = "          " + " ".join([f"{c[:6]:>7}" for c in all_classes])
    print(header)

    for gt_class in all_classes:
        row_data = class_results['confusion'].get(gt_class, {})
        row = f"{gt_class[:8]:<10}" + " ".join([f"{row_data.get(c, 0):>7}" for c in all_classes])
        print(row)
    print()

    # Per-class metrics
    print("Per-Class Metrics:")
    for gt_class in all_classes:
        row = class_results['confusion'].get(gt_class, {})
        tp = row.get(gt_class, 0)
        total_gt = sum(row.values())
        total_pred = sum(class_results['confusion'].get(c, {}).get(gt_class, 0) for c in all_classes)

        recall = tp / total_gt if total_gt > 0 else 0
        precision = tp / total_pred if total_pred > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        status = " <- WEAK" if recall < 0.3 else ""
        print(f"  {gt_class:<18}: P={precision:.1%}  R={recall:.1%}  F1={f1:.1%}{status}")
    print()

    # 3. Temporal accuracy
    print("-" * 70)
    print("3. TEMPORAL ACCURACY")
    print("-" * 70)

    if class_results['frame_errors']:
        errors = np.array(class_results['frame_errors'])
        print(f"Mean frame error:   {errors.mean():.1f} frames ({errors.mean()/4:.2f} sec)")
        print(f"Median frame error: {np.median(errors):.1f} frames")
        print()
        print("Accuracy by tolerance:")
        for tol in [1, 2, 3, 4]:
            acc = np.mean(errors <= tol)
            print(f"  Within ±{tol} frame{'s' if tol > 1 else ''}:  {acc:.1%}")
    print()

    # 4. Failure cases summary
    print("-" * 70)
    print("4. FAILURE CASES")
    print("-" * 70)

    print(f"Missed fouls: {len(det_results['missed_fouls'])}")
    for case in det_results['missed_fouls'][:5]:
        gt_info = case['gt_events'][0]
        max_score = max([e['score'] for e in case['all_preds']], default=0)
        print(f"  {case['video']}: {gt_info['label']} @ frame {gt_info['frame']}, max_score={max_score:.3f}")

    if len(det_results['missed_fouls']) > 5:
        print(f"  ... and {len(det_results['missed_fouls']) - 5} more")
    print()

    print(f"False alarms: {len(det_results['false_alarms'])}")
    for case in det_results['false_alarms'][:5]:
        top_pred = case['pred_events'][0]
        print(f"  {case['video']}: predicted {top_pred['label']} (score={top_pred['score']:.3f})")

    if len(det_results['false_alarms']) > 5:
        print(f"  ... and {len(det_results['false_alarms']) - 5} more")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if det_results['f1'] >= 0.9:
        print("Detection: EXCELLENT")
    elif det_results['f1'] >= 0.7:
        print("Detection: GOOD")
    else:
        print("Detection: NEEDS IMPROVEMENT")

    if class_results['accuracy'] >= 0.7:
        print("Classification: GOOD")
    elif class_results['accuracy'] >= 0.5:
        print("Classification: MODERATE")
    else:
        print("Classification: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
