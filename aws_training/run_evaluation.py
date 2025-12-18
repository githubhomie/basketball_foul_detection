#!/usr/bin/env python3
"""
Run full evaluation suite and save results.

One-command evaluation for a trained model checkpoint.
Runs evaluate.py and threshold_sweep.py, saves results to checkpoint/results/,
and optionally syncs to S3.

Usage:
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx --sync-s3
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx --split val
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: {description} returned exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run full evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation on test set
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx

    # Evaluate on validation set
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx --split val

    # Evaluate and sync results to S3
    python aws_training/run_evaluation.py --checkpoint /data/checkpoints/exp_xxx --sync-s3
"""
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Data split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.10, help="Default threshold for evaluate.py")
    parser.add_argument("--sync-s3", action="store_true", help="Sync results to S3 after evaluation")
    parser.add_argument("--s3-bucket", default="nba-foul-checkpoints-oh", help="S3 bucket for results")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("NBA FOUL DETECTION - FULL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint:   {checkpoint_dir}")
    print(f"Split:        {args.split}")
    print(f"Results dir:  {results_dir}")
    print()

    # 1. Run threshold sweep first (to find best threshold)
    threshold_cmd = [
        sys.executable,
        "aws_training/analysis/threshold_sweep.py",
        "--checkpoint", str(checkpoint_dir),
        "--split", args.split,
        "--save-results"
    ]
    run_command(threshold_cmd, "Threshold Sweep Analysis")

    # 2. Load best threshold from results
    best_threshold_file = results_dir / "best_threshold.json"
    if best_threshold_file.exists():
        import json
        with open(best_threshold_file) as f:
            best = json.load(f)
        threshold = best.get("threshold", args.threshold)
        print(f"\nUsing best threshold from sweep: {threshold:.2f}")
    else:
        threshold = args.threshold
        print(f"\nUsing default threshold: {threshold:.2f}")

    # 3. Run detailed evaluation at best threshold
    eval_cmd = [
        sys.executable,
        "aws_training/analysis/evaluate.py",
        "--checkpoint", str(checkpoint_dir),
        "--split", args.split,
        "--threshold", str(threshold),
        "--save-results"
    ]
    run_command(eval_cmd, "Detailed Evaluation")

    # 4. Sync to S3 if requested
    if args.sync_s3:
        exp_name = checkpoint_dir.name
        s3_path = f"s3://{args.s3_bucket}/{exp_name}/results/"

        sync_cmd = [
            "aws", "s3", "sync",
            str(results_dir),
            s3_path,
            "--only-show-errors"
        ]
        print(f"\n{'='*60}")
        print("Syncing results to S3")
        print(f"{'='*60}")
        print(f"$ {' '.join(sync_cmd)}")

        result = subprocess.run(sync_cmd)
        if result.returncode == 0:
            print(f"Results synced to: {s3_path}")
        else:
            print("WARNING: S3 sync failed")

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}")
    print("\nFiles created:")
    for f in sorted(results_dir.glob("*")):
        print(f"  - {f.name}")

    # Print key metrics if available
    eval_file = results_dir / f"eval_{args.split}_t{threshold:.2f}.json"
    if eval_file.exists():
        import json
        with open(eval_file) as f:
            results = json.load(f)

        print(f"\nKey Metrics (threshold={threshold:.2f}):")
        det = results.get("detection", {})
        print(f"  Detection Recall:    {det.get('recall', 0):.1%}")
        print(f"  Detection Precision: {det.get('precision', 0):.1%}")
        print(f"  Detection F1:        {det.get('f1', 0):.1%}")

        cls = results.get("classification", {})
        print(f"  Classification Acc:  {cls.get('accuracy', 0):.1%}")


if __name__ == "__main__":
    main()
