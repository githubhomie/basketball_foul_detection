#!/usr/bin/env python3
"""
Visualize frames from false positives and false negatives.

Extracted from TRAIN4.ipynb Cell 35.

Generates grid images showing the frames from:
- Missed fouls (false negatives)
- False alarms (false positives on non-foul clips)

Usage:
    python aws_training/analysis/visualize_failures.py \
        --checkpoint /data/checkpoints/experiment_xxx \
        --output /data/checkpoints/experiment_xxx/failure_analysis/

    python aws_training/analysis/visualize_failures.py \
        --checkpoint /data/checkpoints/experiment_xxx \
        --frame-dir /data/frames \
        --max-cases 20
"""

import os
import sys
import json
import gzip
import argparse
from pathlib import Path
from glob import glob

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "basketball_foul_detection"))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib or PIL not installed. Install with: pip install matplotlib pillow")


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
            return json.load(f)
    else:
        with open(pred_path) as f:
            return json.load(f)


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


def load_frames(frame_dir: str, clip_id: str):
    """Load all frames from a clip directory."""
    clip_path = Path(frame_dir) / clip_id

    if not clip_path.exists():
        # Try alternate naming conventions
        for alt in [clip_id.replace('_', '/'), clip_id]:
            alt_path = Path(frame_dir) / alt
            if alt_path.exists():
                clip_path = alt_path
                break
        else:
            return []

    # Find frame files
    frame_files = sorted(clip_path.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(clip_path.glob("*.png"))

    frames = []
    for f in frame_files[:30]:  # Limit to 30 frames
        try:
            img = Image.open(f)
            frames.append(np.array(img))
        except Exception as e:
            pass

    return frames


def create_frame_grid(frames, title, events=None, output_path=None, cols=6):
    """Create a grid visualization of frames."""
    if not frames:
        print(f"  No frames to visualize for: {title[:50]}")
        return False

    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    fig.suptitle(title, fontsize=12, fontweight='bold', wrap=True)

    # Flatten axes for easy iteration
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flatten()

    # Create set of event frames for highlighting
    event_frames = {}
    if events:
        for e in events:
            event_frames[e['frame']] = e

    for idx, ax in enumerate(axes_flat):
        if idx < len(frames):
            ax.imshow(frames[idx])

            # Highlight event frames
            if idx in event_frames:
                event = event_frames[idx]
                label = event.get('label', 'event')
                score = event.get('score', None)

                if score:
                    ax.set_title(f"F{idx}: {label[:8]}\n({score:.2f})",
                                fontsize=8, color='red', fontweight='bold')
                else:
                    ax.set_title(f"F{idx}: {label[:10]}",
                                fontsize=8, color='red', fontweight='bold')

                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                ax.set_title(f"Frame {idx}", fontsize=8)

        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return True
    else:
        plt.show()
        plt.close(fig)
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize Failure Cases")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--output", required=True, help="Output directory for visualizations")
    parser.add_argument("--frame-dir", default="/data/frames", help="Directory containing frames")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--max-cases", type=int, default=10, help="Max cases to visualize per category")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib and PIL are required for visualization")
        print("Install with: pip install matplotlib pillow")
        sys.exit(1)

    # Setup output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "false_negatives").mkdir(exist_ok=True)
    (output_dir / "false_positives").mkdir(exist_ok=True)

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "basketball_foul_detection" / "data" / "basketball"
        if not data_dir.exists():
            data_dir = PROJECT_ROOT / "data" / "basketball"

    print("=" * 60)
    print("FAILURE CASE VISUALIZATION")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Frame dir:   {args.frame_dir}")
    print(f"Output dir:  {args.output}")
    print(f"Threshold:   {args.threshold}")
    print(f"Max cases:   {args.max_cases}")
    print()

    # Load data
    gt_data = load_json(str(data_dir / f"{args.split}.json"))
    gt_by_video = {e['video']: e for e in gt_data}

    pred_data = load_predictions(args.checkpoint, args.split)
    pred_by_video = {e['video']: e for e in pred_data}

    # Find false negatives (missed fouls)
    false_negatives = []
    for video, gt_entry in gt_by_video.items():
        if not gt_entry.get('events'):
            continue

        pred_entry = pred_by_video.get(video, {'events': []})
        filtered = [e for e in pred_entry.get('events', []) if e['score'] >= args.threshold]
        filtered = apply_nms(filtered)

        if not filtered:
            false_negatives.append({
                'video': video,
                'gt_events': gt_entry['events'],
                'all_preds': pred_entry.get('events', [])
            })

    print(f"False Negatives (missed fouls): {len(false_negatives)}")

    # Visualize false negatives
    fn_count = 0
    for case in false_negatives[:args.max_cases]:
        video = case['video']
        frames = load_frames(args.frame_dir, video)

        if frames:
            gt_event = case['gt_events'][0]
            max_score = max([e['score'] for e in case['all_preds']], default=0)

            title = f"MISSED FOUL: {video}\nGT: {gt_event['label']} @ frame {gt_event['frame']}, max_pred_score={max_score:.3f}"
            output_path = output_dir / "false_negatives" / f"{fn_count+1:02d}_{video}.png"

            if create_frame_grid(frames, title, case['gt_events'], str(output_path)):
                fn_count += 1
                print(f"  Saved: {output_path.name}")

    print(f"  Generated {fn_count} visualizations")
    print()

    # Find false positives (false alarms on non-foul clips)
    false_positives = []
    for video, gt_entry in gt_by_video.items():
        if gt_entry.get('events'):
            continue  # Has ground truth, skip

        pred_entry = pred_by_video.get(video, {'events': []})
        filtered = [e for e in pred_entry.get('events', []) if e['score'] >= args.threshold]
        filtered = apply_nms(filtered)

        if filtered:
            false_positives.append({
                'video': video,
                'pred_events': filtered
            })

    # Sort by confidence (highest first)
    false_positives.sort(key=lambda x: max(e['score'] for e in x['pred_events']), reverse=True)

    print(f"False Positives (false alarms): {len(false_positives)}")

    # Visualize false positives
    fp_count = 0
    for case in false_positives[:args.max_cases]:
        video = case['video']
        frames = load_frames(args.frame_dir, video)

        if frames:
            top_preds = case['pred_events'][:3]
            pred_info = ", ".join([f"{e['label'][:6]}({e['score']:.2f})@{e['frame']}" for e in top_preds])

            title = f"FALSE ALARM: {video}\nPred: {pred_info}"
            output_path = output_dir / "false_positives" / f"{fp_count+1:02d}_{video}.png"

            if create_frame_grid(frames, title, case['pred_events'], str(output_path)):
                fp_count += 1
                print(f"  Saved: {output_path.name}")

    print(f"  Generated {fp_count} visualizations")
    print()

    print("=" * 60)
    print(f"Visualizations saved to: {args.output}")
    print("=" * 60)
    print()
    print("View the images:")
    print(f"  ls {args.output}/false_negatives/")
    print(f"  ls {args.output}/false_positives/")


if __name__ == "__main__":
    main()
