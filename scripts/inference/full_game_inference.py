#!/usr/bin/env python3
"""
Full game inference for basketball foul detection.

Takes a video file, runs foul detection, and outputs:
1. detections.json with all detected fouls (timestamp, type, confidence)
2. Video clips centered on each detected foul

Usage:
    python scripts/inference/full_game_inference.py \
        --video /path/to/game.mp4 \
        --output-dir /path/to/output \
        --checkpoint /path/to/checkpoint_dir \
        --threshold 0.15

    # With custom settings
    python scripts/inference/full_game_inference.py \
        --video /path/to/game.mp4 \
        --output-dir /path/to/output \
        --checkpoint /path/to/checkpoint_dir \
        --threshold 0.20 \
        --clip-duration 10 \
        --nms-window 12
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.frame import ActionSpotVideoDataset
from train_e2e import E2EModel, evaluate
from util.eval import non_maximum_supression
from util.io import load_json
from util.dataset import load_classes


def extract_frames(video_path: str, output_dir: str, fps: int = 4) -> int:
    """Extract frames from video at specified FPS using ffmpeg.

    Returns the number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames at target FPS
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality JPEG
        os.path.join(output_dir, '%06d.jpg'),
        '-y',  # Overwrite
        '-hide_banner', '-loglevel', 'error'
    ]

    print(f"Extracting frames at {fps} FPS...")
    subprocess.run(cmd, check=True)

    # Count extracted frames
    frame_files = list(Path(output_dir).glob('*.jpg'))
    num_frames = len(frame_files)
    print(f"  Extracted {num_frames} frames")

    return num_frames


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def create_temp_label_file(video_name: str, num_frames: int, fps: int, output_path: str):
    """Create a minimal label JSON file for the dataset."""
    labels = [{
        'video': video_name,
        'num_frames': num_frames,
        'fps': fps,
        'events': []  # No ground truth for inference
    }]

    with open(output_path, 'w') as f:
        json.dump(labels, f)


def frame_to_timestamp(frame: int, fps: int) -> str:
    """Convert frame number to timestamp string (HH:MM:SS.mmm)."""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def extract_clip(video_path: str, output_path: str, center_time: float,
                 duration: float = 10.0):
    """Extract a clip centered on the given time."""
    start_time = max(0, center_time - duration / 2)

    cmd = [
        'ffmpeg', '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',  # Fast copy without re-encoding
        output_path,
        '-y',  # Overwrite
        '-hide_banner', '-loglevel', 'error'
    ]

    subprocess.run(cmd, check=True)


def run_inference(args):
    """Main inference pipeline."""

    print("=" * 60)
    print("FULL GAME FOUL DETECTION")
    print("=" * 60)
    print(f"Video:      {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold:  {args.threshold}")
    print(f"NMS window: {args.nms_window} frames")
    print(f"Output:     {args.output_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / 'clips'
    clips_dir.mkdir(exist_ok=True)

    # Get video duration
    video_duration = get_video_duration(args.video)
    print(f"Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix='foul_detection_')
    frames_dir = os.path.join(temp_dir, 'game_video')
    os.makedirs(frames_dir)

    try:
        # Step 1: Extract frames
        print("\n[1/5] Extracting frames...")
        inference_fps = 4  # Must match training
        num_frames = extract_frames(args.video, frames_dir, fps=inference_fps)

        # Step 2: Create dataset
        print("\n[2/5] Setting up dataset...")
        label_file = os.path.join(temp_dir, 'labels.json')
        create_temp_label_file('game_video', num_frames, inference_fps, label_file)

        # Load model config
        config_path = os.path.join(args.checkpoint, 'config.json')
        config = load_json(config_path)

        # Load classes
        classes = load_classes(os.path.join(PROJECT_ROOT, 'data', config['dataset'], 'class.txt'))

        # Create dataset with overlap for better detection
        dataset = ActionSpotVideoDataset(
            classes=classes,
            label_file=label_file,
            frame_dir=temp_dir,
            modality=config['modality'],
            clip_len=config['clip_len'],
            overlap_len=config['clip_len'] // 2,  # 50% overlap
            crop_dim=config['crop_dim']
        )

        print(f"  Clip length: {config['clip_len']} frames")
        print(f"  Total clips: {len(dataset)}")

        # Step 3: Load model
        print("\n[3/5] Loading model...")
        model = E2EModel(
            len(classes) + 1,
            config['feature_arch'],
            config['temporal_arch'],
            clip_len=config['clip_len'],
            modality=config['modality'],
            multi_gpu=config.get('gpu_parallel', False)
        )

        # Find checkpoint file
        checkpoint_files = list(Path(args.checkpoint).glob('checkpoint_*.pt'))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {args.checkpoint}")
        checkpoint_file = sorted(checkpoint_files)[-1]  # Latest epoch
        print(f"  Loading: {checkpoint_file.name}")

        model.load(torch.load(str(checkpoint_file)))

        # Step 4: Run inference
        print("\n[4/5] Running inference...")
        pred_file = os.path.join(temp_dir, 'predictions')
        evaluate(model, dataset, 'INFERENCE', classes, pred_file,
                 calc_stats=False, save_scores=False)

        # Load predictions (high-recall version for flexibility)
        import gzip
        with gzip.open(pred_file + '.recall.json.gz', 'rt') as f:
            predictions = json.load(f)

        # Step 5: Post-process and extract clips
        print("\n[5/5] Post-processing detections...")

        # Filter by threshold
        for video_pred in predictions:
            video_pred['events'] = [
                e for e in video_pred['events']
                if e['score'] >= args.threshold
            ]

        # Apply NMS
        predictions = non_maximum_supression(predictions, args.nms_window)

        # Get detections
        detections = []
        video_pred = predictions[0]  # Only one video

        for i, event in enumerate(video_pred['events']):
            frame = event['frame']
            timestamp = frame_to_timestamp(frame, inference_fps)
            center_time = frame / inference_fps

            # Generate clip filename
            time_str = timestamp.replace(':', 'm').replace('.', 's')[:8]
            clip_filename = f"foul_{i+1:03d}_{time_str}_{event['label']}_{event['score']:.2f}.mp4"

            detection = {
                'id': i + 1,
                'frame': frame,
                'timestamp': timestamp,
                'foul_type': event['label'],
                'confidence': round(event['score'], 4),
                'clip_file': clip_filename
            }
            detections.append(detection)

            # Extract clip
            clip_path = clips_dir / clip_filename
            print(f"  Extracting clip {i+1}: {event['label']} @ {timestamp} (conf: {event['score']:.2f})")
            extract_clip(args.video, str(clip_path), center_time, args.clip_duration)

        # Create summary
        foul_types = Counter(d['foul_type'] for d in detections)

        results = {
            'video': os.path.basename(args.video),
            'duration_seconds': round(video_duration, 2),
            'threshold': args.threshold,
            'nms_window': args.nms_window,
            'model_checkpoint': os.path.basename(args.checkpoint),
            'processed_at': datetime.now().isoformat(),
            'detections': detections,
            'summary': {
                'total_detections': len(detections),
                'by_type': dict(foul_types)
            }
        }

        # Save results
        results_file = output_dir / 'detections.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total fouls detected: {len(detections)}")
        for foul_type, count in foul_types.most_common():
            print(f"  {foul_type}: {count}")
        print()
        print(f"Results saved to: {results_file}")
        print(f"Clips saved to:   {clips_dir}")
        print("=" * 60)

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run foul detection on full game video",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--video', required=True,
                        help='Path to input video file')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for output (detections.json + clips/)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Confidence threshold for detections (default: 0.15)')
    parser.add_argument('--clip-duration', type=float, default=10.0,
                        help='Duration of extracted clips in seconds (default: 10)')
    parser.add_argument('--nms-window', type=int, default=12,
                        help='NMS window in frames (default: 12 = 3 sec at 4fps)')
    parser.add_argument('--game-id', type=str, default=None,
                        help='Optional game identifier for output naming')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint directory not found: {args.checkpoint}")
        sys.exit(1)

    run_inference(args)


if __name__ == '__main__':
    main()
