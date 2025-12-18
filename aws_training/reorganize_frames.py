#!/usr/bin/env python3
"""
reorganize_frames.py - Convert S3 frame structure to training format

S3 structure:     {raw_dir}/2023-24/{game_id}/{game_id}_{event_id}_frame_{X}.jpg
Training format:  {output_dir}/{game_id}_{event_id}/{X:06d}.jpg

Usage:
    python aws_training/reorganize_frames.py --raw /data/frames_raw --output /data/frames
"""

import os
import re
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def find_all_frames(raw_dir):
    """Find all frames and group by clip_id."""
    clips = defaultdict(list)
    pattern = re.compile(r'(\d+)_(\d+)_frame_(\d+)\.jpg')

    # Walk through season/game directories
    for season_folder in os.listdir(raw_dir):
        season_path = os.path.join(raw_dir, season_folder)
        if not os.path.isdir(season_path):
            continue

        for game_folder in os.listdir(season_path):
            game_path = os.path.join(season_path, game_folder)
            if not os.path.isdir(game_path):
                continue

            for filename in os.listdir(game_path):
                match = pattern.match(filename)
                if match:
                    game_id, event_id, frame_idx = match.groups()
                    clip_id = f"{game_id}_{event_id}"
                    src_path = os.path.join(game_path, filename)
                    clips[clip_id].append((src_path, int(frame_idx)))

    return clips


def copy_clip(clip_id, frames, output_dir):
    """Copy all frames for one clip to output directory."""
    clip_dir = os.path.join(output_dir, clip_id)
    os.makedirs(clip_dir, exist_ok=True)

    for src_path, frame_idx in frames:
        dst_path = os.path.join(clip_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    return clip_id, len(frames)


def main():
    parser = argparse.ArgumentParser(description='Reorganize frames for training')
    parser.add_argument('--raw', type=str, default='/data/frames_raw',
                        help='Directory with raw S3 frame structure')
    parser.add_argument('--output', type=str, default='/data/frames',
                        help='Output directory for training format')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--delete-raw', action='store_true',
                        help='Delete raw directory after reorganization')
    args = parser.parse_args()

    print("=" * 50)
    print("Frame Reorganization")
    print("=" * 50)
    print(f"Raw directory:    {args.raw}")
    print(f"Output directory: {args.output}")
    print(f"Workers:          {args.workers}")
    print()

    # Check raw directory
    if not os.path.exists(args.raw):
        print(f"Error: Raw directory not found: {args.raw}")
        return 1

    # Find all frames
    print("Scanning for frames...")
    clips = find_all_frames(args.raw)
    total_frames = sum(len(frames) for frames in clips.values())
    print(f"Found {len(clips)} clips with {total_frames} frames")
    print()

    if len(clips) == 0:
        print("No frames found! Check directory structure.")
        return 1

    # Check existing output
    os.makedirs(args.output, exist_ok=True)
    existing = set(os.listdir(args.output)) if os.path.exists(args.output) else set()
    clips_to_copy = {k: v for k, v in clips.items() if k not in existing}

    if len(clips_to_copy) == 0:
        print("All clips already reorganized!")
        return 0

    print(f"Copying {len(clips_to_copy)} clips ({len(clips) - len(clips_to_copy)} already exist)...")

    # Copy in parallel
    completed = 0
    total_copied = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(copy_clip, clip_id, frames, args.output): clip_id
            for clip_id, frames in clips_to_copy.items()
        }

        for future in as_completed(futures):
            clip_id, n_frames = future.result()
            completed += 1
            total_copied += n_frames

            if completed % 100 == 0 or completed == len(clips_to_copy):
                print(f"  Progress: {completed}/{len(clips_to_copy)} clips "
                      f"({total_copied} frames copied)")

    print()
    print("Reorganization complete!")
    print(f"  Total clips: {len(clips)}")
    print(f"  Total frames: {total_frames}")

    # Optionally delete raw directory
    if args.delete_raw:
        print()
        print(f"Deleting raw directory: {args.raw}")
        shutil.rmtree(args.raw)
        print("Raw directory deleted (saved disk space)")

    return 0


if __name__ == '__main__':
    exit(main())
