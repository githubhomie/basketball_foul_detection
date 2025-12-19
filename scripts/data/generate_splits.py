#!/usr/bin/env python3
"""
Generate train/val/test splits from S3 annotations.

CRITICAL: This script MUST produce identical splits to TRAIN4-2.ipynb
to ensure checkpoint compatibility with Google Drive checkpoints.

Logic replicates Cell 6 of TRAIN4-2.ipynb exactly:
- Downloads annotations from S3
- Filters fouls: foul_frame >= 0, valid foul types only
- Creates stratified 70/15/15 splits with np.random.seed(42)

Usage:
    python scripts/data/generate_splits.py
    python scripts/data/generate_splits.py --output-dir data/basketball/
    python scripts/data/generate_splits.py --dry-run  # Preview without writing
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

# S3 bucket and paths (DO NOT CHANGE - must match TRAIN4-2.ipynb)
BUCKET = "nba-foul-dataset-oh"
FOULS_CSV = "metadata/nba_fouls_multi-season_1863clips_20251122_002439.csv"
NON_FOULS_CSV = "metadata/non_fouls/non_fouls_2023-24_1000clips_20251114_162731.csv"
ANNOTATIONS_PREFIX = "annotations/"

# Valid foul types (must match TRAIN4-2.ipynb)
VALID_FOUL_TYPES = [
    'shooting_foul',
    'personal_foul',
    'loose_ball',
    'offensive_foul',
    'charging'
]


def download_s3_file(s3, bucket, key, local_path):
    """Download a file from S3."""
    print(f"  Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)


def load_foul_metadata(s3, tmpdir):
    """Load foul metadata CSV from S3."""
    local_path = os.path.join(tmpdir, 'fouls.csv')
    download_s3_file(s3, BUCKET, FOULS_CSV, local_path)
    df = pd.read_csv(local_path)

    # Create lookup: (game_id_str, event_num) -> foul_type
    # game_id stored as STRING to match annotation JSON format
    foul_types = {}
    for _, row in df.iterrows():
        # Keep game_id as string (matches annotation format)
        game_id = str(row['game_id']).zfill(10)
        event_num = int(row['event_num'])
        key = (game_id, event_num)
        foul_types[key] = row['foul_type']

    return foul_types


def load_annotations(s3, tmpdir, foul_types):
    """Load annotation JSONs from S3 and filter valid fouls."""
    # List all annotation files
    paginator = s3.get_paginator('list_objects_v2')
    annotations = []

    print("  Listing annotation files...")
    ann_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=ANNOTATIONS_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                ann_keys.append(obj['Key'])

    print(f"  Found {len(ann_keys)} annotation files")

    # Download and process each annotation
    valid_count = 0
    invalid_count = 0

    for key in ann_keys:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        ann = json.loads(response['Body'].read().decode('utf-8'))

        # Normalize game_id to 10-digit zero-padded string
        game_id = str(ann['game_id']).zfill(10)
        event_num = int(ann['event_num'])
        foul_frame = ann.get('foul_frame', -1)

        # Lookup foul type using normalized key
        lookup_key = (game_id, event_num)
        foul_type = foul_types.get(lookup_key, 'unknown')

        # Filter: foul_frame >= 0 and valid foul type
        if foul_frame >= 0 and foul_type in VALID_FOUL_TYPES:
            video_id = f"{game_id}_{event_num}"
            annotations.append({
                'video': video_id,
                'foul_frame': foul_frame,
                'foul_type': foul_type
            })
            valid_count += 1
        else:
            invalid_count += 1

    print(f"  Valid foul annotations: {valid_count}")
    print(f"  Filtered out: {invalid_count}")

    return annotations


def load_non_fouls(s3, tmpdir):
    """Load non-foul clips from S3 CSV.

    NOTE: The CSV has one row PER FRAME (30 frames × N clips),
    so we need to deduplicate by (game_id, event_num).
    """
    local_path = os.path.join(tmpdir, 'non_fouls.csv')
    download_s3_file(s3, BUCKET, NON_FOULS_CSV, local_path)
    df = pd.read_csv(local_path)

    # Deduplicate: CSV has one row per frame, we want unique clips
    seen_clips = set()
    non_fouls = []
    for _, row in df.iterrows():
        game_id = str(row['game_id']).zfill(10)
        event_num = int(row['event_num'])
        video_id = f"{game_id}_{event_num}"

        if video_id not in seen_clips:
            seen_clips.add(video_id)
            non_fouls.append({'video': video_id})

    print(f"  Non-foul clips: {len(non_fouls)} (deduplicated from {len(df)} rows)")
    return non_fouls


def create_e2e_spot_entries(annotations, non_fouls):
    """Convert to E2E-Spot format entries."""
    # Foul entries
    foul_entries = []
    for ann in annotations:
        foul_entries.append({
            'video': ann['video'],
            'num_frames': 30,
            'num_events': 1,
            'events': [{'frame': ann['foul_frame'], 'label': ann['foul_type']}],
            'fps': 4,
            'width': 1920,
            'height': 1080,
            'class': ann['foul_type']  # For stratified splitting
        })

    # Non-foul entries
    nonfoul_entries = []
    for nf in non_fouls:
        nonfoul_entries.append({
            'video': nf['video'],
            'num_frames': 30,
            'num_events': 0,
            'events': [],
            'fps': 4,
            'width': 1920,
            'height': 1080,
            'class': 'non_foul'  # For stratified splitting
        })

    return foul_entries, nonfoul_entries


def stratified_split(all_entries):
    """
    Create stratified 70/15/15 train/val/test split.

    CRITICAL: Must use np.random.seed(42) to match TRAIN4-2.ipynb exactly.
    """
    np.random.seed(42)  # MUST MATCH TRAIN4-2.ipynb

    # Group by class
    class_data = {}
    for entry in all_entries:
        cls = entry['class']
        if cls not in class_data:
            class_data[cls] = []
        class_data[cls].append(entry)

    train_data, val_data, test_data = [], [], []

    # Split each class 70/15/15
    for cls, entries in class_data.items():
        n = len(entries)
        indices = np.random.permutation(n)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_data.extend([entries[i] for i in indices[:n_train]])
        val_data.extend([entries[i] for i in indices[n_train:n_train + n_val]])
        test_data.extend([entries[i] for i in indices[n_train + n_val:]])

    return train_data, val_data, test_data


def remove_class_field(data):
    """Remove the 'class' field used for stratification before saving."""
    for entry in data:
        if 'class' in entry:
            del entry['class']
    return data


def print_stats(name, data):
    """Print statistics for a split."""
    foul_count = sum(1 for d in data if d['num_events'] > 0)
    nonfoul_count = len(data) - foul_count

    # Count by class
    class_counts = Counter()
    for d in data:
        if d['num_events'] > 0:
            class_counts[d['events'][0]['label']] += 1
        else:
            class_counts['non_foul'] += 1

    print(f"\n{name}: {len(data)} clips ({foul_count} fouls, {nonfoul_count} non-fouls)")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits from S3 annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data/basketball",
        help="Output directory for train.json, val.json, test.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATE DATASET SPLITS")
    print("=" * 60)
    print(f"S3 Bucket: {BUCKET}")
    print(f"Output: {args.output_dir}")
    print()

    # Initialize S3 client
    s3 = boto3.client('s3')

    with tempfile.TemporaryDirectory() as tmpdir:
        # Load data from S3
        print("Loading foul metadata...")
        foul_types = load_foul_metadata(s3, tmpdir)

        print("\nLoading annotations...")
        annotations = load_annotations(s3, tmpdir, foul_types)

        print("\nLoading non-fouls...")
        non_fouls = load_non_fouls(s3, tmpdir)

    # Create E2E-Spot format entries
    print("\nCreating E2E-Spot format entries...")
    foul_entries, nonfoul_entries = create_e2e_spot_entries(annotations, non_fouls)
    all_entries = foul_entries + nonfoul_entries
    print(f"  Total entries: {len(all_entries)}")

    # Stratified split
    print("\nCreating stratified 70/15/15 splits (seed=42)...")
    train_data, val_data, test_data = stratified_split(all_entries)

    # Print statistics
    print_stats("Train", train_data)
    print_stats("Val", val_data)
    print_stats("Test", test_data)

    total = len(train_data) + len(val_data) + len(test_data)
    print(f"\nTotal: {total} clips")

    if args.dry_run:
        print("\n[DRY RUN] No files written")
        return

    # Remove class field before saving
    train_data = remove_class_field(train_data)
    val_data = remove_class_field(val_data)
    test_data = remove_class_field(test_data)

    # Save files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_dir}/...")

    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  train.json: {len(train_data)} clips")

    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  val.json: {len(val_data)} clips")

    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  test.json: {len(test_data)} clips")

    print("\nDone!")


if __name__ == "__main__":
    main()
