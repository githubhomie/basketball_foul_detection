#!/usr/bin/env python3
"""
Prepare NBA foul dataset for E2E-Spot training.

This script converts the collected foul data into the format expected by E2E-Spot:
1. Loads metadata CSVs and annotation JSONs
2. Joins foul clips with their annotations
3. Converts to E2E-Spot JSON format
4. Creates stratified train/val/test splits
5. Outputs train.json, val.json, test.json, and class.txt
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
METADATA_DIR = Path('data/metadata')
ANNOTATIONS_DIR = Path('data/annotations')
OUTPUT_DIR = Path('data/e2e_format')

# Metadata files
FOUL_METADATA_CSV = METADATA_DIR / 'nba_fouls_2023-24_1138clips_20251015_090917.csv'
NON_FOUL_METADATA_CSV = METADATA_DIR / 'non_fouls_2023-24_1000clips_20251114_162731.csv'

# Class list (order matters - will be class indices 1-5, with 0=background)
FOUL_CLASSES = [
    'shooting_foul',
    'personal_foul',
    'loose_ball',
    'offensive_foul',
    'charging'
]


def load_annotations():
    """Load all annotation JSONs into a dictionary keyed by (game_id, event_num)."""
    print("Loading annotations...")
    annotations = {}

    for ann_file in ANNOTATIONS_DIR.glob('*_annotation.json'):
        with open(ann_file) as f:
            data = json.load(f)
            key = (data['game_id'], data['event_num'])
            annotations[key] = data['foul_frame']

    print(f"  Loaded {len(annotations)} annotations")
    return annotations


def load_foul_clips(annotations):
    """Load foul clip metadata and join with annotations."""
    print("\nLoading foul clips...")
    df = pd.read_csv(FOUL_METADATA_CSV)

    # Get unique clips (one row per clip, not per frame)
    clips = df.groupby(['game_id', 'event_num']).first().reset_index()
    print(f"  Found {len(clips)} foul clips in metadata")

    # Join with annotations
    clips['video_id'] = clips.apply(
        lambda row: f"{row['game_id']:010d}_{row['event_num']}", axis=1)
    clips['foul_frame'] = clips.apply(
        lambda row: annotations.get((f"{row['game_id']:010d}", row['event_num']), None),
        axis=1)

    # Filter to only annotated clips
    annotated = clips[clips['foul_frame'].notna()].copy()
    print(f"  {len(annotated)} clips have annotations")

    # Create E2E-Spot format for fouls
    foul_data = []
    for _, row in annotated.iterrows():
        entry = {
            'video': row['video_id'],
            'num_frames': 30,
            'num_events': 1,
            'events': [{
                'frame': int(row['foul_frame']),
                'label': row['foul_type']
            }],
            'fps': 4,
            'width': 1920,
            'height': 1080,
            'class': row['foul_type']  # For stratified splitting
        }
        foul_data.append(entry)

    print(f"  Created {len(foul_data)} foul entries")

    # Print class distribution
    print("\n  Foul type distribution:")
    for foul_type in FOUL_CLASSES:
        count = sum(1 for e in foul_data if e['class'] == foul_type)
        print(f"    {foul_type:20s}: {count:4d}")

    return foul_data


def load_non_foul_clips():
    """Load non-foul clip metadata."""
    print("\nLoading non-foul clips...")
    df = pd.read_csv(NON_FOUL_METADATA_CSV)

    # Get unique clips
    clips = df.groupby(['game_id', 'event_num']).first().reset_index()
    print(f"  Found {len(clips)} non-foul clips")

    # Create E2E-Spot format for non-fouls (no events)
    non_foul_data = []
    for _, row in clips.iterrows():
        video_id = f"{row['game_id']:010d}_{row['event_num']}"
        entry = {
            'video': video_id,
            'num_frames': 30,
            'num_events': 0,
            'events': [],  # No foul events
            'fps': 4,
            'width': 1920,
            'height': 1080,
            'class': 'non_foul'  # For splitting
        }
        non_foul_data.append(entry)

    print(f"  Created {len(non_foul_data)} non-foul entries")
    return non_foul_data


def stratified_split(data, train_ratio, val_ratio, test_ratio, random_seed):
    """Create stratified train/val/test splits."""
    print("\nCreating stratified splits...")

    # Separate by class for stratified splitting
    class_data = {}
    for entry in data:
        cls = entry['class']
        if cls not in class_data:
            class_data[cls] = []
        class_data[cls].append(entry)

    train_data = []
    val_data = []
    test_data = []

    # Split each class separately
    np.random.seed(random_seed)
    for cls, entries in class_data.items():
        n = len(entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Shuffle
        indices = np.random.permutation(n)

        # Split
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        train_data.extend([entries[i] for i in train_idx])
        val_data.extend([entries[i] for i in val_idx])
        test_data.extend([entries[i] for i in test_idx])

        print(f"  {cls:20s}: train={len(train_idx):4d}, val={len(val_idx):3d}, test={len(test_idx):3d}")

    print(f"\n  TOTAL: train={len(train_data):4d}, val={len(val_data):3d}, test={len(test_data):3d}")

    return train_data, val_data, test_data


def save_split(data, filename):
    """Save a data split to JSON file."""
    # Remove 'class' field (only used for splitting)
    for entry in data:
        entry.pop('class', None)

    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {len(data)} entries to {output_path}")


def save_class_file():
    """Save class names to class.txt."""
    output_path = OUTPUT_DIR / 'class.txt'
    with open(output_path, 'w') as f:
        for cls in FOUL_CLASSES:
            f.write(f"{cls}\n")

    print(f"  Saved {len(FOUL_CLASSES)} classes to {output_path}")


def main():
    print("="*80)
    print("NBA FOUL DATASET PREPARATION FOR E2E-SPOT")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    annotations = load_annotations()
    foul_data = load_foul_clips(annotations)
    non_foul_data = load_non_foul_clips()

    # Combine foul and non-foul data
    all_data = foul_data + non_foul_data
    print(f"\nTotal clips: {len(all_data)}")

    # Create splits
    train_data, val_data, test_data = stratified_split(
        all_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    # Save splits
    print("\nSaving outputs...")
    save_split(train_data, 'train.json')
    save_split(val_data, 'val.json')
    save_split(test_data, 'test.json')
    save_class_file()

    print("\n" + "="*80)
    print("✓ Dataset preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: train.json, val.json, test.json, class.txt")
    print("="*80)


if __name__ == '__main__':
    main()
