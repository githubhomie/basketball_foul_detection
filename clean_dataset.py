#!/usr/bin/env python3
"""
Clean dataset JSON files by removing clips with frame=-1 (missing annotations)
"""

import json
import os

# Video IDs to remove (clips with frame=-1)
TRAIN_REMOVE = [
    '0022301064_244', '0022301064_282', '0022301089_423', '0022301091_435',
    '0022301095_299', '0022301099_283', '0022301101_291', '0022301102_420',
    '0022301106_303', '0022301111_357', '0022301113_426', '0022301116_520',
    '0022301120_541', '0022301128_527', '0022301131_282', '0022301133_411'
]

VAL_REMOVE = [
    '0022301150_427', '0022301167_589', '0022301176_156'
]

TEST_REMOVE = [
    '0022301182_416', '0022301188_437', '0022301169_338',
    '0022301170_382', '0022301181_387'
]

def clean_json_file(input_path, output_path, video_ids_to_remove):
    """Remove clips from JSON file by video ID"""
    print(f"Processing {input_path}...")

    # Read the entire JSON array
    with open(input_path, 'r') as f:
        data = json.load(f)

    kept_clips = []
    removed_clips = []

    for clip in data:
        video_id = clip.get('video', '')

        # Check if this clip should be removed
        if video_id in video_ids_to_remove:
            removed_clips.append(video_id)
        else:
            # Verify it doesn't have frame=-1
            has_invalid_frame = False
            for event in clip.get('events', []):
                if event.get('frame', 0) == -1:
                    has_invalid_frame = True
                    removed_clips.append(f"{video_id} (frame=-1 found!)")
                    break

            if not has_invalid_frame:
                kept_clips.append(clip)

    # Write cleaned file (as JSON array with nice formatting)
    with open(output_path, 'w') as f:
        json.dump(kept_clips, f, indent=2)

    print(f"  Kept: {len(kept_clips)} clips")
    print(f"  Removed: {len(removed_clips)} clips")
    if removed_clips:
        print(f"  Removed IDs: {removed_clips[:5]}{'...' if len(removed_clips) > 5 else ''}")

    return len(kept_clips), len(removed_clips)

def main():
    base_dir = 'data/basketball'

    # Clean train.json
    print("\n=== Cleaning train.json ===")
    train_kept, train_removed = clean_json_file(
        os.path.join(base_dir, 'train.json'),
        os.path.join(base_dir, 'train.json.clean'),
        set(TRAIN_REMOVE)
    )

    # Clean val.json
    print("\n=== Cleaning val.json ===")
    val_kept, val_removed = clean_json_file(
        os.path.join(base_dir, 'val.json'),
        os.path.join(base_dir, 'val.json.clean'),
        set(VAL_REMOVE)
    )

    # Clean test.json
    print("\n=== Cleaning test.json ===")
    test_kept, test_removed = clean_json_file(
        os.path.join(base_dir, 'test.json'),
        os.path.join(base_dir, 'test.json.clean'),
        set(TEST_REMOVE)
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train: {train_kept} clips (removed {train_removed})")
    print(f"Val:   {val_kept} clips (removed {val_removed})")
    print(f"Test:  {test_kept} clips (removed {test_removed})")
    print(f"Total: {train_kept + val_kept + test_kept} clips (removed {train_removed + val_removed + test_removed})")
    print()
    print("Cleaned files created:")
    print("  - data/basketball/train.json.clean")
    print("  - data/basketball/val.json.clean")
    print("  - data/basketball/test.json.clean")
    print()
    print("To apply changes, run:")
    print("  mv data/basketball/train.json data/basketball/train.json.backup")
    print("  mv data/basketball/val.json data/basketball/val.json.backup")
    print("  mv data/basketball/test.json data/basketball/test.json.backup")
    print("  mv data/basketball/train.json.clean data/basketball/train.json")
    print("  mv data/basketball/val.json.clean data/basketball/val.json")
    print("  mv data/basketball/test.json.clean data/basketball/test.json")

if __name__ == '__main__':
    main()
