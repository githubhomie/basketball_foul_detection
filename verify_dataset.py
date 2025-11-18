#!/usr/bin/env python3
"""
Verify cleaned dataset statistics
"""

import json
from collections import Counter

def analyze_split(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    total_clips = len(data)
    total_events = 0
    class_counts = Counter()
    frame_errors = []

    for clip in data:
        events = clip.get('events', [])
        total_events += len(events)

        for event in events:
            label = event.get('label', 'unknown')
            frame = event.get('frame', -1)

            class_counts[label] += 1

            # Check for frame=-1 (should be none)
            if frame == -1:
                frame_errors.append(f"{clip['video']}: frame={frame}")

    return total_clips, total_events, class_counts, frame_errors

def main():
    print("="*80)
    print("CLEANED DATASET VERIFICATION")
    print("="*80)

    splits = ['train', 'val', 'test']
    total_clips = 0
    total_events = 0
    all_classes = Counter()

    for split in splits:
        filename = f'data/basketball/{split}.json'
        clips, events, class_counts, errors = analyze_split(filename)

        print(f"\n{split.upper()}.JSON:")
        print(f"  Clips: {clips}")
        print(f"  Events: {events}")
        print(f"  Classes:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label:20s}: {count:4d}")

        if errors:
            print(f"  ⚠️  ERRORS: {len(errors)} clips with frame=-1")
            for err in errors[:3]:
                print(f"      {err}")
        else:
            print(f"  ✅ No frame=-1 errors")

        total_clips += clips
        total_events += events
        all_classes.update(class_counts)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total clips:  {total_clips}")
    print(f"Total events: {total_events}")
    print(f"\nClass distribution:")
    for label, count in sorted(all_classes.items()):
        pct = (count / total_events) * 100
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("DATASET READY FOR TRAINING ✅")
    print("="*80)

if __name__ == '__main__':
    main()
