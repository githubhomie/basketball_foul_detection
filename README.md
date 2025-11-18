# Basketball Foul Detection

Automated detection and classification of NBA fouls using temporal action spotting.

Built on the [E2E-Spot architecture](https://jhong93.github.io/projects/spot.html) (ECCV 2022) - adapted for basketball foul detection from broadcast video.

## What This Does

Detects and classifies fouls in NBA game footage:
- **5 foul types:** shooting, personal, loose ball, charging, offensive
- **Frame-level precision:** Identifies the exact moment a foul occurs
- **End-to-end learning:** Direct from video frames to foul detection

**Dataset:** 1,808 NBA clips (2023-24 season), 808 annotated foul events

## Full Pipeline

This repository contains the complete workflow from data collection to model training:

1. **Data Collection** ([`data_pipeline/`](data_pipeline/)) - Collect foul clips from NBA API
2. **Annotation** ([`data_pipeline/annotation_tool/`](data_pipeline/annotation_tool/)) - Frame-level foul labeling
3. **Dataset Preparation** ([`data_pipeline/prepare_for_training.py`](data_pipeline/prepare_for_training.py)) - Convert to E2E-Spot format
4. **Training** ([`train_basketball.sh`](train_basketball.sh)) - Train foul detection model

See [`data_pipeline/README.md`](data_pipeline/README.md) for detailed documentation on data collection and annotation.

## Quick Start

### Training
```bash
./train_basketball.sh
```

Or manually:
```bash
python3 train_e2e.py basketball /path/to/frames \
    -s ./checkpoints \
    -m rny002_gsm \
    -t gru \
    --clip_len 30 \
    --batch_size 8 \
    --num_epochs 50
```

### Evaluation
```bash
python3 eval.py -s test checkpoints/basketball_YYYYMMDD/
```

### Inference
```bash
python3 test_e2e.py checkpoints/basketball_best/ /path/to/frames -s test --save
```

## Data Format

### Dataset Structure
```
data/basketball/
├── train.json      # 1,265 clips
├── val.json        # 269 clips
├── test.json       # 274 clips
└── class.txt       # 5 foul types
```

### JSON Format
```json
[
  {
    "video": "0022301194_219",
    "num_frames": 30,
    "num_events": 1,
    "events": [{"frame": 10, "label": "charging"}],
    "fps": 4,
    "width": 1920,
    "height": 1080
  }
]
```

### Frame Directory
```
frames/
├── 0022301194_219/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ... (30 frames)
└── ...
```

## Model Architecture

- **Backbone:** RegNet-Y 200MF with Gated Shift Module (GSM)
- **Temporal:** Bidirectional GRU
- **Input:** 30-frame clips at 224×224
- **Output:** Per-frame foul predictions (6 classes including background)

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch 1.11+, CUDA 11.x

---

**Original E2E-Spot Paper:**
*Spotting Temporally Precise, Fine-Grained Events in Video* (ECCV 2022)
James Hong, Haotian Zhang, Michael Gharbi, Matthew Fisher, Kayvon Fatahalian
[Project Page](https://jhong93.github.io/projects/spot.html)
