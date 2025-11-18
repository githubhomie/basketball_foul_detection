# Data Collection & Annotation Pipeline

This directory contains the complete data collection and annotation workflow for creating the basketball foul detection dataset.

## Pipeline Overview

```
NBA API → Foul Clips → Annotation Tool → E2E-Spot Format → Training
```

## 1. Data Collection

### Foul Clip Collection
**Script:** `collect_data.py`

Collects foul clips from NBA games using the official NBA API:
- Fetches play-by-play data for 2023-24 season
- Identifies foul events (shooting, personal, loose ball, charging, offensive)
- Downloads 30-frame clips (7.5 seconds) around each foul
- Uploads frames to S3: `s3://nba-foul-dataset-oh/frames/`

**Usage:**
```bash
python collect_data.py --season 2023-24 --games 100
```

### Non-Foul Clip Collection
**Script:** `collect_non_fouls_cdn.py`

Collects negative examples (plays without fouls):
- Samples random plays from NBA games
- Filters out foul events
- Creates balanced dataset (1,000 non-foul clips)

**Charging Foul Collection:**
**Script:** `collect_charging.py`

Special collection for rare charging fouls (underrepresented in data).

### Output
- **Frames:** 54,240 images (1,832 clips × 30 frames)
- **Metadata:** CSV files with game IDs, event numbers, timestamps
- **Storage:** S3 bucket for cloud access

## 2. Frame-Level Annotation

### Annotation Tool
**Location:** `annotation_tool/`

Streamlit web app for annotating the exact frame where a foul occurs.

**Features:**
- Loads clips from S3 on-demand
- Video player with frame-by-frame navigation
- Mark exact foul frame with keyboard shortcuts
- Progress tracking (907/1,213 clips annotated = 74%)
- Saves annotations to S3

**Usage:**
```bash
cd annotation_tool
pip install -r requirements.txt
streamlit run app.py
```

**Annotation Format:**
```json
{
  "video_id": "0022301194_219",
  "foul_frame": 10,
  "foul_type": "charging",
  "annotator": "human",
  "timestamp": "2024-11-06T15:30:00Z"
}
```

### Annotation Tool Architecture
- **Frontend:** Streamlit (Python web framework)
- **Backend:** S3 streaming (no local storage needed)
- **Caching:** LRU cache for fast frame loading
- **Storage:** Annotations saved as JSON to S3

## 3. Dataset Preparation

### Convert to Training Format
**Script:** `prepare_for_training.py`

Converts raw annotations into E2E-Spot format for training:
- Reads metadata CSVs and annotation JSONs from S3
- Creates train/val/test splits (70/15/15 stratified by foul type)
- Generates `train.json`, `val.json`, `test.json`
- Removes clips with missing annotations (frame=-1)

**Usage:**
```bash
python prepare_for_training.py
```

**Output:** `../data/basketball/`
- `train.json` (1,265 clips)
- `val.json` (269 clips)
- `test.json` (274 clips)
- `class.txt` (5 foul types)

## Dataset Statistics

**Final Dataset:** 1,808 clips (808 foul events)

| Foul Type | Count | Percentage |
|-----------|-------|------------|
| Shooting foul | 218 | 27.0% |
| Personal foul | 241 | 29.8% |
| Loose ball | 182 | 22.5% |
| Charging | 84 | 10.4% |
| Offensive foul | 83 | 10.3% |
| **Non-foul** | **1,000** | **(negative examples)** |

**Frames:** 30 per clip at 4 FPS (7.5 second clips)
**Resolution:** 1920×1080 (broadcast quality)
**Source:** 2023-24 NBA season

## AWS Setup

**S3 Bucket:** `nba-foul-dataset-oh`
**Region:** `us-east-2`

```
s3://nba-foul-dataset-oh/
├── frames/              # 12.9 GB (foul clips)
├── frames/non_fouls/    # 9.7 GB (non-foul clips)
├── annotations/         # 907 JSON files
└── metadata/            # CSV files with clip metadata
```

**Note:** AWS credentials required (not included in repo for security)

## Requirements

```bash
# Data collection
pip install nba_api boto3 pandas opencv-python

# Annotation tool
pip install streamlit boto3 pillow pandas
```

## Next Steps

After running this pipeline:
1. Data collected → `s3://nba-foul-dataset-oh/frames/`
2. Annotations complete → `s3://nba-foul-dataset-oh/annotations/`
3. Dataset prepared → `../data/basketball/*.json`
4. Ready for training → `../train_basketball.sh`
