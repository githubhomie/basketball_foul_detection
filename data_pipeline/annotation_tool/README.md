# NBA Foul Annotator

Web app for annotating foul contact frames in NBA clips.

## Setup

```bash
cd annotation_tool
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Usage

1. Upload CSV in sidebar
2. Scrub through frames with slider
3. Click "SELECT [frame]" when you find foul contact
4. Click "SAVE [frame]" to save and move to next clip

## Features

- Frame scrubber (8-22 by default, toggle for all 30)
- Filter by foul type
- Hide completed annotations
- Progress tracking
- Flag unclear clips
- Export annotations to CSV
- Multi-user support (S3 backend)

## Annotations

Saved as JSON to:
- S3: `s3://nba-foul-dataset-oh/annotations/`
- Local backup: `../data/annotations/`

Format:
```json
{
  "game_id": "0022301192",
  "event_num": 37,
  "foul_frame": 13,
  "annotator": "your_name",
  "timestamp": "2025-10-15T11:15:34",
  "notes": ""
}
```

## Tips

- Take ~15 sec per clip
- Flag unclear clips instead of guessing
- Use category filter to split work among team
- Uncheck "Show annotated clips" to hide completed work
