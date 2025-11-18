# annotation_tool/utils/annotation_io.py
import json
import os
from datetime import datetime
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")
load_dotenv("../.env")
load_dotenv(".env")

# S3 configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'nba-foul-dataset-oh')
S3_ANNOTATIONS_PREFIX = 'annotations/'

# Local fallback for development
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")

@st.cache_resource
def get_s3_client():
    """Get cached S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-2')
    )

def get_s3_key(game_id, event_num):
    """Get S3 key for annotation"""
    return f"{S3_ANNOTATIONS_PREFIX}{game_id}_{event_num}_annotation.json"

def ensure_annotations_dir():
    """Create local annotations directory if it doesn't exist (for backup)"""
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def get_annotation_path(game_id, event_num):
    """Get local file path for annotation (backup only)"""
    ensure_annotations_dir()
    return os.path.join(ANNOTATIONS_DIR, f"{game_id}_{event_num}_annotation.json")

def save_annotation(game_id, event_num, foul_frame, annotator="default", notes=""):
    """
    Save annotation to S3 (and local backup)

    Args:
        game_id: Game ID
        event_num: Event number
        foul_frame: Frame index where foul occurred
        annotator: Name of annotator
        notes: Optional notes
    """
    annotation = {
        "game_id": str(game_id),
        "event_num": int(event_num),
        "foul_frame": int(foul_frame) if foul_frame is not None else None,
        "annotator": annotator,
        "timestamp": datetime.now().isoformat(),
        "notes": notes
    }

    annotation_json = json.dumps(annotation, indent=2)

    try:
        # Save to S3 (primary)
        s3_client = get_s3_client()
        s3_key = get_s3_key(game_id, event_num)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=annotation_json,
            ContentType='application/json'
        )

        # Also save local backup for development
        try:
            file_path = get_annotation_path(game_id, event_num)
            with open(file_path, 'w') as f:
                f.write(annotation_json)
        except Exception:
            pass  # Local backup is optional

        return s3_key

    except Exception as e:
        st.error(f"Failed to save annotation to S3: {e}")
        # Fallback to local only
        file_path = get_annotation_path(game_id, event_num)
        with open(file_path, 'w') as f:
            f.write(annotation_json)
        return file_path

def load_annotation(game_id, event_num):
    """
    Load annotation from S3 (with local fallback)

    Returns:
        dict or None if annotation doesn't exist
    """
    try:
        # Try S3 first
        s3_client = get_s3_client()
        s3_key = get_s3_key(game_id, event_num)
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        annotation_data = response['Body'].read().decode('utf-8')
        return json.loads(annotation_data)

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # File doesn't exist in S3, try local fallback
            file_path = get_annotation_path(game_id, event_num)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
        else:
            # Other S3 error, try local fallback
            file_path = get_annotation_path(game_id, event_num)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
    except Exception:
        # Any other error, try local fallback
        file_path = get_annotation_path(game_id, event_num)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

def get_annotated_clips():
    """
    Get set of (game_id, event_num) tuples that have been annotated from S3

    Returns:
        set of tuples
    """
    annotated = set()

    try:
        # List all annotations from S3
        s3_client = get_s3_client()
        paginator = s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_ANNOTATIONS_PREFIX):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]  # Get filename from key

                if filename.endswith('_annotation.json'):
                    # Parse filename: {game_id}_{event_num}_annotation.json
                    parts = filename.replace('_annotation.json', '').split('_')
                    if len(parts) >= 2:
                        game_id = parts[0]
                        event_num = int(parts[1])
                        annotated.add((game_id, event_num))

    except Exception as e:
        # Fallback to local directory if S3 fails
        try:
            ensure_annotations_dir()
            for filename in os.listdir(ANNOTATIONS_DIR):
                if filename.endswith('_annotation.json'):
                    parts = filename.replace('_annotation.json', '').split('_')
                    if len(parts) >= 2:
                        game_id = parts[0]
                        event_num = int(parts[1])
                        annotated.add((game_id, event_num))
        except Exception:
            pass

    return annotated

def export_annotations_to_csv(csv_path, output_path=None):
    """
    Export all annotations back to CSV with additional columns

    Args:
        csv_path: Path to original CSV
        output_path: Path to save annotated CSV (default: adds _annotated suffix)
    """
    # Load original CSV
    df = pd.read_csv(csv_path, dtype={'game_id': str})

    # Add annotation columns
    df['foul_frame'] = None
    df['annotated'] = False
    df['annotator'] = None
    df['annotation_date'] = None

    # Load all annotations
    annotated_clips = get_annotated_clips()

    for game_id, event_num in annotated_clips:
        annotation = load_annotation(game_id, event_num)
        if annotation and annotation['foul_frame'] is not None:
            # Update all frames for this clip
            mask = (df['game_id'] == game_id) & (df['event_num'] == event_num)
            df.loc[mask, 'foul_frame'] = annotation['foul_frame']
            df.loc[mask, 'annotated'] = True
            df.loc[mask, 'annotator'] = annotation['annotator']
            df.loc[mask, 'annotation_date'] = annotation['timestamp']

    # Save annotated CSV
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_annotated{ext}"

    df.to_csv(output_path, index=False)
    return output_path

def get_annotation_stats(csv_path=None):
    """
    Get annotation statistics

    Returns:
        dict with stats
    """
    annotated_clips = get_annotated_clips()

    stats = {
        'total_annotated': len(annotated_clips),
        'annotated_clips': annotated_clips
    }

    if csv_path:
        df = pd.read_csv(csv_path, dtype={'game_id': str})
        total_clips = len(df.groupby(['game_id', 'event_num']))
        stats['total_clips'] = total_clips
        stats['percent_complete'] = (len(annotated_clips) / total_clips * 100) if total_clips > 0 else 0

    return stats
