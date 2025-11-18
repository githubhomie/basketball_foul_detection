# annotation_tool/utils/s3_loader.py
import boto3
import os
from PIL import Image
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
# Try multiple paths to find .env
load_dotenv("../../.env")  # From annotation_tool/utils -> parent/parent
load_dotenv("../.env")      # From annotation_tool -> parent
load_dotenv(".env")         # Current directory

@st.cache_resource
def get_s3_client():
    """Get cached S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

@st.cache_data(ttl=300, max_entries=100)
def load_frame_from_s3(s3_url):
    """
    Load a single frame from S3 URL
    Returns PIL Image

    Cache limited to 100 entries with 5 minute TTL to prevent memory buildup
    """
    try:
        # Parse S3 URL
        # Format: https://bucket.s3.region.amazonaws.com/key
        parts = s3_url.replace('https://', '').split('/')
        bucket = parts[0].split('.')[0]
        key = '/'.join(parts[1:])

        # Download from S3
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()

        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        return image

    except Exception as e:
        st.error(f"Error loading frame: {e}")
        return None

def get_clip_frames(clip_df, frame_indices=None):
    """
    Load specific frames for a clip

    Args:
        clip_df: DataFrame rows for a single clip (30 frames)
        frame_indices: List of frame indices to load (e.g., [9, 11, 13, ...])
                      If None, loads all frames

    Returns:
        List of (frame_index, PIL Image) tuples
    """
    if frame_indices is None:
        frame_indices = sorted(clip_df['frame_index'].unique())

    frames = []
    for idx in frame_indices:
        frame_row = clip_df[clip_df['frame_index'] == idx]
        if not frame_row.empty:
            s3_url = frame_row.iloc[0]['s3_url']
            image = load_frame_from_s3(s3_url)
            if image:
                frames.append((idx, image))

    return frames

def get_candidate_frames(clip_df, num_candidates=8):
    """
    Get 8 candidate frames around expected foul moment

    Args:
        clip_df: DataFrame rows for a single clip (30 frames)
        num_candidates: Number of frames to suggest (default 8)

    Returns:
        List of frame indices
    """
    total_frames = len(clip_df)
    center = total_frames // 2  # Frame 15 for 30 frames

    # Select 8 frames spanning ±4 frames around center, skip every other
    # [9, 11, 13, 15, 17, 19, 21, 23]
    half_span = num_candidates // 2
    indices = []
    for i in range(-half_span, half_span):
        frame_idx = center + (i * 2)
        if 0 <= frame_idx < total_frames:
            indices.append(frame_idx)

    return sorted(indices)

def get_next_candidate_frames(clip_df, previous_indices, num_candidates=8):
    """
    Get next set of candidate frames (when user presses N)

    Args:
        clip_df: DataFrame rows for a single clip
        previous_indices: Previously shown frame indices
        num_candidates: Number of frames to show

    Returns:
        List of frame indices
    """
    total_frames = len(clip_df)
    all_indices = set(range(total_frames))
    shown_indices = set(previous_indices)
    remaining_indices = sorted(all_indices - shown_indices)

    # Return next batch
    return remaining_indices[:num_candidates]
