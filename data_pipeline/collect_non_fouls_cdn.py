#!/usr/bin/env python3
"""
Non-Foul Data Collection Script - Uses CDN Endpoint
Collects non-foul gameplay clips (made shots, missed shots, rebounds, turnovers)
for negative examples in the foul detection dataset.

This script uses the working CDN endpoint instead of the broken stats.nba.com API.
Safe to run alongside foul annotation - uses separate directories and S3 paths.
"""

import os
import time
import logging
import requests
import cv2
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/non_foul_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Referer': 'https://stats.nba.com/',
    'Accept': 'application/json'
}

# Non-foul action types from CDN endpoint
# CDN uses different action types than stats.nba.com
NON_FOUL_ACTION_TYPES = {
    '2pt': 'made_shot',          # 2-point made
    '3pt': 'made_shot',          # 3-point made
    'miss': 'missed_shot',       # Any missed shot
    'rebound': 'rebound',        # Rebound
    'turnover': 'turnover',      # Turnover (includes steals)
    'steal': 'turnover',         # Steal is a type of turnover
}

# Collection target
NON_FOUL_TARGET = 1000  # Total non-foul clips to collect


class NonFoulCollector:
    def __init__(self, resume_from=None):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket = os.getenv('S3_BUCKET', 'nba-foul-dataset-oh')

        # Create separate directories for non-foul data
        os.makedirs('data/non_foul_videos', exist_ok=True)
        os.makedirs('data/non_foul_frames', exist_ok=True)
        os.makedirs('data/metadata', exist_ok=True)
        os.makedirs('data/checkpoints', exist_ok=True)

        self.dataset_records = []

        # Track collection by event type
        self.event_counts = {
            'made_shot': 0,
            'missed_shot': 0,
            'rebound': 0,
            'turnover': 0,
        }

        # Checkpoint settings
        self.checkpoint_interval = 50  # Save every 50 clips
        self.checkpoint_path = None
        self.processed_clips = set()  # (game_id, action_number) tuples

        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)

    def load_checkpoint(self, checkpoint_path):
        """Load existing checkpoint and resume collection"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return

        logger.info(f"📂 Loading checkpoint from {checkpoint_path}")

        try:
            df = pd.read_csv(checkpoint_path, dtype={'game_id': str})

            # Load dataset records
            self.dataset_records = df.to_dict('records')

            # Track processed clips (to skip duplicates)
            self.processed_clips = set(zip(df['game_id'], df['event_num']))

            # Restore event counts
            for event_type in self.event_counts.keys():
                clips_with_type = df[df['event_type'] == event_type]
                if not clips_with_type.empty:
                    self.event_counts[event_type] = clips_with_type.groupby(['game_id', 'event_num']).ngroups

            self.checkpoint_path = checkpoint_path

            clips_loaded = len(self.processed_clips)
            frames_loaded = len(self.dataset_records)

            logger.info(f"✅ Checkpoint loaded: {clips_loaded} clips, {frames_loaded} frames")
            logger.info(f"Event breakdown from checkpoint:")
            for event, count in self.event_counts.items():
                if count > 0:
                    logger.info(f"  {event}: {count} clips")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh collection...")

    def save_checkpoint(self, season, clips_collected):
        """Save checkpoint CSV locally and to S3"""
        if not self.dataset_records:
            return

        try:
            df = pd.DataFrame(self.dataset_records)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create or update checkpoint path
            if not self.checkpoint_path:
                self.checkpoint_path = f"data/checkpoints/non_foul_checkpoint_{season}_{timestamp}.csv"

            # Save locally
            df.to_csv(self.checkpoint_path, index=False)

            # Upload to S3
            checkpoint_filename = os.path.basename(self.checkpoint_path)
            s3_key = f"checkpoints/non_fouls/{checkpoint_filename}"

            try:
                self.s3_client.upload_file(self.checkpoint_path, self.bucket, s3_key)
                logger.info(f"💾 Checkpoint saved: {clips_collected} clips ({len(self.dataset_records)} frames)")
            except Exception as s3_error:
                logger.warning(f"Local checkpoint saved, but S3 upload failed: {s3_error}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_season_games(self, season='2023-24', max_games=None):
        """Get game IDs for a season using nba_api leaguegamefinder (still works)"""
        logger.info(f"Fetching games for {season}...")

        try:
            from nba_api.stats.endpoints import leaguegamefinder

            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                timeout=30
            )
            games = gamefinder.get_data_frames()[0]
            game_ids = games['GAME_ID'].unique().tolist()

            if max_games:
                game_ids = game_ids[:max_games]

            logger.info(f"Found {len(game_ids)} games")
            return game_ids

        except Exception as e:
            logger.error(f"Failed to get games via leaguegamefinder: {e}")
            logger.info("Falling back to game ID generation...")

            # Fallback: generate game IDs
            season_start_year = season.split('-')[0][-2:]
            game_ids = []
            max_game_num = max_games if max_games else 100

            for game_num in range(1, min(max_game_num + 1, 1231)):
                game_id = f"0022{season_start_year}{game_num:04d}"
                game_ids.append(game_id)

            logger.info(f"Generated {len(game_ids)} potential game IDs")
            return game_ids

    def get_playbyplay_from_cdn(self, game_id):
        """Get play-by-play data from CDN endpoint"""
        cdn_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

        time.sleep(0.6)  # Rate limiting

        try:
            response = requests.get(cdn_url, timeout=30)

            if response.status_code == 404:
                # Game doesn't exist
                return None

            response.raise_for_status()
            data = response.json()

            if 'game' in data and 'actions' in data['game']:
                return data['game']['actions']
            else:
                logger.warning(f"Unexpected CDN response structure for {game_id}")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"CDN request failed for {game_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting play-by-play for {game_id}: {e}")
            return None

    def filter_non_foul_actions(self, actions):
        """Filter for non-foul gameplay actions"""
        non_foul_actions = []

        for action in actions:
            action_type = action.get('actionType', '')

            # Check if this is a non-foul action we want
            if action_type in NON_FOUL_ACTION_TYPES:
                non_foul_actions.append(action)

        return non_foul_actions

    def get_video_url(self, game_id, action_number):
        """Get video URL for a specific action"""
        url = 'https://stats.nba.com/stats/videoeventsasset'
        params = {'GameEventID': action_number, 'GameID': game_id}

        time.sleep(0.6)  # Rate limiting

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            data = response.json()
            video_url = data['resultSets']['Meta']['videoUrls'][0]['lurl']
            return video_url
        except Exception as e:
            # Silently skip - many non-fouls won't have video
            return None

    def download_video(self, video_url, game_id, action_number):
        """Download video from NBA CDN"""
        video_path = f"data/non_foul_videos/{game_id}_{action_number}.mp4"

        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return video_path
        except Exception as e:
            return None

    def extract_frames(self, video_path, num_frames=30):
        """Extract frames evenly from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = []

        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()

            if ret:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frames.append({
                    'frame': frame,
                    'frame_index': target_idx,
                    'timestamp_sec': timestamp_ms / 1000.0
                })

        cap.release()
        return frames

    def upload_frame_to_s3(self, frame_data, game_id, action_number, frame_idx, season):
        """Upload single frame to S3 (separate path for non-fouls)"""
        frame_filename = f"{game_id}_{action_number}_frame_{frame_idx:03d}.jpg"
        local_path = f"data/non_foul_frames/{frame_filename}"

        cv2.imwrite(local_path, frame_data['frame'])

        # Use separate S3 path for non-fouls
        s3_key = f"frames/non_fouls/{season}/{game_id}/{frame_filename}"

        try:
            self.s3_client.upload_file(
                local_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )

            url = f"https://{self.bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
            os.remove(local_path)
            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None

    def process_clip(self, game_id, action_data, season):
        """Process one non-foul clip"""
        action_number = action_data.get('actionNumber')
        description = action_data.get('description', '')
        action_type = action_data.get('actionType', '')

        # Map CDN action type to our event categories
        event_type = NON_FOUL_ACTION_TYPES.get(action_type, 'other')

        # Get video URL
        video_url = self.get_video_url(game_id, action_number)
        if not video_url:
            return None

        # Download video
        video_path = self.download_video(video_url, game_id, action_number)
        if not video_path:
            return None

        # Extract frames
        frames = self.extract_frames(video_path, num_frames=30)
        if not frames:
            os.remove(video_path)
            return None

        # Upload frames and create records
        for frame_idx, frame_data in enumerate(frames):
            s3_url = self.upload_frame_to_s3(
                frame_data, game_id, action_number, frame_idx, season
            )

            if s3_url:
                record = {
                    'game_id': game_id,
                    'event_num': action_number,
                    'frame_index': frame_idx,
                    'frame_timestamp_sec': frame_data['timestamp_sec'],
                    'season': season,
                    'period': action_data.get('period'),
                    'game_clock': action_data.get('clock'),
                    'description': description,
                    'action_type': action_type,
                    'event_type': event_type,
                    'team_id': action_data.get('teamId'),
                    'team_tricode': action_data.get('teamTricode'),
                    'player_id': action_data.get('personId'),
                    'player_name': action_data.get('playerName'),
                    's3_url': s3_url,
                    'is_foul_frame': False,  # Explicitly mark as non-foul
                }

                self.dataset_records.append(record)

        # Mark as processed
        self.processed_clips.add((game_id, action_number))

        # Clean up video
        os.remove(video_path)
        return event_type

    def collect(self, season='2023-24', target_clips=1000, max_games=None):
        """Main collection loop"""
        logger.info(f"🚀 Starting NON-FOUL collection from {season}")
        logger.info(f"Target: {target_clips} non-foul clips")
        logger.info(f"Using CDN endpoint: cdn.nba.com")

        if self.processed_clips:
            logger.info(f"📍 Resuming from checkpoint with {len(self.processed_clips)} clips already collected")

        start_time = time.time()

        # Get games
        game_ids = self.get_season_games(season, max_games)

        clips_collected = len(self.processed_clips)
        clips_failed = 0
        games_processed = 0

        with tqdm(total=target_clips, desc="Collecting non-foul clips", initial=clips_collected) as pbar:
            for game_id in game_ids:
                if clips_collected >= target_clips:
                    break

                # Get play-by-play from CDN
                actions = self.get_playbyplay_from_cdn(game_id)

                if actions is None:
                    continue  # Game doesn't exist, skip silently

                games_processed += 1

                # Filter for non-foul actions
                non_foul_actions = self.filter_non_foul_actions(actions)

                if not non_foul_actions:
                    continue

                # Shuffle to get variety
                import random
                random.shuffle(non_foul_actions)

                # Process clips from this game
                for action in non_foul_actions:
                    if clips_collected >= target_clips:
                        break

                    action_number = action.get('actionNumber')
                    if (game_id, action_number) in self.processed_clips:
                        continue

                    try:
                        event_type = self.process_clip(game_id, action, season)

                        if event_type:
                            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
                            clips_collected += 1
                            pbar.update(1)

                            # Save checkpoint
                            if clips_collected % self.checkpoint_interval == 0:
                                self.save_checkpoint(season, clips_collected)

                            # Update progress
                            elapsed = time.time() - start_time
                            avg_time = elapsed / max(clips_collected - len(self.processed_clips) + clips_collected, 1)

                            event_str = " | ".join([
                                f"{et[:4].upper()}:{self.event_counts[et]}"
                                for et in ['made_shot', 'missed_shot', 'rebound', 'turnover']
                            ])

                            pbar.set_postfix({
                                'Failed': clips_failed,
                                'Games': games_processed,
                                'Events': event_str,
                                'Avg': f'{avg_time:.1f}s'
                            })
                        else:
                            clips_failed += 1

                    except Exception as e:
                        logger.error(f"Error processing clip: {e}")
                        clips_failed += 1

        # Finalize
        self.finalize_collection(season, clips_collected, clips_failed, start_time)

    def finalize_collection(self, season, clips_collected, clips_failed, start_time):
        """Save final metadata and print report"""
        df = pd.DataFrame(self.dataset_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/metadata/non_fouls_{season}_{clips_collected}clips_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Upload to S3
        self.upload_metadata_to_s3(csv_path)

        # Save final checkpoint
        if clips_collected > 0:
            self.save_checkpoint(season, clips_collected)

        # Report
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"NON-FOUL COLLECTION COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total clips collected: {clips_collected}")
        logger.info(f"Clips failed: {clips_failed}")

        logger.info(f"\nBreakdown by event type:")
        for event, count in sorted(self.event_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                logger.info(f"  {event:15s}: {count:4d} clips")

        logger.info(f"\nTotal time: {timedelta(seconds=int(total_time))}")
        logger.info(f"Frames in S3: {len(self.dataset_records)}")
        logger.info(f"Metadata saved: {csv_path}")
        logger.info(f"{'='*60}")

    def upload_metadata_to_s3(self, csv_path):
        """Upload CSV metadata to S3"""
        filename = os.path.basename(csv_path)
        s3_key = f"metadata/non_fouls/{filename}"

        try:
            self.s3_client.upload_file(csv_path, self.bucket, s3_key)
            s3_url = f"https://{self.bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Metadata uploaded to S3: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Failed to upload metadata: {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect NBA non-foul clips using CDN endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 100 non-foul clips
  python collect_non_fouls_cdn.py --clips 100

  # Collect from 2024-25 season
  python collect_non_fouls_cdn.py --season 2024-25 --clips 500

  # Resume from checkpoint
  python collect_non_fouls_cdn.py --clips 1000 --resume-from data/checkpoints/non_foul_checkpoint_*.csv

Features:
  ✅ Uses working CDN endpoint (cdn.nba.com)
  ✅ Separate storage from foul data (won't interfere with annotation)
  ✅ Auto-checkpoint every 50 clips
  ✅ Resume from interruption
  ✅ All clips marked as is_foul_frame=False (no annotation needed)
        """
    )
    parser.add_argument('--season', default='2023-24', help='NBA season (e.g., 2023-24)')
    parser.add_argument('--clips', type=int, default=1000, help='Number of non-foul clips to collect')
    parser.add_argument('--max-games', type=int, default=None, help='Max games to scan')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint CSV')

    args = parser.parse_args()

    collector = NonFoulCollector(resume_from=args.resume_from)
    collector.collect(
        season=args.season,
        target_clips=args.clips,
        max_games=args.max_games
    )
