# collect_data.py
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
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2

# Load environment
load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collection.log'),
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

# Collection targets - based on NBA API EVENTMSGACTIONTYPE codes
# 5 foul categories + 1 non-foul category
COLLECTION_TARGETS = {
    'shooting_foul': 400,      # Code 2 - Contact during shot attempt (most common ~56%)
    'personal_foul': 350,      # Code 1 - Defensive contact (blocking, reach-in, holding) (~29%)
    'loose_ball': 200,         # Code 3 - Loose ball/rebounding situations (~6%)
    'charging': 150,           # Code 26 - Offensive charge (player drives into set defender) (~3%)
    'offensive_foul': 100,     # Code 4 - Other offensive fouls (screens, push-offs) (~4%)
    'non_foul': 1000,          # Non-foul gameplay clips (for negative examples)
}

# Non-foul event types to collect from NBA API
# EVENTMSGTYPE codes for non-foul events
NON_FOUL_EVENT_TYPES = {
    'made_shot': 1,            # Field goal made
    'missed_shot': 2,          # Field goal missed
    'rebound': 3,              # Rebound
    'turnover': 5,             # Turnover (includes steals)
}

# Sub-targets for non-foul diversity
NON_FOUL_TARGETS = {
    'random_gameplay': 500,    # Made shots, missed shots (easy negatives)
    'physical_clean': 300,     # Rebounds, steals (hard negatives - physical but legal)
    'similar_plays': 200,      # Drives/layups with no foul (hardest negatives)
}

# NBA API EVENTMSGACTIONTYPE mapping (verified from real data)
# Maps API codes directly to our 5 collection categories
ACTION_TYPE_MAP = {
    1: 'personal_foul',      # Personal foul (defensive: blocking, reach-in, holding)
    2: 'shooting_foul',      # Shooting foul
    3: 'loose_ball',         # Loose ball foul
    4: 'offensive_foul',     # Offensive foul (screens, push-offs, etc.)
    26: 'charging',          # Offensive charge foul
    # Other codes we skip or count as "other"
    6: 'other',              # Away from play foul
    11: 'other',             # Technical foul
    14: 'other',             # Flagrant foul type 1 (rare, skip)
    15: 'other',             # Flagrant foul type 2 (rare, skip)
    17: 'other',             # Defensive 3 seconds
    18: 'other',             # Delay of game
    28: 'other',             # Personal take foul
    31: 'other',             # Transition take foul
    32: 'other',             # Flopping technical
}

def classify_foul_type(action_type):
    """
    Classify foul using NBA API EVENTMSGACTIONTYPE code.
    Returns one of 5 categories or 'other'/'unknown'.
    """
    return ACTION_TYPE_MAP.get(action_type, 'unknown')

class NBAFoulCollector:
    def __init__(self, use_targets=False, resume_from=None):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket = os.getenv('S3_BUCKET', 'nba-foul-dataset-oh')

        # Create temp directories
        os.makedirs('data/temp_videos', exist_ok=True)
        os.makedirs('data/temp_frames', exist_ok=True)
        os.makedirs('data/metadata', exist_ok=True)
        os.makedirs('data/checkpoints', exist_ok=True)

        self.dataset_records = []

        # Track collection by foul type (5 foul categories + non-foul + other/unknown)
        self.use_targets = use_targets
        self.category_counts = {
            'shooting_foul': 0,
            'personal_foul': 0,
            'loose_ball': 0,
            'charging': 0,
            'offensive_foul': 0,
            'non_foul': 0,
            'other': 0,
            'unknown': 0
        }

        # Checkpoint settings
        self.checkpoint_interval = 50  # Save every 50 clips
        self.checkpoint_path = None
        self.processed_clips = set()  # (game_id, event_num) tuples

        # Team diversity tracking (bonus)
        self.team_counts = {}

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

            # Restore category counts
            for foul_type in self.category_counts.keys():
                clips_with_type = df[df['foul_type'] == foul_type]
                if not clips_with_type.empty:
                    # Count unique clips (not frames)
                    self.category_counts[foul_type] = clips_with_type.groupby(['game_id', 'event_num']).ngroups

            # Restore team counts
            if 'fouler_team' in df.columns:
                team_counts = df.groupby('fouler_team')['game_id'].nunique().to_dict()
                self.team_counts.update(team_counts)

            # Set checkpoint path for continued saving
            self.checkpoint_path = checkpoint_path

            clips_loaded = len(self.processed_clips)
            frames_loaded = len(self.dataset_records)

            logger.info(f"✅ Checkpoint loaded: {clips_loaded} clips, {frames_loaded} frames")
            logger.info(f"Category breakdown from checkpoint:")
            for cat, count in self.category_counts.items():
                if count > 0:
                    logger.info(f"  {cat}: {count} clips")

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
                self.checkpoint_path = f"data/checkpoints/checkpoint_{season}_{timestamp}.csv"

            # Save locally
            df.to_csv(self.checkpoint_path, index=False)

            # Upload to S3 for extra safety
            checkpoint_filename = os.path.basename(self.checkpoint_path)
            s3_key = f"checkpoints/{checkpoint_filename}"

            try:
                self.s3_client.upload_file(self.checkpoint_path, self.bucket, s3_key)
                logger.info(f"💾 Checkpoint saved: {clips_collected} clips ({len(self.dataset_records)} frames)")
            except Exception as s3_error:
                logger.warning(f"Local checkpoint saved, but S3 upload failed: {s3_error}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_season_games(self, season='2023-24', max_games=None):
        """Get all game IDs for a season"""
        logger.info(f"Fetching games for {season}...")

        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        games = gamefinder.get_data_frames()[0]
        game_ids = games['GAME_ID'].unique().tolist()

        if max_games:
            game_ids = game_ids[:max_games]

        logger.info(f"Found {len(game_ids)} games")
        return game_ids

    def build_foul_inventory(self, game_ids, max_games_to_scan=150):
        """
        Build inventory of available fouls from many games (WITHOUT downloading videos).
        This allows us to intelligently select rare foul types instead of searching randomly.

        Returns: Dictionary mapping foul_type -> list of (game_id, foul_data) tuples
        """
        logger.info(f"🔍 Building foul inventory from {min(len(game_ids), max_games_to_scan)} games...")
        logger.info("   (Querying play-by-play metadata only - fast, no video downloads)")

        inventory = {cat: [] for cat in COLLECTION_TARGETS.keys()}
        inventory['other'] = []
        inventory['unknown'] = []

        games_scanned = 0

        with tqdm(total=min(len(game_ids), max_games_to_scan), desc="Scanning games") as pbar:
            for game_id in game_ids[:max_games_to_scan]:
                fouls = self.get_fouls_from_game(game_id)

                if not fouls.empty:
                    for _, foul in fouls.iterrows():
                        event_num = foul['EVENTNUM']

                        # Skip if already processed
                        if (game_id, event_num) in self.processed_clips:
                            continue

                        # Classify foul
                        action_type = foul['EVENTMSGACTIONTYPE']
                        foul_type = classify_foul_type(action_type)

                        # Add to inventory
                        inventory[foul_type].append((game_id, foul))

                games_scanned += 1
                pbar.update(1)

                # Show running totals
                cat_str = " | ".join([
                    f"{cat[:3].upper()}:{len(inventory[cat])}"
                    for cat in ['shooting_foul', 'personal_foul', 'loose_ball', 'charging', 'offensive_foul']
                ])
                pbar.set_postfix_str(cat_str)

        # Report inventory
        logger.info(f"✅ Inventory built from {games_scanned} games:")
        for cat in ['shooting_foul', 'personal_foul', 'loose_ball', 'charging', 'offensive_foul']:
            available = len(inventory[cat])
            needed = max(0, COLLECTION_TARGETS[cat] - self.category_counts[cat])
            status = "✓" if available >= needed else "⚠"
            logger.info(f"  {status} {cat:15s}: {available:4d} available, {needed:4d} needed")

        return inventory

    def build_non_foul_inventory(self, game_ids, max_games_to_scan=150):
        """
        Build inventory of non-foul clips from games.

        Returns: List of (game_id, play_data) tuples
        """
        logger.info(f"🔍 Building non-foul inventory from {min(len(game_ids), max_games_to_scan)} games...")

        inventory = []
        games_scanned = 0

        with tqdm(total=min(len(game_ids), max_games_to_scan), desc="Scanning for non-fouls") as pbar:
            for game_id in game_ids[:max_games_to_scan]:
                non_fouls = self.get_non_fouls_from_game(game_id)

                if not non_fouls.empty:
                    for _, play in non_fouls.iterrows():
                        event_num = play['EVENTNUM']

                        # Skip if already processed
                        if (game_id, event_num) in self.processed_clips:
                            continue

                        inventory.append((game_id, play))

                games_scanned += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Found: {len(inventory)}")

        logger.info(f"✅ Non-foul inventory: {len(inventory)} clips available")
        return inventory

    def select_balanced_fouls(self, inventory):
        """
        Select fouls from inventory to meet targets.
        Prioritizes rare types and ensures we don't exceed targets.

        Returns: List of (game_id, foul_data, foul_type) tuples to process
        """
        selected = []

        # Calculate what we still need
        needs = {}
        for cat, target in COLLECTION_TARGETS.items():
            current = self.category_counts[cat]
            needs[cat] = max(0, target - current)

        logger.info("📋 Selecting clips to meet targets:")
        for cat, need in needs.items():
            available = len(inventory[cat])
            to_select = min(need, available)

            # Select from inventory (take all we need, up to what's available)
            selected_from_cat = inventory[cat][:to_select]

            for game_id, foul_data in selected_from_cat:
                selected.append((game_id, foul_data, cat))

            logger.info(f"  {cat:15s}: selecting {to_select:3d} clips (need {need:3d}, available {available:3d})")

        logger.info(f"✅ Total clips selected: {len(selected)}")
        return selected

    def get_fouls_from_game(self, game_id):
        """Get all fouls with video from a game"""
        time.sleep(0.6)  # Rate limiting

        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            plays_df = pbp.get_data_frames()[0]

            fouls = plays_df[
                (plays_df['EVENTMSGTYPE'] == 6) &
                (plays_df['VIDEO_AVAILABLE_FLAG'] == 1)
            ].copy()

            return fouls
        except Exception as e:
            logger.error(f"Error getting fouls for {game_id}: {e}")
            return pd.DataFrame()

    def get_non_fouls_from_game(self, game_id, event_types=[1, 2, 3, 5]):
        """
        Get non-foul events with video from a game

        Args:
            game_id: NBA game ID
            event_types: List of EVENTMSGTYPE codes to collect
                1 = Made shot
                2 = Missed shot
                3 = Rebound
                5 = Turnover (includes steals)
        """
        time.sleep(0.6)  # Rate limiting

        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            plays_df = pbp.get_data_frames()[0]

            # Get plays that match event types and have video
            non_fouls = plays_df[
                (plays_df['EVENTMSGTYPE'].isin(event_types)) &
                (plays_df['VIDEO_AVAILABLE_FLAG'] == 1)
            ].copy()

            return non_fouls
        except Exception as e:
            logger.error(f"Error getting non-fouls for {game_id}: {e}")
            return pd.DataFrame()

    def get_video_url(self, game_id, event_num):
        """Get video URL for a specific play"""
        url = 'https://stats.nba.com/stats/videoeventsasset'
        params = {'GameEventID': event_num, 'GameID': game_id}

        time.sleep(0.6)  # Rate limiting

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            data = response.json()
            video_url = data['resultSets']['Meta']['videoUrls'][0]['lurl']
            return video_url
        except Exception as e:
            logger.warning(f"No video for {game_id}/{event_num}: {e}")
            return None

    def download_video(self, video_url, game_id, event_num):
        """Download video from NBA CDN"""
        video_path = f"data/temp_videos/{game_id}_{event_num}.mp4"

        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return video_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def extract_frames(self, video_path, num_frames=30):
        """Extract frames evenly from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

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

    def upload_frame_to_s3(self, frame_data, game_id, event_num, frame_idx, season):
        """Upload single frame to S3"""
        # Save frame temporarily
        frame_filename = f"{game_id}_{event_num}_frame_{frame_idx:03d}.jpg"
        local_path = f"data/temp_frames/{frame_filename}"

        cv2.imwrite(local_path, frame_data['frame'])

        # Upload to S3
        s3_key = f"frames/{season}/{game_id}/{frame_filename}"

        try:
            self.s3_client.upload_file(
                local_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )

            # Generate public URL
            url = f"https://{self.bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

            # Clean up local file
            os.remove(local_path)

            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None

    def process_clip(self, game_id, play_data, season, is_non_foul=False):
        """
        Process one clip (foul or non-foul)

        Args:
            game_id: NBA game ID
            play_data: Play-by-play data from NBA API
            season: Season string (e.g., '2023-24')
            is_non_foul: Whether this is a non-foul clip
        """
        event_num = play_data['EVENTNUM']
        description = play_data.get('HOMEDESCRIPTION') or play_data.get('VISITORDESCRIPTION')
        action_type = play_data['EVENTMSGACTIONTYPE']

        # Classify clip type
        if is_non_foul:
            clip_type = 'non_foul'
        else:
            clip_type = classify_foul_type(action_type)

        # Get video URL
        video_url = self.get_video_url(game_id, event_num)
        if not video_url:
            return None

        # Download video
        video_path = self.download_video(video_url, game_id, event_num)
        if not video_path:
            return None

        # Extract frames
        frames = self.extract_frames(video_path, num_frames=30)
        if not frames:
            os.remove(video_path)
            return None

        # Track team for diversity
        team = play_data.get('PLAYER1_TEAM_ABBREVIATION')
        if team:
            self.team_counts[team] = self.team_counts.get(team, 0) + 1

        # Upload frames and create records
        for frame_idx, frame_data in enumerate(frames):
            s3_url = self.upload_frame_to_s3(
                frame_data, game_id, event_num, frame_idx, season
            )

            if s3_url:
                record = {
                    'game_id': game_id,
                    'event_num': event_num,
                    'frame_index': frame_idx,
                    'frame_timestamp_sec': frame_data['timestamp_sec'],
                    'season': season,
                    'period': play_data.get('PERIOD'),
                    'game_clock': play_data.get('PCTIMESTRING'),
                    'description': description,
                    'action_type': action_type,     # NBA API code
                    'foul_type': clip_type,         # Our classification (or 'non_foul')
                    'fouler_id': play_data.get('PLAYER1_ID'),
                    'fouler_name': play_data.get('PLAYER1_NAME'),
                    'fouler_team': team,
                    'fouled_player_id': play_data.get('PLAYER2_ID'),
                    'fouled_player_name': play_data.get('PLAYER2_NAME'),
                    'fouled_team': play_data.get('PLAYER2_TEAM_ABBREVIATION'),
                    's3_url': s3_url,
                    'is_foul_frame': False if is_non_foul else None,  # Non-fouls: explicitly False, fouls: to be annotated
                }

                self.dataset_records.append(record)

        # Mark as processed
        self.processed_clips.add((game_id, event_num))

        # Clean up video
        os.remove(video_path)
        return clip_type  # Return clip type for tracking

    def upload_metadata_to_s3(self, csv_path):
        """Upload CSV metadata to S3"""
        filename = os.path.basename(csv_path)
        s3_key = f"metadata/{filename}"

        try:
            self.s3_client.upload_file(csv_path, self.bucket, s3_key)
            s3_url = f"https://{self.bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Metadata uploaded to S3: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"Failed to upload metadata: {e}")
            return None

    def check_category_complete(self, foul_type):
        """Check if we've reached target for this category"""
        if not self.use_targets or foul_type not in COLLECTION_TARGETS:
            return False
        return self.category_counts[foul_type] >= COLLECTION_TARGETS[foul_type]

    def all_targets_met(self):
        """Check if all collection targets are met"""
        if not self.use_targets:
            return False
        for category, target in COLLECTION_TARGETS.items():
            if self.category_counts[category] < target:
                return False
        return True

    def collect(self, season='2023-24', target_clips=100, max_games=None, collect_non_fouls=False):
        """Main collection loop with optional per-category targeting"""
        if collect_non_fouls:
            # Collect non-foul clips
            return self.collect_non_fouls(season, max_games)
        elif self.use_targets:
            # Use optimized inventory-based collection
            return self.collect_with_inventory(season, max_games)
        else:
            # Simple collection without targets
            return self.collect_simple(season, target_clips, max_games)

    def collect_with_inventory(self, season='2023-24', max_games=None):
        """Optimized collection using inventory-based approach"""
        logger.info(f"🚀 Starting OPTIMIZED targeted collection from {season}")
        logger.info("Collection targets (realistic distribution):")
        for cat, target in COLLECTION_TARGETS.items():
            logger.info(f"  {cat}: {target} clips")
        total_target = sum(COLLECTION_TARGETS.values())

        # Check if resuming from checkpoint
        if self.processed_clips:
            logger.info(f"📍 Resuming from checkpoint with {len(self.processed_clips)} clips already collected")

        start_time = time.time()

        # Get games
        game_ids = self.get_season_games(season, max_games)

        # Build inventory (scan many games for metadata only - FAST)
        inventory = self.build_foul_inventory(game_ids, max_games_to_scan=150)

        # Select clips to meet targets
        selected_fouls = self.select_balanced_fouls(inventory)

        if not selected_fouls:
            logger.warning("No clips selected from inventory. Targets may already be met.")
            return

        # Now download and process only the selected clips
        clips_collected = len(self.processed_clips)  # Start from checkpoint if resuming
        clips_failed = 0

        logger.info(f"\n🎬 Downloading and processing {len(selected_fouls)} selected clips...")

        with tqdm(total=total_target, desc="Processing clips", initial=clips_collected) as pbar:
            for game_id, foul_data, expected_foul_type in selected_fouls:
                try:
                    foul_type = self.process_clip(game_id, foul_data, season)

                    if foul_type:
                        self.category_counts[foul_type] += 1
                        clips_collected += 1
                        pbar.update(1)

                        # Save checkpoint every N clips
                        if clips_collected % self.checkpoint_interval == 0:
                            self.save_checkpoint(season, clips_collected)

                        # Update progress display
                        elapsed = time.time() - start_time
                        new_clips = clips_collected - len(self.processed_clips)
                        avg_time = elapsed / max(new_clips, 1)
                        remaining = total_target - clips_collected
                        eta = remaining * avg_time

                        # Show category breakdown
                        category_str = " | ".join([
                            f"{cat[:3].upper()}:{self.category_counts[cat]}/{COLLECTION_TARGETS[cat]}"
                            for cat in ['shooting_foul', 'personal_foul', 'loose_ball', 'charging', 'offensive_foul']
                        ])

                        team_diversity = len(self.team_counts)

                        pbar.set_postfix({
                            'Failed': clips_failed,
                            'Teams': team_diversity,
                            'Cats': category_str,
                            'Avg': f'{avg_time:.1f}s',
                            'ETA': str(timedelta(seconds=int(eta)))
                        })
                    else:
                        clips_failed += 1

                except Exception as e:
                    logger.error(f"Error processing clip: {e}")
                    clips_failed += 1

        # Save final metadata and report
        self.finalize_collection(season, clips_collected, clips_failed, start_time)

    def collect_simple(self, season='2023-24', target_clips=100, max_games=None):
        """Simple collection without inventory (for small collections)"""
        logger.info(f"Starting collection: {target_clips} clips from {season}")

        if self.processed_clips:
            logger.info(f"📍 Resuming from checkpoint with {len(self.processed_clips)} clips already collected")

        start_time = time.time()
        game_ids = self.get_season_games(season, max_games)
        clips_collected = len(self.processed_clips)
        clips_failed = 0

        with tqdm(total=target_clips, desc="Collecting clips", initial=clips_collected) as pbar:
            for game_id in game_ids:
                if clips_collected >= target_clips:
                    break

                fouls = self.get_fouls_from_game(game_id)
                if fouls.empty:
                    continue

                for _, foul in fouls.iterrows():
                    if clips_collected >= target_clips:
                        break

                    event_num = foul['EVENTNUM']
                    if (game_id, event_num) in self.processed_clips:
                        continue

                    try:
                        foul_type = self.process_clip(game_id, foul, season)
                        if foul_type:
                            self.category_counts[foul_type] += 1
                            clips_collected += 1
                            pbar.update(1)

                            if clips_collected % self.checkpoint_interval == 0:
                                self.save_checkpoint(season, clips_collected)
                        else:
                            clips_failed += 1
                    except Exception as e:
                        logger.error(f"Error processing clip: {e}")
                        clips_failed += 1

        self.finalize_collection(season, clips_collected, clips_failed, start_time)

    def collect_non_fouls(self, season='2023-24', max_games=None):
        """Collect non-foul clips for negative examples"""
        target = COLLECTION_TARGETS.get('non_foul', 1000)
        logger.info(f"🚀 Starting non-foul collection from {season}")
        logger.info(f"Target: {target} non-foul clips")
        logger.info("These clips will NOT require frame annotation")

        # Check if resuming
        if self.processed_clips:
            logger.info(f"📍 Resuming from checkpoint with {len(self.processed_clips)} clips already collected")

        start_time = time.time()

        # Get games
        game_ids = self.get_season_games(season, max_games)

        # Build inventory
        inventory = self.build_non_foul_inventory(game_ids, max_games_to_scan=150)

        if not inventory:
            logger.warning("No non-foul clips found in inventory.")
            return

        # Calculate how many we need
        current = self.category_counts['non_foul']
        needed = max(0, target - current)
        to_collect = min(needed, len(inventory))

        logger.info(f"Will collect {to_collect} clips (current: {current}, target: {target})")

        # Randomly sample from inventory for diversity
        import random
        random.shuffle(inventory)
        selected = inventory[:to_collect]

        # Process selected clips
        clips_collected = current  # Start from current count
        clips_failed = 0

        logger.info(f"\n🎬 Downloading and processing {len(selected)} non-foul clips...")

        with tqdm(total=target, desc="Processing non-fouls", initial=clips_collected) as pbar:
            for game_id, play_data in selected:
                try:
                    clip_type = self.process_clip(game_id, play_data, season, is_non_foul=True)

                    if clip_type == 'non_foul':
                        self.category_counts['non_foul'] += 1
                        clips_collected += 1
                        pbar.update(1)

                        # Save checkpoint every N clips
                        if clips_collected % self.checkpoint_interval == 0:
                            self.save_checkpoint(season, clips_collected)

                        # Update progress
                        elapsed = time.time() - start_time
                        new_clips = clips_collected - current
                        avg_time = elapsed / max(new_clips, 1)
                        remaining = target - clips_collected
                        eta = remaining * avg_time

                        pbar.set_postfix({
                            'Failed': clips_failed,
                            'Collected': f'{clips_collected}/{target}',
                            'Avg': f'{avg_time:.1f}s',
                            'ETA': str(timedelta(seconds=int(eta)))
                        })
                    else:
                        clips_failed += 1

                except Exception as e:
                    logger.error(f"Error processing non-foul clip: {e}")
                    clips_failed += 1

        # Finalize
        self.finalize_collection(season, clips_collected, clips_failed, start_time)

    def finalize_collection(self, season, clips_collected, clips_failed, start_time):
        """Save final metadata and print report"""
        # Save final metadata
        df = pd.DataFrame(self.dataset_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/metadata/nba_fouls_{season}_{clips_collected}clips_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Upload to S3
        s3_url = self.upload_metadata_to_s3(csv_path)

        # Save final checkpoint
        if clips_collected > 0:
            self.save_checkpoint(season, clips_collected)

        # Final report
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total clips collected: {clips_collected}")
        logger.info(f"Clips failed: {clips_failed}")

        logger.info(f"\nBreakdown by foul type:")
        for cat, count in sorted(self.category_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                target = COLLECTION_TARGETS.get(cat, 'N/A')
                status = f"({count}/{target})" if self.use_targets and cat in COLLECTION_TARGETS else ""
                logger.info(f"  {cat:15s}: {count:4d} clips {status}")

        if self.team_counts:
            logger.info(f"\nTeam diversity:")
            logger.info(f"  Total teams represented: {len(self.team_counts)}")
            top_teams = sorted(self.team_counts.items(), key=lambda x: -x[1])[:5]
            logger.info(f"  Top 5 teams: {', '.join([f'{t}({c})' for t, c in top_teams])}")

        logger.info(f"\nTotal time: {timedelta(seconds=int(total_time))}")
        new_clips = clips_collected - (len(self.processed_clips) - clips_collected)
        if new_clips > 0:
            logger.info(f"Average per clip: {total_time/max(new_clips, 1):.2f}s")
        logger.info(f"Frames in S3: {len(self.dataset_records)}")
        logger.info(f"Metadata saved locally: {csv_path}")
        if s3_url:
            logger.info(f"Metadata saved to S3: {s3_url}")
        if self.checkpoint_path:
            logger.info(f"Checkpoint saved: {self.checkpoint_path}")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect NBA foul clips with 5-category classification and checkpoint support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 50 foul clips (any types)
  python collect_data.py --clips 50

  # Use balanced collection targets (1200 foul clips across 5 categories)
  python collect_data.py --use-targets

  # Collect non-foul clips for negative examples (1000 clips, no annotation needed)
  python collect_data.py --collect-non-fouls

  # Collect from specific season with game limit
  python collect_data.py --season 2022-23 --clips 100 --max-games 50

  # Resume from checkpoint (auto-saves every 50 clips)
  python collect_data.py --use-targets --resume-from data/checkpoints/checkpoint_2023-24_*.csv

Collection Categories (based on NBA API codes):
  FOUL CATEGORIES:
  - shooting_foul: 400 clips (Code 2) - Contact during shot attempt (~56%)
  - personal_foul: 350 clips (Code 1) - Defensive contact (~29%)
  - loose_ball: 200 clips (Code 3) - Loose ball situations (~6%)
  - charging: 150 clips (Code 26) - Offensive charge (~3%)
  - offensive_foul: 100 clips (Code 4) - Other offensive fouls (~4%)

  NON-FOUL CATEGORY:
  - non_foul: 1000 clips - Made shots, missed shots, rebounds, turnovers
    (No frame annotation needed - just verify no foul occurred)

Features:
  ✅ OPTIMIZED inventory-based collection (scans games first, then downloads)
  ✅ Auto-checkpoint every 50 clips (saved locally + S3)
  ✅ Resume from interruption with --resume-from
  ✅ Team diversity tracking
  ✅ Uses NBA API action codes (not text parsing)
        """
    )
    parser.add_argument('--season', default='2023-24', help='NBA season (e.g., 2023-24)')
    parser.add_argument('--clips', type=int, default=100, help='Number of clips to collect (ignored if --use-targets)')
    parser.add_argument('--max-games', type=int, default=None, help='Max games to process')
    parser.add_argument('--use-targets', action='store_true',
                        help='Use OPTIMIZED 5-category targets (400 shooting, 350 personal, 200 loose ball, 150 charging, 100 offensive)')
    parser.add_argument('--collect-non-fouls', action='store_true',
                        help='Collect non-foul clips for negative examples (target: 1000 clips, no annotation needed)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume collection from checkpoint CSV file (e.g., data/checkpoints/checkpoint_2023-24_*.csv)')

    args = parser.parse_args()

    collector = NBAFoulCollector(use_targets=args.use_targets, resume_from=args.resume_from)
    collector.collect(
        season=args.season,
        target_clips=args.clips,
        max_games=args.max_games,
        collect_non_fouls=args.collect_non_fouls
    )
