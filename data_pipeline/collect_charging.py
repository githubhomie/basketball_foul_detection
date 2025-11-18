# Specialized script to collect ONLY charging fouls across multiple seasons
# Target: 165 charging fouls (10% overshoot for 150 target)

import sys
sys.path.insert(0, '/Users/oliverhazard/AI_Referee/foul_data_aws')

from collect_data import *

# Override targets to collect ONLY charging fouls
COLLECTION_TARGETS = {
    'shooting_foul': 400,      # Already complete
    'personal_foul': 349,      # Already complete
    'loose_ball': 200,         # Already complete
    'charging': 165,           # TARGET: 165 (10% overshoot), current: 86, need: 79
    'offensive_foul': 100,     # Already complete
}

def collect_charging_across_seasons(resume_from):
    """Collect charging fouls by scanning multiple seasons"""

    collector = NBAFoulCollector(use_targets=True, resume_from=resume_from)

    # Seasons to scan (newest to oldest)
    seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20']

    logger.info(f"🎯 CHARGING FOUL COLLECTION (targeting 165 total)")
    logger.info(f"   Current: {collector.category_counts['charging']}")
    logger.info(f"   Need: {165 - collector.category_counts['charging']} more")
    logger.info(f"   Will scan across seasons: {', '.join(seasons)}")

    start_time = time.time()
    total_inventory = {'charging': []}

    # Scan across multiple seasons until we have enough
    for season in seasons:
        current_charging = collector.category_counts['charging']
        if current_charging >= 165:
            logger.info(f"✅ Target reached! Collected {current_charging}/165 charging fouls")
            break

        needed = 165 - current_charging
        logger.info(f"\n🔍 Scanning season {season} for charging fouls (need {needed} more)...")

        try:
            # Get games for this season
            game_ids = collector.get_season_games(season, max_games=None)

            # Scan more games (300 per season if needed)
            games_to_scan = min(300, len(game_ids))
            logger.info(f"   Scanning {games_to_scan} games from {season}...")

            # Build inventory for this season
            season_inventory = collector.build_foul_inventory(game_ids, max_games_to_scan=games_to_scan)

            # Add charging fouls to total inventory
            charging_found = len(season_inventory['charging'])
            total_inventory['charging'].extend(season_inventory['charging'])

            logger.info(f"   Found {charging_found} charging fouls in {season}")
            logger.info(f"   Total inventory now: {len(total_inventory['charging'])} charging fouls")

            # If we have enough in inventory, start downloading
            if len(total_inventory['charging']) >= needed:
                logger.info(f"✅ Inventory has enough charging fouls! Starting download...")
                break

        except Exception as e:
            logger.error(f"Error scanning season {season}: {e}")
            continue

    # Now download only charging fouls from inventory
    if not total_inventory['charging']:
        logger.warning("⚠️ No charging fouls found in inventory!")
        return

    # Select charging fouls to download
    current = collector.category_counts['charging']
    needed = 165 - current
    to_download = min(needed, len(total_inventory['charging']))

    logger.info(f"\n📥 Downloading {to_download} charging fouls...")

    selected_charging = total_inventory['charging'][:to_download]
    clips_collected = len(collector.processed_clips)
    clips_failed = 0

    with tqdm(total=to_download, desc="Downloading charging fouls") as pbar:
        for game_id, foul_data in selected_charging:
            try:
                # Process clip
                foul_type = collector.process_clip(game_id, foul_data, foul_data.get('SEASON', '2023-24'))

                if foul_type:
                    collector.category_counts[foul_type] += 1
                    clips_collected += 1
                    pbar.update(1)

                    # Save checkpoint every 50 clips
                    if clips_collected % 50 == 0:
                        collector.save_checkpoint('multi-season', clips_collected)

                    pbar.set_postfix({
                        'Charging': f"{collector.category_counts['charging']}/165",
                        'Failed': clips_failed
                    })
                else:
                    clips_failed += 1

            except Exception as e:
                logger.error(f"Error processing charging foul: {e}")
                clips_failed += 1

    # Finalize
    collector.finalize_collection('multi-season', clips_collected, clips_failed, start_time)

if __name__ == "__main__":
    checkpoint_path = "data/checkpoints/checkpoint_2023-24_20251014_155958.csv"
    collect_charging_across_seasons(checkpoint_path)
