import pandas as pd
from collections import defaultdict

def extract_player_match_stats(df):
    """
    Extract per-player per-match stats from raw events data.
    Each row = one player's aggregated stats for one match.
    """

    required = ['match_id', 'player', 'type', 'minute']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    stats = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        if not isinstance(row['player'], dict):
            continue

        player_id = row['player']['id']
        match_id = row['match_id']
        minute = row['minute']
        event_type = row['type']['name'] if isinstance(row['type'], dict) else None
        key = (match_id, player_id)

        stats[key]['player_id'] = player_id
        stats[key]['match_id'] = match_id
        stats[key]['minutes_played'] = max(stats[key].get('minutes_played', 0), minute)

        if event_type:
            stats[key][event_type] += 1

    return pd.DataFrame(stats.values())
