import pandas as pd

def extract_player_features(df):
    df['player_name'] = df['player'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    player_stats = df.groupby('player_name')['type'].apply(lambda x: x.map(lambda e: e['name'] if isinstance(e, dict) else None).value_counts()).unstack().fillna(0)
    return player_stats
