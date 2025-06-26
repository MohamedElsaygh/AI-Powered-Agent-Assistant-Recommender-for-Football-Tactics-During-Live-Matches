import pandas as pd
import matplotlib.pyplot as plt

# 1. Plot event type distribution
def plot_event_type_distribution(df):
    if 'type' not in df.columns:
        print("Missing 'type' column.")
        return
    event_types = df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    counts = event_types.value_counts().head(15)
    counts.plot(kind='bar', title='Top 15 Event Types')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# 2. List substitutions
def list_substitutions(df):
    if 'type' not in df.columns:
        print("Missing 'type' column.")
        return pd.DataFrame()
    return df[df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None) == 'Substitution']

# 3. Count passes per player
def count_passes_per_player(df):
    if 'type' not in df.columns or 'player' not in df.columns:
        print("Missing columns.")
        return pd.DataFrame()
    passes = df[df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None) == 'Pass']
    player_names = passes['player'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    return player_names.value_counts().head(15)

# 4. Plot substitutions over time
def plot_substitutions_by_minute(df):
    subs = list_substitutions(df)
    if 'minute' not in subs.columns:
        print("Missing 'minute' column in substitution events.")
        return
    subs['minute'].plot.hist(bins=10, title='Substitutions per Time Bucket')
    plt.xlabel('Match Minute')
    plt.tight_layout()
    plt.show()

# 5. Helper to inspect player actions
def get_events_for_player(df, player_name):
    return df[df['player'].apply(lambda x: x.get('name') if isinstance(x, dict) else None) == player_name]
def count_event_types(df):
    return df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None).value_counts()
