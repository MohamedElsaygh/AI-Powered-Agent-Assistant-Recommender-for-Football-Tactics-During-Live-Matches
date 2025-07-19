import pandas as pd
import numpy as np

def extract_player_match_features(events_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["match_id", "player_id", "type", "timestamp", "position"]
    for col in required_cols:
        if col not in events_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract event type names
    events_df["event_type"] = events_df["type"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    # Count event types per player per match
    counts = events_df.groupby(["match_id", "player_id", "event_type"]).size().unstack(fill_value=0)

    # Estimate minutes played
    time_df = events_df.groupby(["match_id", "player_id"])["timestamp"].agg(['min', 'max']).reset_index()
    time_df["minutes_played"] = (
        pd.to_timedelta(time_df["max"]) - pd.to_timedelta(time_df["min"])
    ).dt.total_seconds() / 60

    # Extract position
    pos_df = events_df.dropna(subset=["position"]).groupby(["match_id", "player_id"])["position"].last().reset_index()
    pos_df["position_name"] = pos_df["position"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    # Merge features
    feature_df = counts.reset_index()
    feature_df = feature_df.merge(time_df[["match_id", "player_id", "minutes_played"]], on=["match_id", "player_id"], how="left")
    feature_df = feature_df.merge(pos_df[["match_id", "player_id", "position_name"]], on=["match_id", "player_id"], how="left")
    
    # Extract duels lost
    duel_events = events_df[events_df["type_name"] == "Duel"].copy()

# Filter for lost duels only
    duel_events["duel_outcome"] = duel_events["duel"].apply(
    lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
    )

    duels_lost = duel_events[duel_events["duel_outcome"] == "Lost"]

# Count duels lost per player per match
    duels_lost_count = duels_lost.groupby(["match_id", "player_id"]).size().reset_index(name="duels_lost")

# Merge into features_df
    feature_df = feature_df.merge(duels_lost_count, on=["match_id", "player_id"], how="left")
    feature_df["duels_lost"] = feature_df["duels_lost"].fillna(0).astype(int)

    return feature_df.fillna(0)


def extract_stamina_features(events_df: pd.DataFrame) -> pd.DataFrame:
    # Filter only "Carry" events
    carry_df = events_df[events_df["type"].apply(lambda x: isinstance(x, dict) and x.get("name") == "Carry")].copy()

    # Extract start and end locations
    carry_df["start_x"] = carry_df["location"].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else None)
    carry_df["end_x"] = carry_df["carry"].apply(lambda c: c.get("end_location")[0] if isinstance(c, dict) and "end_location" in c else None)

    # Drop rows with missing values
    carry_df = carry_df.dropna(subset=["start_x", "end_x"])

    # Compute distance
    carry_df["distance"] = (carry_df["end_x"] - carry_df["start_x"])**2
    carry_df["distance"] += 0  # skip y-axis to simplify stamina idea
    carry_df["distance"] = carry_df["distance"]**0.5

    # Aggregate features
    stamina_df = carry_df.groupby(["match_id", "player_id"]).agg({
        "distance": "sum",
        "start_x": "mean",
        "end_x": "mean"
    }).reset_index()

    # Rename columns
    stamina_df.rename(columns={
        "distance": "total_running_distance",
        "start_x": "stamina_start",
        "end_x": "stamina_end"
    }, inplace=True)

    return stamina_df


def add_match_context_features(df):
    # Ensure goals are available
    df["team_score"] = df["team"].apply(lambda x: x.get("score", 0) if isinstance(x, dict) else 0)
    df["opponent_score"] = df["possession_team"].apply(lambda x: x.get("score", 0) if isinstance(x, dict) else 0)

    df["score_margin"] = df["team_score"] - df["opponent_score"]
    df["team_losing"] = (df["score_margin"] < 0).astype(int)

    return df

def add_timing_features(df):
    df["minute"] = df["minute"].fillna(0)

    df["window_45_60"] = ((df["minute"] >= 45) & (df["minute"] < 60)).astype(int)
    df["window_60_75"] = ((df["minute"] >= 60) & (df["minute"] < 75)).astype(int)
    df["window_75_90"] = ((df["minute"] >= 75)).astype(int)

    return df

def add_position_feature(df):
    df["position_name"] = df["position"].apply(lambda x: x.get("name") if isinstance(x, dict) else "Unknown")
    return df

def compute_15_min_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_type"] = df["type"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
    df["timestamp"] = pd.to_timedelta(df["timestamp"])

    context_features = []

    for (match_id, player_id), group in df.groupby(["match_id", "player_id"]):
        group = group.sort_values("timestamp")
        if len(group) == 0:
            continue

        last_time = group["timestamp"].max()
        cutoff_time = last_time - pd.to_timedelta("00:15:00")
        recent = group[group["timestamp"] >= cutoff_time]

        context_features.append({
            "match_id": match_id,
            "player_id": player_id,
            "passes_last_15_minute": (recent["event_type"] == "Pass").sum()
        })

    return pd.DataFrame(context_features)


def extract_event_counts(df, event_type, time_window=15):
    """
    Count number of specific events per player in a rolling window.
    """
    df = df.copy()
    df['event_type'] = df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    df = df[df['event_type'] == event_type]
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
    agg = df.groupby(['match_id', 'player_id'])['minute'].apply(lambda x: (x <= time_window).sum()).reset_index()
    agg = agg.rename(columns={'minute': f"{event_type.lower()}_count_{time_window}min"})
    return agg


def extract_performance_deltas(df):
    df = df.copy()

    # Flatten nested match_id and player
    df["match_id"] = df["match_id"].apply(lambda x: x["id"] if isinstance(x, dict) and "id" in x else x)
    df["player_id"] = df["player"].apply(lambda x: x["id"] if isinstance(x, dict) and "id" in x else None)

    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove match_id and player_id from aggregation if they're in numeric columns
    numeric_cols = [col for col in numeric_cols if col not in ["match_id", "player_id"]]

    agg = df.groupby(["match_id", "player_id"])[numeric_cols].sum().reset_index()

    return agg


