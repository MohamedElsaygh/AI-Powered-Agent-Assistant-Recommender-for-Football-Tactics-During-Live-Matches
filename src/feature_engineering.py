import pandas as pd

def extract_player_match_features(events_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["match_id", "player_id", "type", "timestamp", "position"]
    for col in required_cols:
        if col not in events_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create feature columns
    events_df["event_type"] = events_df["type"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    # Count events per player per match
    counts = events_df.groupby(["match_id", "player_id", "event_type"]).size().unstack(fill_value=0)

    # Estimate minutes played per player (based on earliest → latest event timestamps)
    time_df = events_df.groupby(["match_id", "player_id"])["timestamp"].agg(['min', 'max']).reset_index()
    time_df["minutes_played"] = (pd.to_timedelta(time_df["max"]) - pd.to_timedelta(time_df["min"])).dt.total_seconds() / 60

    # Extract last known position
    pos_df = events_df.dropna(subset=["position"]).groupby(["match_id", "player_id"])["position"].last().reset_index()
    pos_df["position_name"] = pos_df["position"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    # Merge all features
    feature_df = counts.reset_index()
    feature_df = feature_df.merge(time_df[["match_id", "player_id", "minutes_played"]], on=["match_id", "player_id"], how="left")
    feature_df = feature_df.merge(pos_df[["match_id", "player_id", "position_name"]], on=["match_id", "player_id"], how="left")

    return feature_df.fillna(0)
