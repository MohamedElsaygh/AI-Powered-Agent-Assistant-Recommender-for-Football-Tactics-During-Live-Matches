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

    return feature_df.fillna(0)


def extract_stamina_features(events_df: pd.DataFrame) -> pd.DataFrame:
    # Filter only "Carry" events
    carry_df = events_df[events_df["type"].apply(lambda x: isinstance(x, dict) and x.get("name") == "Carry")].copy()

    # Extract start location
    carry_df["start_x"] = carry_df["location"].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else None)
    carry_df["start_y"] = carry_df["location"].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else None)

    # Extract end location from nested 'carry' field
    carry_df["end_x"] = carry_df["carry"].apply(lambda c: c.get("end_location")[0] if isinstance(c, dict) and "end_location" in c else None)
    carry_df["end_y"] = carry_df["carry"].apply(lambda c: c.get("end_location")[1] if isinstance(c, dict) and "end_location" in c else None)

    # Drop rows with missing coordinates
    carry_df = carry_df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

    # Calculate distance
    carry_df["distance"] = ((carry_df["end_x"] - carry_df["start_x"])**2 + (carry_df["end_y"] - carry_df["start_y"])**2)**0.5

    # Aggregate total distance per match and player
    stamina_df = carry_df.groupby(["match_id", "player_id"])["distance"].sum().reset_index()
    stamina_df.rename(columns={"distance": "total_running_distance"}, inplace=True)

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

def compute_15_min_context(df):
    df["minute"] = df["minute"].fillna(0)
    context_stats = []

    for (match_id, player_id), group in df.groupby(["match_id", "player_id"]):
        max_minute = group["minute"].max()
        window_start = max(0, max_minute - 15)
        context_group = group[group["minute"] >= window_start]

        stats = {
            "match_id": match_id,
            "player_id": player_id,
            "context_event_count": len(context_group),
            "context_carry": (context_group["type_name"] == "Carry").sum(),
            "context_duel": (context_group["type_name"] == "Duel").sum(),
            "context_shot": (context_group["type_name"] == "Shot").sum()
        }
        context_stats.append(stats)

    return pd.DataFrame(context_stats)
