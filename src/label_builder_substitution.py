import pandas as pd

def compute_fatigue_score(df: pd.DataFrame) -> pd.Series:
    stamina_drop = df["stamina_start"] - df["stamina_end"]
    return stamina_drop + df["minutes_played"]

def compute_mistake_score(df: pd.DataFrame) -> pd.Series:
    duel_losses = df.get("Duel:Lost", 0)
    bad_passes = df.get("Pass:Out of Play", 0) + df.get("Pass:Incomplete", 0)
    return duel_losses + bad_passes

def compute_performance_drop(df: pd.DataFrame) -> pd.Series:
    return df["xThreat_start"] - df["xThreat_end"]

def define_should_be_subbed(df: pd.DataFrame,
                             fatigue_threshold=70,
                             mistake_threshold=5,
                             performance_drop_threshold=0.1) -> pd.Series:
    fatigue = compute_fatigue_score(df)
    mistakes = compute_mistake_score(df)
    drop = compute_performance_drop(df)

    should_sub = (
        (fatigue >= fatigue_threshold) |
        (mistakes >= mistake_threshold) |
        (drop >= performance_drop_threshold)
    )
    return should_sub.astype(int)
