import os
import json
import pandas as pd

def load_substitution_labels(data_path):
    lineups_path = os.path.join(data_path, "lineups")
    files = [f for f in os.listdir(lineups_path) if f.endswith(".json")]

    labels = []

    for file in files:
        match_id = int(file.replace(".json", ""))
        with open(os.path.join(lineups_path, file), "r", encoding="utf-8") as f:
            try:
                teams = json.load(f)
                for team in teams:
                    for player in team.get("lineup", []):
                        positions = player.get("positions", [])
                        if positions and isinstance(positions[-1], dict):
                            substituted = positions[-1].get("substituted", False)
                        else:
                            substituted = False


                        labels.append({
                        "match_id": match_id,
                        "player_id": player["player_id"],
                        "substituted": substituted
    })

            except json.JSONDecodeError:
                print(f"❌ Could not parse {file}")

    df_labels = pd.DataFrame(labels)
    print(f"✅ Loaded substitution labels: {len(df_labels)}")
    return df_labels

def extract_substitution_labels_from_events(events_df):
    labels = []

    for _, row in events_df.iterrows():
        if row["type"]["name"] == "Substitution" and isinstance(row.get("player"), dict):
            labels.append({
                "match_id": row["match_id"],
                "player_id": row["player"]["id"],
                "substituted": True
            })

    return pd.DataFrame(labels)

