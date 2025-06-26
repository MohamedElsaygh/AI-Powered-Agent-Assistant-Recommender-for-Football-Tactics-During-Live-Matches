import os
import json
import pandas as pd

def load_event_data(data_path, max_files=10):
    all_events = []
    files = os.listdir(os.path.join(data_path, "events"))
    print("Files in events folder:", files)

    count = 0
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(data_path, "events", file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    match_events = json.load(f)
                    match_id = int(file.replace(".json", ""))
                    for event in match_events:
                        event["match_file"] = file
                        event["match_id"] = match_id   # ✅ Add match_id here
                    all_events.extend(match_events)
                    count += 1
            except json.JSONDecodeError:
                print(f"Failed to load {file}")
            if count >= max_files:
                break

    df = pd.DataFrame(all_events)
    print(f"✅ Loaded {len(df)} events from {count} files.")
    return df





def load_lineups(data_path):
    return [json.load(open(os.path.join(data_path, f), 'r', encoding='utf-8'))
            for f in os.listdir(data_path) if "lineup" in f.lower()]
