import os
import json
import pandas as pd

def load_event_data(data_path):
    event_data = []
    all_files = os.listdir(data_path)
    event_files = [f for f in all_files if f.endswith(".json") and f.replace(".json", "").isdigit()]

    for filename in event_files:
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                match_events = json.load(f)
                if isinstance(match_events, list) and len(match_events) > 0:
                    for event in match_events:
                        event['match_file'] = filename
                    event_data.extend(match_events)
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse: {filename}")

    df = pd.DataFrame(event_data)
    return df