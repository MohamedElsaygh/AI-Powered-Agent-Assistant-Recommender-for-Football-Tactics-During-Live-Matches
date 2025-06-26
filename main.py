import os
from src.data_loader import load_event_data
from src.explorer import count_event_types
from src.features import extract_player_features
from src.model import train_baseline_model
from src.substitution_predictor import extract_player_match_stats
from src.label_builder import extract_substitution_labels_from_events
from src.feature_engineering import extract_player_match_features
from src.model_trainer import train_and_evaluate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.model import train_random_forest
from src.model import train_baseline_model
from src.model import train_random_forest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model import tune_random_forest
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.visualizations import plot_confusion_matrix, plot_feature_importance




DATA_PATH = r"H:\AI AS\Individual Project\AI-Agent-For-Football-Tactics-During-Live-Matches\data"

# Load event data
df = load_event_data(DATA_PATH, max_files=100)
df["player_id"] = df["player"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
print(f"✅ Loaded {len(df)} events")

# Show event type distribution
print("Top 10 event types:")
print(count_event_types(df).head(10))

# Feature Engineering (for model)
features_df = extract_player_match_features(df)
print("✅ Feature DataFrame shape:", features_df.shape)

# Build label dataset from substitution events
# Step 1: All players
all_players = df[["match_id", "player_id"]].drop_duplicates()
all_players["player_id"] = all_players["player_id"].astype("Int64")
all_players["match_id"] = all_players["match_id"].astype("Int64")

# Step 2: Get substitution events
true_subs = extract_substitution_labels_from_events(df)
true_subs["substituted"] = True
true_subs["player_id"] = true_subs["player_id"].astype("Int64")
true_subs["match_id"] = true_subs["match_id"].astype("Int64")

# Step 3: Merge (fill False where no match)
labels_df = all_players.merge(true_subs, on=["match_id", "player_id"], how="left")
labels_df["substituted"] = labels_df["substituted"].fillna(False)
labels_df = labels_df[["match_id", "player_id", "substituted"]]
print("✅ Loaded substitution labels:", len(labels_df))

# Merge features + labels
features_df["player_id"] = features_df["player_id"].astype("Int64")
features_df["match_id"] = features_df["match_id"].astype("Int64")

final_df = features_df.merge(labels_df, on=["match_id", "player_id"], how="inner")
print("✅ Final merged shape:", final_df.shape)
print(final_df["substituted"].value_counts())
drop_cols = ['Own Goal For', 'Own Goal Against', '50/50', 'Bad Behaviour']
final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

# Check correlation with the label
correlations = final_df.corr(numeric_only=True)["substituted"].sort_values(ascending=False)
print("\n🔍 Feature correlations with 'substituted':")
print(correlations)

leakage_cols = ["Substitution", "Player On", "Player Off"]
X = final_df.drop(columns=["substituted"] + leakage_cols)

#  Drop non-numeric columns
X = X.select_dtypes(include='number')
y = final_df["substituted"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 1: Split into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE only on training data
X_train_val_resampled, y_train_val_resampled = smote.fit_resample(X_train_val, y_train_val)


# Step 2: Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg_cv_scores = cross_val_score(logreg, X_train_val, y_train_val, cv=cv, scoring="f1")
logreg.fit(X_train_val, y_train_val)
y_pred_logreg = logreg.predict(X_test)
logreg_report = classification_report(y_test, y_pred_logreg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_cv_scores = cross_val_score(rf, X_train_val, y_train_val, cv=cv, scoring="f1")
rf.fit(X_train_val, y_train_val)
y_pred_rf = rf.predict(X_test)
rf_report = classification_report(y_test, y_pred_rf)

print("Logistic Regression CV F1:", logreg_cv_scores.mean())
print("Random Forest CV F1:", rf_cv_scores.mean())
print("\n=== Logistic Regression Test Report ===\n", logreg_report)
print("\n=== Random Forest Test Report ===\n", rf_report)


# Train models and collect reports
model, baseline_report = train_baseline_model(X_scaled, y)
model2, resampled_logreg_report = train_baseline_model(X_resampled, y_resampled)
model3, random_forest_report = train_random_forest(X_resampled, y_resampled)

tuned_model, tuned_params, tuned_score = tune_random_forest(X_train_val_resampled, y_train_val_resampled)
best_rf = tuned_model

# Save all reports to file
os.makedirs("outputs/reports", exist_ok=True)

with open("outputs/reports/model_comparison.txt", "w") as f:
    f.write("=== Baseline Logistic Regression (scaled) ===\n")
    f.write(baseline_report + "\n\n")

    f.write("=== Resampled Logistic Regression ===\n")
    f.write(resampled_logreg_report + "\n\n")

    f.write("=== Random Forest on Resampled ===\n")
    f.write(random_forest_report + "\n\n")

# Save tuned model report
os.makedirs("outputs/reports", exist_ok=True)

with open("outputs/reports/tuned_model.txt", "w") as f:
    f.write("=== Tuned Random Forest Model ===\n")
    f.write("Best Parameters:\n")
    for k, v in tuned_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nBest Cross-Validation F1 Score: {tuned_score:.4f}\n")

# Evaluate on test set
y_test_pred = best_rf.predict(X_test)

print("\n=== Best Tuned Random Forest Test Report ===")
print(classification_report(y_test, y_test_pred))


os.makedirs("outputs/models", exist_ok=True)
joblib.dump(best_rf, "outputs/models/best_random_forest.pkl")
print("✅ Best Random Forest model saved.")


importances = best_rf.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)

# Confusion Matrix
plot_confusion_matrix(y_test, y_test_pred)

# Feature Importance
plot_feature_importance(best_rf, feature_names=X.columns.tolist())

