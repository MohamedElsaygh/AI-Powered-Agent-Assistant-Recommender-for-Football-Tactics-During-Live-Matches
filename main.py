import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.data_loader import load_event_data
from src.label_builder import extract_substitution_labels_from_events
from src.feature_engineering import (
    extract_player_match_features,
    extract_stamina_features,
    add_match_context_features,
    add_timing_features,
    add_position_feature,
    compute_15_min_context
)
from src.visualizations import plot_confusion_matrix, plot_feature_importance, plot_pca_3d
from src.model import train_knn, train_svm, train_ann, tune_random_forest, train_xgboost
from src.explainability import explain_model_shap

# === Setup ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/{timestamp}"
os.makedirs(f"{output_dir}/models", exist_ok=True)
os.makedirs(f"{output_dir}/plots", exist_ok=True)
os.makedirs(f"{output_dir}/reports", exist_ok=True)

# === Load and preprocess ===
DATA_PATH = r"H:\AI AS\Individual Project\AI-Agent-For-Football-Tactics-During-Live-Matches\data"
df = load_event_data(DATA_PATH, max_files=100)
df["player_id"] = df["player"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
df["type_name"] = df["type"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
df = add_match_context_features(df)
df = add_timing_features(df)
df = add_position_feature(df)

features_df = extract_player_match_features(df)
context_df = compute_15_min_context(df)
features_df = features_df.merge(context_df, on=["match_id", "player_id"], how="left").fillna(0)
features_df = pd.get_dummies(features_df.drop(columns=["minutes_played"], errors='ignore'), columns=["position_name"], dummy_na=True)

stamina_df = extract_stamina_features(df)
features_df = features_df.merge(stamina_df, on=["match_id", "player_id"], how="left").fillna(0)

# === Build labels ===
all_players = df[["match_id", "player_id"]].drop_duplicates().astype("Int64")
true_subs = extract_substitution_labels_from_events(df)
true_subs["substituted"] = True
true_subs = true_subs.astype({"player_id": "Int64", "match_id": "Int64"})
labels_df = all_players.merge(true_subs, on=["match_id", "player_id"], how="left")
labels_df["substituted"] = labels_df["substituted"].fillna(False).infer_objects(copy=False)

# === Final merge and preprocessing ===
features_df = features_df.astype({"player_id": "Int64", "match_id": "Int64"})
final_df = features_df.merge(labels_df, on=["match_id", "player_id"], how="inner")
drop_cols = ['Own Goal For', 'Own Goal Against', '50/50', 'Bad Behaviour', 'Substitution', 'Player On', 'Player Off']
final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

X = final_df.drop(columns=["substituted"]).select_dtypes(include="number")
y = final_df["substituted"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split + Resample ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === Model training ===
logreg = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train_resampled, y_train_resampled)
logreg_probs = logreg.predict_proba(X_test)[:, 1]
logreg_preds = (logreg_probs >= 0.35).astype(int)

tuned_rf, _, _ = tune_random_forest(X_train_resampled, y_train_resampled)
rf_preds = tuned_rf.predict(X_test)

xgb_model, _ = train_xgboost(X_train_resampled, y_train_resampled, X_test, y_test)
xgb_preds = xgb_model.predict(X_test)

ann_model, _ = train_ann(X_train_resampled, y_train_resampled, X_test, y_test)
ann_preds = ann_model.predict(X_test)

svm_model, _ = train_svm(X_train_resampled, y_train_resampled, X_test, y_test)
svm_preds = svm_model.predict(X_test)

knn_model, _ = train_knn(X_train_resampled, y_train_resampled, X_test, y_test)
knn_preds = knn_model.predict(X_test)

# === Stacking ensemble ===
stacked_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("rf", tuned_rf), ("ann", ann_model)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
).fit(X_train_resampled, y_train_resampled)
stacked_preds = stacked_model.predict(X_test)

# === Evaluation ===
models = [logreg, tuned_rf, xgb_model, ann_model, svm_model, knn_model, stacked_model]
names = ["LogReg", "Tuned RF", "XGBoost", "ANN", "SVM", "KNN", "Stacked"]
predictions = [logreg_preds, rf_preds, xgb_preds, ann_preds, svm_preds, knn_preds, stacked_preds]

best_model = None
best_score = 0

with open(f"{output_dir}/reports/model_comparison.txt", "w") as f:
    for model, name, preds in zip(models, names, predictions):
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
        f1_true = f1_score(y_test, preds, pos_label=True, zero_division=0)

        f.write(f"\n=== {name} ===\n")
        f.write(f"Accuracy: {acc:.3f}\nMacro F1: {f1_macro:.3f}\nF1 (True): {f1_true:.3f}\n")
        f.write(classification_report(y_test, preds, zero_division=0))

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f} | Macro F1: {f1_macro:.3f} | F1 (True): {f1_true:.3f}")

        plot_confusion_matrix(y_test, preds, title=f"{name} Confusion Matrix", save_path=f"{output_dir}/plots/conf_matrix_{name}.png")
        joblib.dump(model, f"{output_dir}/models/{name.lower().replace(' ', '_')}.pkl")

        if acc > best_score:
            best_score = acc
            best_model = name

print(f"\n🏆 Best Model: {best_model} (Accuracy = {best_score:.3f})")

# === Extras ===
plot_feature_importance(tuned_rf, feature_names=X.columns.tolist())
plot_pca_3d(X_scaled, y)
explain_model_shap(xgb_model, X_test, X.columns.tolist())
final_df.to_csv(f"{output_dir}/final_df.csv", index=False)
