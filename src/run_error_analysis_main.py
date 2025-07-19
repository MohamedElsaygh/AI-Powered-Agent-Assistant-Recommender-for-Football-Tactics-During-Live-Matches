import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
X_test = joblib.load(
    r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\outputs\20250719_022825\X_test.pkl"
)
y_test = joblib.load(
    r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\outputs\20250719_022825\y_test.pkl"
)

test_df = pd.read_csv(
    r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\outputs\20250719_022825\final_df.csv"
)

# Ensure index alignment
X_test = pd.DataFrame(X_test).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Output folder
os.makedirs("outputs/error_analysis", exist_ok=True)

# Model info with correct paths
base_model_dir = r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\outputs\20250719_022825\models"

model_info = {
    "LogisticRegression": {"path": os.path.join(base_model_dir, "logreg.pkl"), "threshold": os.path.join(base_model_dir, "logreg_threshold.txt")},
    "RandomForest": {"path": os.path.join(base_model_dir, "tuned_rf.pkl"), "threshold": os.path.join(base_model_dir, "tuned_rf_threshold.txt")},
    "XGBoost": {"path": os.path.join(base_model_dir, "xgboost.pkl"), "threshold": os.path.join(base_model_dir, "xgboost_threshold.txt")},
    "ANN": {"path": os.path.join(base_model_dir, "ann.pkl"), "threshold": None},
    "SVM": {"path": os.path.join(base_model_dir, "svm.pkl"), "threshold": os.path.join(base_model_dir, "svm_threshold.txt")},
    "KNN": {"path": os.path.join(base_model_dir, "knn.pkl"), "threshold": None},
    "Stacked": {"path": os.path.join(base_model_dir, "stacked.pkl"), "threshold": None}
}

results = []

for model_name, info in model_info.items():
    model_path = info["path"]
    threshold_path = info["threshold"]

    try:
        model = joblib.load(model_path)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        else:
            probs = model.predict(X_test)

        threshold = 0.5
        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())

        preds = (probs >= threshold).astype(int)

        # Align indices
        preds = pd.Series(preds, index=y_test.index)
        probs = pd.Series(probs, index=y_test.index)

        # Metrics
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"outputs/error_analysis/classification_report_{model_name}_main.csv")

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix (main)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/error_analysis/confusion_matrix_{model_name}_main.png")
        plt.close()

        # Misclassified rows
        misclassified_idx = y_test.index[y_test != preds]
        misclassified = test_df.loc[misclassified_idx].copy()
        misclassified["true"] = y_test.loc[misclassified_idx].values
        misclassified["pred"] = preds.loc[misclassified_idx].values
        misclassified["prob"] = probs.loc[misclassified_idx].values
        misclassified.to_csv(f"outputs/error_analysis/misclassified_{model_name}_main.csv", index=False)

        print(f"\n--- {model_name} ---")
        cols = ["true", "pred", "prob"]
        if "player_name" in misclassified.columns:
            cols.insert(0, "player_name")
        elif "player" in misclassified.columns:
            cols.insert(0, "player")
        print(misclassified.sort_values("prob", ascending=False).head(5)[cols])


        results.append({
            "model": model_name,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"]
        })

    except Exception as e:
        print(f"[ERROR] {model_name}: {e}")

# Save summary
summary_df = pd.DataFrame(results)
summary_df.to_csv("outputs/error_analysis/error_summary_main.csv", index=False)
print("\nSaved all error analysis to outputs/error_analysis/")
print(y_test.value_counts(normalize=True))
