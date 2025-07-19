import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

# === Load and preprocess dataset ===
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df["should_be_subbed"] = df["should_be_subbed_y"]
leak_features = ["passes_last_15_minute", "duels_lost"]
drop_cols = ["should_be_subbed_x", "should_be_subbed_y", "match_id", "player_id", "position_name", "index"]
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=["should_be_subbed"] + leak_features)
y = df["should_be_subbed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# === Define models and grids ===
models = {
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    "ANN": MLPClassifier(max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

param_grids = {
    "RandomForest": {"n_estimators": [100], "max_depth": [10], "min_samples_split": [2], "min_samples_leaf": [1]},
    "LogisticRegression": {"C": [1], "penalty": ["l2"], "solver": ["lbfgs"]},
    "SVM": {"C": [1], "gamma": ["scale"], "kernel": ["rbf"]},
    "ANN": {"hidden_layer_sizes": [(64, 32)], "activation": ["relu"], "solver": ["adam"], "alpha": [0.0001]},
    "XGBoost": {"n_estimators": [100], "max_depth": [6], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8]}
}

# === Setup directories ===
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
results = []
feature_ranks = {}

# === Train and evaluate ===
for name, model in models.items():
    print(f"\n🔍 Tuning and training {name}")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)
    best_model = grid.best_estimator_
    
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(X_train_resampled, y_train_resampled)
    y_probs = calibrated.predict_proba(X_test)[:, 1]

    prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    y_pred = (y_probs >= best_thresh).astype(int)

    auc = roc_auc_score(y_test, y_probs)
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    acc = report["accuracy"]

    joblib.dump(calibrated, f"models/{name}_model.pkl")
    with open(f"models/{name}_threshold.txt", "w") as f:
        f.write(str(best_thresh))

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/roc_curve_{name}.png")
    plt.close()

    report_df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(report_df, annot=True, cmap="Blues")
    plt.title(f"Classification Report - {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/report_heatmap_{name}.png")
    plt.close()

    # Feature Importances + SHAP
    if name in ["RandomForest", "XGBoost"]:
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances.sort_values(ascending=False).plot(kind="bar", title=f"{name} - Feature Importances")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/feature_importance_{name}.png")
        plt.close()

        feature_ranks[name] = importances.sort_values(ascending=False)

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        shap.summary_plot(shap_values_to_plot, X_test, show=False)

        plt.tight_layout()
        plt.savefig(f"outputs/plots/shap_summary_{name}.png")
        plt.close()

    results.append([name, acc, macro_f1, auc, best_thresh])

# === Save model comparison ===
comparison_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Macro_F1", "ROC_AUC", "Best_Threshold"])
comparison_df.to_csv("outputs/model_comparison.csv", index=False)

# === Save averaged feature importances ===
avg_importance = pd.concat(feature_ranks.values(), axis=1).mean(axis=1).sort_values(ascending=False)
avg_importance.to_frame("MeanImportance").to_csv("outputs/feature_importance_ranking.csv")

print("✅ All models trained and evaluated. SHAP and importances saved.")

# === Reduced Feature Models ===

# === Load Feature Importance ===
feature_ranks = pd.read_csv("outputs/feature_importance_ranking.csv", index_col=0)
top_n = 10
top_features = feature_ranks.head(top_n).index.tolist()

# === Load Dataset ===
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df["should_be_subbed"] = df["should_be_subbed_y"]
leak_features = ["passes_last_15_minute", "duels_lost"]
drop_cols = ["should_be_subbed_x", "should_be_subbed_y", "match_id", "player_id", "position_name", "index"]
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=["should_be_subbed"] + leak_features)
X = X[top_features]
y = df["should_be_subbed"]

# === Split and Resample ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# === Define Reduced Feature Models ===
models = {
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             use_label_encoder=False, eval_metric='logloss', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
}

results = []

# === Train, Calibrate, Evaluate, Save ===
for name, model in models.items():
    print(f"🔁 Retraining {name} with top {top_n} features")

    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_train_resampled, y_train_resampled)

    y_probs = calibrated_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    y_pred = (y_probs >= best_thresh).astype(int)

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"Confusion Matrix - {name} (Reduced)")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/confusion_matrix_{name}_reduced.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.title(f"ROC Curve - {name} (Reduced)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/roc_curve_{name}_reduced.png")
    plt.close()

    joblib.dump(calibrated_model, f"models/{name}_reduced.pkl")
    with open(f"models/{name}_threshold_reduced.txt", "w") as f:
        f.write(str(best_thresh))

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]

    results.append({
        "Model": name + " (Reduced)",
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc_score,
        "Best_Threshold": best_thresh
    })

# === Export Comparison ===
results_df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/model_comparison_reduced.csv", index=False)

print("✅ Reduced models trained and results saved.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load uploaded comparison files
full_model_df = pd.read_csv("outputs/model_comparison.csv")
reduced_model_df = pd.read_csv("outputs/model_comparison_reduced.csv")

# Add a column to indicate model type
full_model_df["Type"] = "Full"
reduced_model_df["Type"] = "Reduced"

# Combine both
combined_df = pd.concat([full_model_df, reduced_model_df])

# Visual comparison plots
metrics = ["Accuracy", "Macro_F1", "ROC_AUC"]
plots = {}

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=combined_df, x="Model", y=metric, hue="Type")
    plt.title(f"{metric} Comparison: Full vs Reduced Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = f"outputs/{metric.lower()}_comparison.png"
    plt.savefig(plot_path)
    plots[metric] = plot_path
    plt.close()

combined_df_sorted = combined_df.sort_values(by="Macro_F1", ascending=False)
print(plots)
