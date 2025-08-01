import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# =========================================
# 1. Load & Preprocess Data
# =========================================
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
print("Initial label distribution:\n", df['should_be_subbed_y'].value_counts())

# Filter valid labels
df = df[df['should_be_subbed_y'].isin([0, 1])]
df['label'] = df['should_be_subbed_y'].astype(int)

# Tactical context features (score margin etc.)
if 'score_margin' not in df.columns:
    df['score_margin'] = df['Own Goal For'] - df['Own Goal Against']

# Feature matrix
X = df.drop(columns=['match_id', 'player_id', 'position_name',
                     'should_be_subbed_x', 'should_be_subbed_y', 'label'])
y = df['label']
groups = df['match_id']  # Group by match

# =========================================
# 2. Train-Test Split using GroupKFold
# =========================================
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y, groups))  # first fold
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================================
# 3. Define Models & Hyperparameters
# =========================================
models = {
    "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight='balanced'),
                           {"C": [0.01, 0.1, 1, 10]}),
    "RandomForest": (RandomForestClassifier(class_weight='balanced', random_state=42),
                     {"n_estimators": [100, 200], "max_depth": [5, 8], "min_samples_split": [2, 5]}),
    "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=42,
                              scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
                {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
    "SVM": (SVC(class_weight='balanced', probability=True),
            {"C": [0.1, 1, 10], "kernel": ['rbf', 'linear']}),
    "ANN": (MLPClassifier(max_iter=1000, early_stopping=True),
            {"hidden_layer_sizes": [(32,), (64,), (128,)], "alpha": [0.0001, 0.001, 0.01]})
}

results = {}
best_model = None
best_score = 0
comparison_rows = []

# =========================================
# 4. Train & Evaluate
# =========================================
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    grid = GridSearchCV(pipe, {"model__" + k: v for k, v in params.items()},
                        scoring='f1_macro', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # CV score
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=3, scoring='f1_macro')
    mean_cv = np.mean(cv_scores)

    results[name] = {
        "model": grid.best_estimator_,
        "accuracy": acc,
        "f1_macro": f1,
        "cv_f1_macro": mean_cv,
        "report": classification_report(y_test, y_pred),
        "conf_matrix": cm
    }

    comparison_rows.append([name, acc, f1, mean_cv])
    print(f"=== {name} ===")
    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("Macro F1 Score:", f1)
    print("Mean CV F1:", mean_cv)
    print(results[name]["report"])
    print("Confusion Matrix:\n", cm)

    if f1 > best_score:
        best_score = f1
        best_model = grid.best_estimator_

# =========================================
# 5. Save Best Model & Comparison
# =========================================
joblib.dump(best_model, "best_model_offensive_defensive.pkl")
X_test.to_csv("X_test_off_def.csv", index=False)
comparison_df = pd.DataFrame(comparison_rows, columns=['Model', 'Test_Accuracy', 'Test_F1', 'CV_F1'])
comparison_df.to_csv("model_comparison_off_def.csv", index=False)
print("\nModel comparison saved to model_comparison_off_def.csv")

# =========================================
# 6. Confusion Matrices
# =========================================
for name in results:
    plt.figure(figsize=(5, 4))
    sns.heatmap(results[name]["conf_matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'conf_matrix_{name}_off_def.png')
    plt.close()

# =========================================
# 7. Threshold Calibration (Best Model)
# =========================================
if hasattr(best_model.named_steps['model'], "predict_proba"):
    y_probs = best_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"\nOptimal Threshold: {best_thresh}")
    y_thresh = (y_probs >= best_thresh).astype(int)
    print("Recalibrated Report:\n", classification_report(y_test, y_thresh))
    joblib.dump(best_thresh, "best_threshold.pkl")

# =========================================
# 8. SHAP Analysis for Tree Models
# =========================================
if any(tree_model in str(type(best_model.named_steps['model'])) for tree_model in ['RandomForest', 'XGBClassifier']):
    explainer = shap.TreeExplainer(best_model.named_steps['model'])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Offensive vs Defensive)")
    plt.tight_layout()
    plt.savefig("shap_summary_off_def.png")
    plt.show()