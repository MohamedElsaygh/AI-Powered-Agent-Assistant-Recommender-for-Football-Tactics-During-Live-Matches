import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Output directories
os.makedirs("outputs/off_def", exist_ok=True)

# ========== 1. Load Data ==========
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df = df[df['should_be_subbed_y'].isin([0, 1])]
df['label'] = df['should_be_subbed_y'].astype(int)
if 'score_margin' not in df.columns:
    df['score_margin'] = df['Own Goal For'] - df['Own Goal Against']

X = df.drop(columns=['match_id', 'player_id', 'position_name',
                     'should_be_subbed_x', 'should_be_subbed_y', 'label'])
y = df['label']
groups = df['match_id']

drop_feats = ['duplicate_feature1', 'redundant_feature2']
X = X.drop(columns=[col for col in drop_feats if col in X.columns])

# ========== 2. Define Models ==========
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                           class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False,
                             scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
                             n_estimators=100, max_depth=3, learning_rate=0.03),
    "SVM": SVC(C=0.1, kernel='linear', class_weight='balanced', probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, max_iter=1000, early_stopping=True)
}

results = []
cv_scores = {}
final_scores = {}
logo = LeaveOneGroupOut()
# ========== 3. Train & Evaluate ==========
for model_name, model in models.items():
    print(f"\nRunning LOGO CV for {model_name}...")
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        sm = SMOTE(random_state=fold)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        pipe.fit(X_train_res, y_train_res)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        fold_metrics.append((acc, f1))

    accs, f1s = zip(*fold_metrics)
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)

    print(f"Mean Accuracy: {mean_acc:.3f} | Mean Macro F1: {mean_f1:.3f}")
    results.append((model_name, mean_acc, mean_f1))
    cv_scores[model_name] = mean_f1

    # Final model on all data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    pipe.fit(X_res, y_res)
    joblib.dump(pipe, f"outputs/off_def/{model_name}_final.pkl")

    y_proba = pipe.predict_proba(X)[:, 1]

    # Threshold tuning using PR AUC
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    best_threshold_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls + 1e-8))
    best_thresh = thresholds[best_threshold_idx]
    print(f"{model_name} PR AUC: {pr_auc:.3f} | Best Threshold: {best_thresh:.2f}")

    y_pred_thresh = (y_proba >= best_thresh).astype(int)
    final_f1 = f1_score(y, y_pred_thresh, average='macro')
    final_scores[model_name] = final_f1

    # Save predictions
    pred_df = df[['match_id', 'player_id']].copy()
    pred_df['true_label'] = y
    pred_df['predicted_proba'] = y_proba
    pred_df['predicted_label'] = y_pred_thresh
    pred_df.to_csv(f"outputs/off_def/predictions_{model_name}.csv", index=False)
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_thresh)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"outputs/off_def/cm_{model_name}.png")
    plt.close()

    # SHAP only for tree-based models
    if model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.Explainer(pipe.named_steps['model'], X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.title(f'SHAP Summary: {model_name}')
        plt.savefig(f"outputs/off_def/shap_{model_name}.png")
        plt.close()

# ========== 4. Save Comparison Table ==========
df_results = pd.DataFrame(results, columns=['Model', 'Mean_Accuracy', 'Mean_Macro_F1'])
df_results.sort_values(by='Mean_Macro_F1', ascending=False, inplace=True)
df_results.to_csv("outputs/off_def/model_comparison_LOGO.csv", index=False)
print("Model comparison saved to outputs/off_def/model_comparison_LOGO.csv")

# ========== 5. CV vs Final F1 Plot ==========
plt.figure(figsize=(8, 5))
models_list = list(cv_scores.keys())
cv_f1 = [cv_scores[m] for m in models_list]
final_f1 = [final_scores[m] for m in models_list]

x = np.arange(len(models_list))
width = 0.35
plt.bar(x - width/2, cv_f1, width, label='LOGO CV F1')
plt.bar(x + width/2, final_f1, width, label='Full Model Tuned F1')
plt.xticks(x, models_list, rotation=45)
plt.ylabel("Macro F1 Score")
plt.title("LOGO CV vs Full Tuned Model F1")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/off_def/cv_vs_final_f1.png")
plt.close()
