import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve 
from sklearn.preprocessing import StandardScaler

# ========= Setup =========
input_dir = "outputs/off_def"
output_dir = "outputs/off_def/error_analysis"
os.makedirs(output_dir, exist_ok=True)

model_names = ["LogisticRegression", "SVM", "RandomForest", "XGBoost", "ANN"]

# ========= Load original dataset =========
df_raw = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df_raw = df_raw[df_raw['should_be_subbed_y'].isin([0, 1])]
df_raw['label'] = df_raw['should_be_subbed_y'].astype(int)

# Metadata columns
meta_cols = ['match_id', 'player_id', 'position_name', 'minute', 'team_name',
             'player_name', 'event_type', 'Own Goal For', 'Own Goal Against']
meta_cols = [col for col in meta_cols if col in df_raw.columns]
df_meta = df_raw[meta_cols + ['label']].reset_index(drop=True)

# Drop unnecessary columns
drop_cols = [c for c in ['should_be_subbed_x', 'should_be_subbed_y', 'label'] if c in df_raw.columns]
df_cleaned = df_raw.drop(columns=meta_cols + drop_cols, errors='ignore')
X_full = df_cleaned.copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ========= Loop through Models =========
for model_name in model_names:
    print(f"\n--- Error Analysis for {model_name} ---")

    pred_path = f"{input_dir}/predictions_{model_name}.csv"
    if not os.path.exists(pred_path):
        print(f"Missing: {pred_path}")
        continue

    df_pred = pd.read_csv(pred_path)

    # Auto-detect column names
    column_map = {}
    for col in df_pred.columns:
        col_lower = col.lower()
        if 'true' in col_lower:
            column_map['true'] = col
        elif 'pred' in col_lower and 'prob' not in col_lower:
            column_map['pred'] = col
        elif 'prob' in col_lower:
            column_map['prob'] = col

    if not {'true', 'pred', 'prob'}.issubset(column_map):
        print(f"Missing required columns in {model_name}. Found: {df_pred.columns}")
        continue

    df_pred = df_pred.rename(columns={
        column_map['true']: 'true',
        column_map['pred']: 'pred',
        column_map['prob']: 'prob'
    })

    # Attach metadata
    if len(df_pred) != len(df_meta):
        print(f"Row mismatch: {model_name}. Skipping.")
        continue

    df_pred = pd.concat([df_meta, df_pred], axis=1)
    df_pred.to_csv(f"{output_dir}/predictions_with_meta_{model_name}.csv", index=False)

    # Classification Report
    report = classification_report(df_pred["true"], df_pred["pred"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/classification_report_{model_name}.csv")

    # Confusion Matrix
    cm = confusion_matrix(df_pred["true"], df_pred["pred"])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.png")
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(df_pred["true"], df_pred["prob"])
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve_{model_name}.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(df_pred["true"], df_pred["prob"])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f}) - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve_{model_name}.png")
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(df_pred["true"], df_pred["prob"], n_bins=10, strategy='uniform')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"Calibration Curve - {model_name}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curve_{model_name}.png")
    plt.close()

    # False Positives and False Negatives
    df_fp = df_pred[(df_pred["true"] == 0) & (df_pred["pred"] == 1)].sort_values(by="prob", ascending=False)
    df_fn = df_pred[(df_pred["true"] == 1) & (df_pred["pred"] == 0)].sort_values(by="prob", ascending=True)
    df_fp.to_csv(f"{output_dir}/false_positives_{model_name}.csv", index=False)
    df_fn.to_csv(f"{output_dir}/false_negatives_{model_name}.csv", index=False)

    # Error Heatmap: Position vs. Minute
    error_df = df_pred[df_pred["true"] != df_pred["pred"]]
    if "position_name" in error_df.columns and "minute" in error_df.columns:
        heatmap_data = error_df.groupby(["position_name", "minute"]).size().unstack(fill_value=0)
        plt.figure(figsize=(8, 6))
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, cmap="Reds")
            plt.title(f"Heatmap for {model_name}")
            plt.savefig(f"{output_dir}/heatmap_{model_name}.png")
            plt.close()
        else:
            print(f"Skipped heatmap for {model_name} (no data)")


        
        plt.title(f"Error Heatmap (Position vs Minute) - {model_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_heatmap_position_minute_{model_name}.png")
        plt.close()

    print(f"Saved error analysis for {model_name}")
