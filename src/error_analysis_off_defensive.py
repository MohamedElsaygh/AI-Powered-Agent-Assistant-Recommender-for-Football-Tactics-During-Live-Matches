import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load model and test data
model = joblib.load("best_model_offensive_defensive.pkl")
X_test = pd.read_csv("X_test_off_def.csv")

# Load full dataset
df_full = pd.read_csv("data/processed/should_be_subbed_dataset.csv")

# Ensure alignment and filtering are done properly
df_full_filtered = df_full[df_full['should_be_subbed_y'].isin(['offensive', 'defensive'])].copy()

# Align based on index intersection
X_test = X_test.loc[X_test.index.intersection(df_full_filtered.index)]
y_true = df_full_filtered.loc[X_test.index, 'should_be_subbed_y'].map({'offensive': 1, 'defensive': 0}).values


# 5. Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Prob for 'offensive'

# 6. Save classification report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("off_def_classification_report.csv")

# 7. Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Defensive", "Offensive"],
            yticklabels=["Defensive", "Offensive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Offensive vs Defensive")
plt.tight_layout()
plt.savefig("confusion_matrix_off_def.png")
plt.close()

# 8. Save misclassified samples
misclassified = X_test.copy()
misclassified['true_label'] = y_true
misclassified['predicted_label'] = y_pred
misclassified['predicted_proba'] = y_proba
misclassified['is_misclassified'] = misclassified['true_label'] != misclassified['predicted_label']
misclassified[misclassified['is_misclassified']].to_csv("misclassified_off_def.csv", index=False)

# 9. Top misclassifications
top_fp = misclassified[(misclassified['true_label'] == 0) & (misclassified['predicted_label'] == 1)]
top_fn = misclassified[(misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 0)]

top_fp.sort_values(by='predicted_proba', ascending=False).head(10).to_csv("top_false_positives_off_def.csv", index=False)
top_fn.sort_values(by='predicted_proba').head(10).to_csv("top_false_negatives_off_def.csv", index=False)

# 10. Optional: print dropped labels for diagnosis
dropped = y_raw[~y_raw.isin(['offensive', 'defensive'])]
print("Dropped labels (not offensive/defensive):", len(dropped))
print(dropped.value_counts())

print("Error analysis complete.")
