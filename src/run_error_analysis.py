import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# === Paths ===
model_dir = "models"
threshold_dir = "models"
data_path = "data/processed/should_be_subbed_dataset.csv"
output_dir = "outputs/error_analysis"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(data_path)
df["should_be_subbed"] = df["should_be_subbed_y"]
drop_cols = ["should_be_subbed_x", "should_be_subbed_y", "match_id", "player_id", "position_name", "index"]
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=["should_be_subbed"])
y = df["should_be_subbed"]
_, X_test_full, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Models to evaluate ===
model_names = ["LogisticRegression", "RandomForest", "XGBoost", "ANN", "SVM"]
results = []

for name in model_names:
    print(f"Processing {name}...")
    try:
        # Load model
        model = joblib.load(f"{model_dir}/{name}_model.pkl")
        
        # Extract trained features
        if hasattr(model, "feature_names_in_"):
            trained_features = model.feature_names_in_
        else:
            trained_features = X.columns.intersection(X_test_full.columns)  # fallback

        # Load test set aligned with trained features
        X_test = X_test_full[trained_features]

        # Load threshold
        with open(f"{threshold_dir}/{name}_threshold.txt", "r") as f:
            threshold = float(f.read().strip())

        # Predict
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{name}.png")
        plt.close()

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/classification_report_{name}.csv")

        # Misclassified examples
        misclassified = X_test.copy()
        misclassified["true"] = y_test.values
        misclassified["pred"] = y_pred
        misclassified["proba"] = y_probs
        misclassified = misclassified[misclassified["true"] != misclassified["pred"]]
        misclassified.to_csv(f"{output_dir}/misclassified_{name}.csv", index=False)

        results.append((name, len(misclassified)))

    except Exception as e:
        print(f"Error in {name}: {e}")

# Save summary
pd.DataFrame(results, columns=["Model", "Num_Misclassified"]).to_csv(f"{output_dir}/error_summary.csv", index=False)
print("\n✅ Error analysis completed.")
