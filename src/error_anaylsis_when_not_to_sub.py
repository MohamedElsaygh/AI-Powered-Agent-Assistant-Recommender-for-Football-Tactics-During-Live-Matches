import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ====== Load data again ======
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df.drop(columns=['match_id', 'player_id', 'should_be_subbed_y'], errors='ignore', inplace=True)
df.dropna(subset=['should_be_subbed_x'], inplace=True)

y = df['should_be_subbed_x'].astype(int)
X = df.drop(columns=['should_be_subbed_x'], errors='ignore')

if 'position_name' in X.columns:
    X = pd.get_dummies(X, columns=['position_name'])

categorical_cols = [col for col in X.columns if col.startswith('position_name_')]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

important_features = numerical_cols + categorical_cols
X = X[important_features]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ====== Create outputs directory ======
os.makedirs("outputs/error_analysis_regularized", exist_ok=True)

# ====== Error Analysis ======
models = ['LogisticRegression', 'RandomForest']

for model_name in models:
    print(f"\nAnalyzing {model_name}")
    
    model_path = f"outputs/{model_name}_model_regularized.pkl"
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"outputs/error_analysis_regularized/cm_{model_name}.png")
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"outputs/error_analysis_regularized/classification_report_{model_name}.csv")

    # Misclassified samples
    errors_df = X_test.copy()
    errors_df['y_true'] = y_test.values
    errors_df['y_pred'] = y_pred
    errors_df['prob_class_1'] = y_prob
    errors_df = errors_df[errors_df['y_true'] != errors_df['y_pred']]
    errors_df.to_csv(f"outputs/error_analysis_regularized/misclassified_{model_name}.csv", index=False)

    # Per-class F1 bar plot
    f1_scores = {
        label: metrics["f1-score"]
        for label, metrics in report.items()
        if label in ["0", "1"]
    }
    plt.bar(f1_scores.keys(), f1_scores.values())
    plt.ylim(0, 1)
    plt.title(f"Per-Class F1 Scores: {model_name}")
    plt.ylabel("F1 Score")
    plt.savefig(f"outputs/error_analysis_regularized/f1_per_class_{model_name}.png")
    plt.close()

    print(f"Saved error analysis for {model_name}")
