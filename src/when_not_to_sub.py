import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========== Load and preprocess data ==========
df = pd.read_csv("data/processed/should_be_subbed_dataset.csv")
df.drop(columns=['match_id', 'player_id', 'should_be_subbed_y'], errors='ignore', inplace=True)
df.dropna(subset=['should_be_subbed_x'], inplace=True)

y = df['should_be_subbed_x'].astype(int)
X = df.drop(columns=['should_be_subbed_x'], errors='ignore')

if 'position_name' in X.columns:
    X = pd.get_dummies(X, columns=['position_name'])

categorical_cols = [col for col in X.columns if col.startswith('position_name_')]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Manual SHAP pruning
# You can replace this list with top N features after inspecting previous SHAP plots
important_features = numerical_cols + categorical_cols  # or custom selection

X = X[important_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [col for col in important_features if col not in categorical_cols]),
    ('cat', 'passthrough', [col for col in important_features if col in categorical_cols])
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ========== Models ==========
models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {
        'classifier__C': [0.01, 0.1, 1]
    }),
    'RandomForest': (RandomForestClassifier(class_weight='balanced'), {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5]
    }),
    'SVM': (SVC(probability=True, class_weight='balanced'), {
        'classifier__C': [0.1, 1],
        'classifier__kernel': ['linear', 'rbf']
    }),
    'MLP': (MLPClassifier(max_iter=300), {
        'classifier__hidden_layer_sizes': [(50,), (100,)],
        'classifier__alpha': [0.01, 0.1]
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {

        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3],
        'classifier__learning_rate': [0.01, 0.1]
    })
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, param_grid) in models.items():
    print(f"\nTraining {name}")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=skf, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, cv=3)
    calibrated_model.fit(X_train, y_train)

    y_pred = calibrated_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    val_f1 = cross_val_score(calibrated_model, X_train, y_train, scoring='f1_macro', cv=skf).mean()

    if name in ['RandomForest', 'XGBoost']:
        X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_transformed)

        shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(
            shap_values_class1,
            X_test_transformed,
            feature_names=important_features,
            show=False
        )
        os.makedirs("outputs", exist_ok=True)
        plt.title(f"SHAP summary: {name}")
        plt.savefig(f"outputs/shap_{name}.png")
        plt.close()

    results[name] = {
        'best_params': grid.best_params_,
        'accuracy': acc,
        'f1_macro': f1,
        'val_f1_cv': val_f1,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"{name} | Accuracy: {acc:.3f}, F1 (macro): {f1:.3f}")
    print("Best Params:", grid.best_params_)
    joblib.dump(calibrated_model, f"outputs/{name}_model.pkl")

# ========== Save results ==========
summary_df = pd.DataFrame.from_dict(results, orient='index')[['accuracy', 'f1_macro', 'val_f1_cv']]
summary_df.to_csv("outputs/model_comparison_final.csv")
print("Saved final model metrics to outputs/model_comparison_final.csv")

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


df.drop(columns=['match_id', 'player_id', 'should_be_subbed_y'], errors='ignore', inplace=True)
df.dropna(subset=['should_be_subbed_x'], inplace=True)

# Features and target
y = df['should_be_subbed_x'].astype(int)
X = df.drop(columns=['should_be_subbed_x'], errors='ignore')

# One-hot encode position
if 'position_name' in X.columns:
    X = pd.get_dummies(X, columns=['position_name'])

categorical_cols = [col for col in X.columns if col.startswith('position_name_')]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', 'passthrough', categorical_cols)
])

# Define regularized models and their tuned parameters
models = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'),
        {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l1', 'l2']}
    ),
    'RandomForest': (
        RandomForestClassifier(class_weight='balanced'),
        {'classifier__n_estimators': [50, 100],
         'classifier__max_depth': [3, 5, 10],
         'classifier__min_samples_leaf': [2, 5]}
    ),
    'MLP': (
        MLPClassifier(max_iter=500, early_stopping=True),
        {'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
         'classifier__alpha': [0.001, 0.01, 0.1]}
    )
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

results = {}
os.makedirs("outputs", exist_ok=True)

for name, (model, param_grid) in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring='f1_macro',
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    calibrated_model = CalibratedClassifierCV(best_model, cv=3)
    calibrated_model.fit(X_train, y_train)

    y_pred = calibrated_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    val_score = grid.best_score_

    results[name] = {
        'best_params': grid.best_params_,
        'accuracy': acc,
        'f1_macro': f1,
        'val_f1_cv': val_score,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    joblib.dump(calibrated_model, f"outputs/{name}_model_regularized.pkl")

# Save summary
summary_df = pd.DataFrame.from_dict(results, orient='index')[['accuracy', 'f1_macro', 'val_f1_cv']]
summary_df.to_csv("outputs/model_comparison_regularized.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Model setup ===
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

# === Fit ===
model.fit(X_train, y_train)

# === Predict probs & tune threshold ===
y_probs = model.predict_proba(X_test)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
threshold = 0.35  # Tuned threshold
y_pred = (y_probs >= threshold).astype(int)

# === Evaluation ===
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('outputs/error_analysis/classification_report_RandomForest_Weighted.csv')

# === Save misclassified samples ===
misclassified = X_test[y_pred != y_test].copy()
misclassified['true'] = y_test[y_pred != y_test]
misclassified['pred'] = y_pred[y_pred != y_test]
misclassified['prob'] = y_probs[y_pred != y_test]
misclassified.to_csv('outputs/error_analysis/misclassified_RandomForest_Weighted.csv', index=False)

# === Save model ===
joblib.dump(model, 'models/random_forest_weighted_model.pkl')

# === Confusion matrix plot ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: RandomForest (Weighted)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('outputs/error_analysis/cm_RandomForest_Weighted.png')
plt.close()

# === Per-class F1 bar plot ===
f1_class0 = f1_score(y_test, y_pred, pos_label=0)
f1_class1 = f1_score(y_test, y_pred, pos_label=1)
plt.bar([0, 1], [f1_class0, f1_class1])
plt.title('Per-Class F1 Scores: RandomForest (Weighted)')
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.ylim([0, 1])
plt.savefig('outputs/error_analysis/f1_per_class_RandomForest_Weighted.png')
plt.close()

# === Permutation Feature Importance ===
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='f1_macro')
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Permutation Feature Importance (Weighted RF)')
plt.tight_layout()
plt.savefig('outputs/error_analysis/permutation_importance_RandomForest_Weighted.png')
plt.close()

print("✅ Finished retraining RandomForest with class weights and threshold tuning.")
