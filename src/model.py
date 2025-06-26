from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.model_selection import GridSearchCV

def train_baseline_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report= classification_report(y_test, y_pred)
    print(report)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/final_model.pkl")

    return model, report

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report2= classification_report(y_test, y_pred)
    print(report2)
    joblib.dump(model, "models/final_model.pkl")

    return model, report2

def tune_random_forest(X, y):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("\n✅ Best Parameters:", best_params)
    print("✅ Best CV F1 Score:", best_score)

    joblib.dump(best_model, "models/tuned_random_forest.pkl")

    return best_model, best_params, best_score