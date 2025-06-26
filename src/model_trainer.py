from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_and_evaluate(df, label_col="substituted", test_size=0.2, random_state=42):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Separate label and features
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Use only numeric features
    X = X.select_dtypes(include="number")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report3= classification_report(y_test, y_pred)
    print(report3)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/final_model.pkl")


    return model,report3
