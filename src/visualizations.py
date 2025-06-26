import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=["Not Subbed", "Subbed"], title="Confusion Matrix - Best Random Forest", save_path="outputs/plots/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15, save_path="outputs/plots/feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(12, 5))
    plt.barh(range(top_n), importances[indices][::-1], align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()
    plt.close()
