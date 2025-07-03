import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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


def plot_pca_3d(X, y, output_path="outputs/plots/pca_3d.png"):
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_title("3D PCA Projection")
    plt.legend(*scatter.legend_elements(), title="Substituted")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ PCA 3D plot saved to {output_path}")
    plt.show()
    plt.close()