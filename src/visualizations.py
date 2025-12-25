import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


# === Setup Paths ===
file_path = r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\data\processed\should_be_subbed_dataset.csv"
output_dir = r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\data\processed"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(file_path)
print("✅ Columns in dataset:")
print(df.columns.tolist())  # Print all column names for debugging

# === Replace any missing values just in case ===
df.fillna(0, inplace=True)

# === Column name check/fix ===
# Rename 'position' to 'position_name' if that’s the actual name
if 'position' not in df.columns and 'position_name' in df.columns:
    df.rename(columns={"position_name": "position"}, inplace=True)

# === Safe Feature List ===
features_to_plot = [
    "position",
    "match_context_score_diff",
    "match_context_minutes_remaining",
    "team_stamina_level",
    "opponent_stamina_level",
    "player_rating",
    "player_rating_diff",
    "context_time_window_mean_event_count",
    "context_time_window_total_distance_run",
    "was_previous_substitution",
    "card_status"
]

# === Encode categoricals if present ===
for feature in ["position", "card_status"]:
    if feature in df.columns and df[feature].dtype == "object":
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))

# === Plot distributions ===
for feature in features_to_plot:
    if feature in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{feature}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved: {plot_path}")

# === Additional Features ===
additional_cols = [
    "position",
    "minutes_played",
    "total_running_distance",
    "passes_last_15_minute",
    "should_be_subbed_x"
]

for col in additional_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()
        print(f"✅ Saved: {col}_distribution.png")

# === Optional: PCA 3D ===
def plot_pca_3d(X, y, output_path):
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_title("3D PCA Projection")
    plt.legend(*scatter.legend_elements(), title="Subbed")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ PCA saved to {output_path}")

# PCA example usage
if "should_be_subbed_x" in df.columns:
    target = df["should_be_subbed_x"]
    features_for_pca = df.drop(columns=["should_be_subbed_x"])
    if "position" in features_for_pca.columns and features_for_pca["position"].dtype == "object":
        features_for_pca["position"] = LabelEncoder().fit_transform(features_for_pca["position"])
    plot_pca_3d(features_for_pca.values, target.values, os.path.join(output_dir, "pca_3d.png"))

plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()
print("✅ Correlation matrix saved.")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="should_be_subbed_x")
plt.title("Substitution Class Distribution")
plt.xlabel("Should Be Subbed (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "substitution_class_distribution.png"))
plt.close()
print("✅ Substitution label distribution saved.")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="should_be_subbed_x", y="total_running_distance")
plt.title("Running Distance by Substitution Label")
plt.xlabel("Should Be Subbed")
plt.ylabel("Total Running Distance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "running_distance_by_subbed.png"))
plt.close()
print("✅ Boxplot of running distance saved.")


# Load datasets
final_df = pd.read_csv(r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\outputs\20250721_041759\final_df.csv")
should_be_subbed_df = pd.read_csv(r"H:\AI AS\Individual Project\AI-Powered-Agent-Assistant-Recommender-for-Football-Tactics-During-Live-Matches\data\processed\should_be_subbed_dataset.csv")


final_df['source'] = 'main'
should_be_subbed_df['source'] = 'should_be_subbed'
datasets = {'main': final_df, 'should_be_subbed': should_be_subbed_df}

output_dir = "feature_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

for name, df in datasets.items():
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    corr_matrix.to_csv(f"{output_dir}/{name}_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"Feature Correlation Heatmap ({name})")
    plt.savefig(f"{output_dir}/{name}_correlation_heatmap.png", bbox_inches='tight')
    plt.close()

    tactical_features = ['time_left', 'goal_difference', 'stamina_subbed', 'event_rate', 'fatigue_index']
    for feature in tactical_features:
        if feature in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=df, x=feature, hue='should_be_subbed', kde=True, bins=30, palette='viridis')
            plt.title(f"Distribution of {feature} ({name})")
            plt.savefig(f"{output_dir}/{name}_{feature}_distribution.png", bbox_inches='tight')
            plt.close()
