import shap
import matplotlib.pyplot as plt

def explain_model_shap(model, X, feature_names, output_path="outputs/plots/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ SHAP summary plot saved to {output_path}")
