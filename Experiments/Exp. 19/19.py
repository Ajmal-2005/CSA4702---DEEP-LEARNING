import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_retail_spending(filename="q19_retail_spending_pca.csv"):
    df = pd.read_csv(get_path(filename))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data


def run_pca_decomposition(scaled_data):
    pca = PCA(n_components=6)
    pca_features = pca.fit_transform(scaled_data)
    return pca, pca_features


def generate_outputs(df, pca, pca_features):
    comp_names = [f"PC_{i+1}" for i in range(len(pca.explained_variance_ratio_))]
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Table 1: Explained Variance Breakdown
    var_df = pd.DataFrame(
        {
            "Principal_Component": comp_names,
            "Eigenvalue (Var)": pca.explained_variance_.round(4),
            "Explained_Variance_Ratio": exp_var.round(4),
            "Cumulative_Variance_Ratio": cum_var.round(4),
        }
    )
    var_df.to_csv(get_path("q19_table1_explained_variance.csv"), index=False)

    # Table 2: Component Loadings
    loadings_df = pd.DataFrame(
        pca.components_[:2].T,
        index=df.columns,
        columns=["PC1_Loading", "PC2_Loading"],
    )
    loadings_df.to_csv(get_path("q19_table2_factor_loadings.csv"))

    # Plot 1: Scree Plot (Elbow Curve)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        comp_names,
        exp_var,
        alpha=0.6,
        color="royalblue",
        label="Individual Variance",
    )
    ax.step(
        comp_names,
        cum_var,
        where="mid",
        color="crimson",
        marker="o",
        label="Cumulative Variance",
        linewidth=2,
    )
    ax.axhline(0.9, color="black", linestyle="--", label="90% Threshold")
    ax.set_ylabel("Variance Explained Ratio")
    ax.set_title("PCA Scree Plot: Customer Retail Spending")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q19_plot1_scree_plot.png"))
    plt.close()

    # Plot 2: 2D Projected Data Distribution
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    scatter = ax2.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        c=df["Groceries_Spend"],
        cmap="viridis",
        alpha=0.8,
    )
    fig2.colorbar(scatter, label="Groceries Spend ($)")
    ax2.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% Variance)")
    ax2.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% Variance)")
    ax2.set_title("2D Projection via Principal Component Analysis")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q19_plot2_2d_projection.png"))
    plt.close()


if __name__ == "__main__":
    raw_df, X_scaled = load_retail_spending()
    pca_model, projected_X = run_pca_decomposition(X_scaled)
    generate_outputs(raw_df, pca_model, projected_X)
    print("Lab 19 complete: 2 tables and 2 plots saved in Exp.No.19.")