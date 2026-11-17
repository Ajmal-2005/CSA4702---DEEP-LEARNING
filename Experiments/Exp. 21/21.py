import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_survey_data(filename="q21_bank_satisfaction_survey.csv"):
    df = pd.read_csv(get_path(filename))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data


def extract_latent_factors(scaled_data, n_factors=2):
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_components = fa.fit_transform(scaled_data)

    pca = PCA(n_components=n_factors, random_state=42)
    pca.fit(scaled_data)
    return fa, fa_components, pca


def generate_outputs(df, fa, fa_components, pca):
    factor_names = ["Factor_1 (Digital)", "Factor_2 (In-Branch)"]

    # Table 1: Factor Loadings
    loadings_df = pd.DataFrame(
        fa.components_.T, index=df.columns, columns=factor_names
    )
    loadings_df["Noise_Variance_Psi"] = fa.noise_variance_.round(4)
    loadings_df.to_csv(get_path("q21_table1_factor_loadings.csv"))

    # Table 2: Model Comparison
    comp_df = pd.DataFrame(
        {
            "Dimension": ["Component / Factor 1", "Component / Factor 2"],
            "FA_Log_Likelihood": [
                round(fa.score(StandardScaler().fit_transform(df)), 2),
                np.nan,
            ],
            "PCA_Explained_Var": [
                round(pca.explained_variance_ratio_[0], 4),
                round(pca.explained_variance_ratio_[1], 4),
            ],
        }
    )
    comp_df.to_csv(get_path("q21_table2_fa_vs_pca.csv"), index=False)

    # Plot 1: Factor Loadings Heatmap
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cax = ax.matshow(fa.components_.T, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(factor_names)
    ax.set_yticklabels(df.columns)
    for i in range(len(df.columns)):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{fa.components_[j, i]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    ax.set_title("Latent Factor Loadings Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(get_path("q21_plot1_factor_loadings.png"))
    plt.close()

    # Plot 2: Latent Factor Space Distribution
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    scatter = ax2.scatter(
        fa_components[:, 0],
        fa_components[:, 1],
        c=df["Fee_Fairness"],
        cmap="plasma",
        alpha=0.75,
    )
    fig2.colorbar(scatter, label="Fee Fairness Rating")
    ax2.set_xlabel("Latent Factor 1 (Digital Platform Satisfaction)")
    ax2.set_ylabel("Latent Factor 2 (Branch Service Satisfaction)")
    ax2.set_title("Customer Segmentation in Latent Factor Space")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q21_plot2_latent_scatter.png"))
    plt.close()


if __name__ == "__main__":
    survey_df, X_std = load_survey_data()
    fa_model, fa_comps, pca_model = extract_latent_factors(X_std)
    generate_outputs(survey_df, fa_model, fa_comps, pca_model)
    print("Lab 21 complete: 2 tables and 2 plots saved in Exp.No.21.")