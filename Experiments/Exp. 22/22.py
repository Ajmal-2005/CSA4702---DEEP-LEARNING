import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_telecom_data(filename="q22_telecom_call_mixture.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[["Call_Duration_Min", "Data_Volume_MB"]].values
    return df, X


def fit_gmm_model(X, n_clusters=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    posteriors = gmm.predict_proba(X_scaled)
    cluster_labels = gmm.predict(X_scaled)
    return gmm, posteriors, cluster_labels, scaler


def generate_outputs(df, gmm, posteriors, labels, scaler):
    # Table 1: GMM Component Parameters
    means = scaler.inverse_transform(gmm.means_)
    weights = gmm.weights_

    params_df = pd.DataFrame(
        {
            "Cluster": [f"Cluster_{i}" for i in range(len(weights))],
            "Mixture_Weight": weights.round(4),
            "Mean_Call_Duration_Min": means[:, 0].round(2),
            "Mean_Data_Usage_MB": means[:, 1].round(2),
        }
    )
    params_df.to_csv(get_path("q22_table1_mixture_parameters.csv"), index=False)

    # Table 2: Sample Posterior Probabilities
    post_df = df.head(15).copy()
    post_df["Prob_Cluster_0"] = posteriors[:15, 0].round(4)
    post_df["Prob_Cluster_1"] = posteriors[:15, 1].round(4)
    post_df["Assigned_Cluster"] = labels[:15]
    post_df.to_csv(get_path("q22_table2_posterior_probabilities.csv"), index=False)

    # Plot 1: Soft Posterior Probability Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        posteriors[:, 1],
        bins=20,
        color="teal",
        edgecolor="black",
        alpha=0.75,
    )
    ax.axvline(0.5, color="red", linestyle="--", label="Decision Boundary (p=0.5)")
    ax.set_xlabel("Posterior Probability P(Cluster 1 | X)")
    ax.set_ylabel("Call Count")
    ax.set_title("GMM Cluster Posterior Assignment Confidence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q22_plot1_posterior_hist.png"))
    plt.close()

    # Plot 2: 2D Soft Cluster Clustering
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scatter = ax2.scatter(
        df["Call_Duration_Min"],
        df["Data_Volume_MB"],
        c=posteriors[:, 1],
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )
    fig2.colorbar(scatter, label="P(Long/Data Intensive Cluster)")
    ax2.scatter(
        means[:, 0],
        means[:, 1],
        marker="X",
        s=150,
        c="yellow",
        edgecolors="black",
        label="Cluster Means",
    )
    ax2.set_xlabel("Call Duration (Minutes)")
    ax2.set_ylabel("Data Volume (MB)")
    ax2.set_title("GMM Probabilistic Customer Clustering")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q22_plot2_clustering_space.png"))
    plt.close()


if __name__ == "__main__":
    raw_data, X_vals = load_telecom_data()
    model, post_probs, cluster_idx, std_scaler = fit_gmm_model(X_vals)
    generate_outputs(raw_data, model, post_probs, cluster_idx, std_scaler)
    print("Lab 22 complete: 2 tables and 2 plots saved in Exp.No.22.")