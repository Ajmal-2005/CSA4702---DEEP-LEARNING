import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_sensor_signals(filename="q20_mixed_machine_signals.csv"):
    df = pd.read_csv(get_path(filename))
    X_mixed = df[["Mixed_Sensor_1", "Mixed_Sensor_2"]].values
    time_pts = df["Time_Sec"].values
    return df, X_mixed, time_pts


def separate_sources(X_mixed):
    # FastICA decomposition
    ica = FastICA(n_components=2, random_state=42)
    S_ica = ica.fit_transform(X_mixed)

    # PCA decomposition for comparison
    pca = PCA(n_components=2)
    S_pca = pca.fit_transform(X_mixed)

    return S_ica, S_pca, ica.mixing_


def generate_outputs(df, S_ica, S_pca, mixing_matrix, time_pts):
    # Table 1: Recovered Mixing Matrix
    mix_df = pd.DataFrame(
        mixing_matrix,
        columns=["Est_Source_1", "Est_Source_2"],
        index=["Sensor_1", "Sensor_2"],
    )
    mix_df.to_csv(get_path("q20_table1_estimated_mixing_matrix.csv"))

    # Table 2: Signal Correlation Metrics
    s1_true = df["True_Source_1"].values
    s2_true = df["True_Source_2"].values

    corr_ica_s1 = np.max([
        np.abs(np.corrcoef(S_ica[:, 0], s1_true)[0, 1]),
        np.abs(np.corrcoef(S_ica[:, 1], s1_true)[0, 1]),
    ])
    corr_pca_s1 = np.max([
        np.abs(np.corrcoef(S_pca[:, 0], s1_true)[0, 1]),
        np.abs(np.corrcoef(S_pca[:, 1], s1_true)[0, 1]),
    ])

    comp_df = pd.DataFrame(
        {
            "Method": ["Independent Component Analysis (ICA)", "PCA"],
            "Source_1_Recovery_Correlation": [
                round(corr_ica_s1, 4),
                round(corr_pca_s1, 4),
            ],
            "Separation_Type": [
                "Non-Gaussian Independence",
                "Orthogonal Variance Maximization",
            ],
        }
    )
    comp_df.to_csv(get_path("q20_table2_separation_fidelity.csv"), index=False)

    # Plot 1: Mixed vs Unmixed Signals
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(time_pts[:150], s1_true[:150], color="navy")
    axes[0, 0].set_title("Ground Truth Source 1")
    axes[0, 1].plot(time_pts[:150], s2_true[:150], color="crimson")
    axes[0, 1].set_title("Ground Truth Source 2")

    axes[1, 0].plot(
        time_pts[:150], df["Mixed_Sensor_1"][:150], color="goldenrod"
    )
    axes[1, 0].set_title("Observed Sensor Mixture 1")
    axes[1, 1].plot(time_pts[:150], df["Mixed_Sensor_2"][:150], color="darkcyan")
    axes[1, 1].set_title("Observed Sensor Mixture 2")

    axes[2, 0].plot(time_pts[:150], S_ica[:150, 0], color="teal")
    axes[2, 0].set_title("ICA Recovered Signal 1")
    axes[2, 1].plot(time_pts[:150], S_ica[:150, 1], color="purple")
    axes[2, 1].set_title("ICA Recovered Signal 2")
    plt.tight_layout()
    plt.savefig(get_path("q20_plot1_ica_decomposition.png"))
    plt.close()

    # Plot 2: 2D Scatter Space (PCA vs ICA)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.5))
    ax2[0].scatter(S_pca[:, 0], S_pca[:, 1], alpha=0.6, color="coral")
    ax2[0].set_title("PCA Projection (Orthogonal Axes)")
    ax2[0].set_xlabel("Principal Axis 1")
    ax2[0].set_ylabel("Principal Axis 2")
    ax2[0].grid(alpha=0.3)

    ax2[1].scatter(S_ica[:, 0], S_ica[:, 1], alpha=0.6, color="forestgreen")
    ax2[1].set_title("ICA Projection (Independent Sources)")
    ax2[1].set_xlabel("Independent Axis 1")
    ax2[1].set_ylabel("Independent Axis 2")
    ax2[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q20_plot2_ica_vs_pca_scatter.png"))
    plt.close()


if __name__ == "__main__":
    df_signals, X_mix, t_pts = load_sensor_signals()
    s_ica, s_pca, mix_mat = separate_sources(X_mix)
    generate_outputs(df_signals, s_ica, s_pca, mix_mat, t_pts)
    print("Lab 20 complete: 2 tables and 2 plots saved in Exp.No.20.")