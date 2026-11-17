import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_sensor_alarms(filename="q25_sensor_alarms_rbm.csv"):
    df = pd.read_csv(get_path(filename))
    return df.values.astype(np.float32)


def corrupt_vectors(X, flip_prob=0.10, seed=42):
    """Randomly flip alarm flags to simulate corrupted sensor alarm vectors."""
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < flip_prob
    return np.where(mask, 1.0 - X, X).astype(np.float32)


def reconstruct_vectors(rbm, X):
    """One Gibbs step: visible -> hidden probabilities -> visible probabilities."""
    hidden_probs = sigmoid(np.dot(X, rbm.components_.T) + rbm.intercept_hidden_)
    return sigmoid(np.dot(hidden_probs, rbm.components_) + rbm.intercept_visible_)


def train_rbm(X_clean, X_corrupt, n_components=8, learning_rate=0.1,
              n_iter=200, batch_size=10):
    """Train a Bernoulli RBM with contrastive divergence (CD-1) and record the
    reconstruction error after every training iteration (one pass over the data)."""
    rbm = BernoulliRBM(
        n_components=n_components,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_iter=1,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    n = len(X_clean)
    history = []
    for it in range(1, n_iter + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            rbm.partial_fit(X_clean[order[start:start + batch_size]])
        rec_clean = reconstruct_vectors(rbm, X_clean)
        rec_corrupt = reconstruct_vectors(rbm, X_corrupt)
        history.append(
            {
                "Iteration": it,
                "Clean_Input_MSE": np.mean((X_clean - rec_clean) ** 2),
                "Corrupted_Input_MSE": np.mean((X_clean - rec_corrupt) ** 2),
                "Corrupted_Flag_Accuracy": np.mean((rec_corrupt > 0.5) == X_clean),
            }
        )
    return rbm, pd.DataFrame(history)


def generate_outputs(X, X_corrupt, rbm, history):
    rec_clean = reconstruct_vectors(rbm, X)
    rec_corrupt = reconstruct_vectors(rbm, X_corrupt)

    # Table 1: Reconstruction Error Metrics
    t1_df = pd.DataFrame(
        {
            "Metric": [
                "MSE: clean input reconstruction",
                "Flag accuracy: clean input reconstruction",
                "MSE: corrupted input vs clean target",
                "Flag accuracy: corrupted input before RBM",
                "Flag accuracy: corrupted input after RBM",
                "Corrupted flags restored to original value",
                "Hidden components",
            ],
            "Value": [
                round(np.mean((X - rec_clean) ** 2), 4),
                f"{np.mean((rec_clean > 0.5) == X) * 100:.2f}%",
                round(np.mean((X - rec_corrupt) ** 2), 4),
                f"{np.mean(X_corrupt == X) * 100:.2f}%",
                f"{np.mean((rec_corrupt > 0.5) == X) * 100:.2f}%",
                f"{np.mean(((rec_corrupt > 0.5) == X)[X_corrupt != X]) * 100:.2f}%",
                rbm.n_components,
            ],
        }
    )
    t1_df.to_csv(get_path("q25_table1_reconstruction_metrics.csv"), index=False)

    # Table 2: Sample Reconstruction Log (first alarm vector)
    sample_df = pd.DataFrame(
        {
            "Alarm_Flag": [f"Alarm_{i+1}" for i in range(X.shape[1])],
            "Original_Vector": X[0].astype(int),
            "Corrupted_Vector": X_corrupt[0].astype(int),
            "Reconstructed_Prob": rec_corrupt[0].round(4),
            "Binarized_Output": (rec_corrupt[0] > 0.5).astype(int),
        }
    )
    sample_df.to_csv(get_path("q25_table2_sample_reconstruction.csv"), index=False)

    # Table 3: Reconstruction error across training iterations
    keep = sorted(set([1] + list(range(20, len(history) + 1, 20)) + [len(history)]))
    log_df = history[history["Iteration"].isin(keep)].round(4)
    log_df.to_csv(get_path("q25_table3_error_by_iteration.csv"), index=False)

    # Plot 1: Original vs Corrupted vs Reconstructed
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(X[:10], cmap="Blues", aspect="auto")
    axes[0].set_title("Original Alarm Vectors")
    axes[1].imshow(X_corrupt[:10], cmap="Blues", aspect="auto")
    axes[1].set_title("Corrupted Alarm Vectors")
    axes[2].imshow(rec_corrupt[:10], cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("RBM Reconstruction (from corrupted)")
    for ax in axes:
        ax.set_xlabel("Alarm Index (1-16)")
    axes[0].set_ylabel("Equipment Instance")
    plt.tight_layout()
    plt.savefig(get_path("q25_plot1_reconstruction_heatmap.png"))
    plt.close()

    # Plot 2: Learned RBM Component Filters
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    im = ax2.matshow(rbm.components_, cmap="coolwarm", aspect="auto")
    fig2.colorbar(im)
    ax2.set_xlabel("Visible Alarm Features")
    ax2.set_ylabel("Latent Hidden Components")
    ax2.set_title("RBM Weight Components (Contrastive Divergence)", pad=15)
    plt.tight_layout()
    plt.savefig(get_path("q25_plot2_latent_weights.png"))
    plt.close()

    # Plot 3: Reconstruction error across training iterations
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(history["Iteration"], history["Clean_Input_MSE"], label="Clean input", color="teal", linewidth=2)
    ax3.plot(history["Iteration"], history["Corrupted_Input_MSE"], label="Corrupted input", color="crimson", linewidth=2)
    ax3.set_xlabel("Training Iteration")
    ax3.set_ylabel("Mean Squared Reconstruction Error")
    ax3.set_title("RBM Reconstruction Error vs Training Iterations")
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q25_plot3_error_vs_iterations.png"))
    plt.close()


if __name__ == "__main__":
    X_alarms = load_sensor_alarms()
    X_corrupted = corrupt_vectors(X_alarms)
    rbm_model, error_history = train_rbm(X_alarms, X_corrupted)
    generate_outputs(X_alarms, X_corrupted, rbm_model, error_history)
    print("Lab 25 complete: 3 tables and 3 plots saved in Exp.No.25.")
