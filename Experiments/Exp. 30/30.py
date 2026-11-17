import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_generative_benchmark(filename="q30_generative_benchmark.csv"):
    df = pd.read_csv(get_path(filename))
    real = df[["Real_F1", "Real_F2"]].values
    vae = df[["VAE_F1", "VAE_F2"]].values
    gan = df[["GAN_F1", "GAN_F2"]].values
    return real, vae, gan


def frechet_distance(a, b):
    """Frechet (2-Wasserstein) distance between two Gaussians fitted to the samples:
    ||mu_a - mu_b||^2 + Tr(C_a + C_b - 2 (C_a C_b)^(1/2)), with a true matrix square root."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cov_sqrt = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * cov_sqrt))


def mean_pairwise_distance(x):
    """Average distance between different samples (diagonal zeros excluded)."""
    d = cdist(x, x)
    n = len(x)
    return float(d.sum() / (n * (n - 1)))


def nearest_neighbour_mse(real, gen):
    """Mean squared distance from every real sample to its nearest generated sample."""
    return float(np.mean(cdist(real, gen, "sqeuclidean").min(axis=1)))


def knn_precision_recall(real, gen, k=3):
    """k-NN manifold precision (generated samples lying on the real manifold) and
    recall (real samples covered by the generated manifold)."""
    def radii(x):
        d = cdist(x, x)
        return np.sort(d, axis=1)[:, k]

    r_real, r_gen = radii(real), radii(gen)
    d_gr = cdist(gen, real)
    d_rg = cdist(real, gen)
    precision = float(np.mean((d_gr <= r_real[None, :]).any(axis=1)))
    recall = float(np.mean((d_rg <= r_gen[None, :]).any(axis=1)))
    return precision, recall


def bootstrap_frechet(real, gen, n_boot=50, seed=42):
    """Stability of the distribution distance over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        r = real[rng.integers(0, len(real), len(real))]
        g = gen[rng.integers(0, len(gen), len(gen))]
        vals.append(frechet_distance(r, g))
    return float(np.mean(vals)), float(np.std(vals))


def compute_evaluation_metrics(real, vae, gan):
    out = {}
    for name, gen in (("VAE", vae), ("GAN", gan)):
        prec, rec = knn_precision_recall(real, gen)
        b_mean, b_std = bootstrap_frechet(real, gen)
        out[name] = {
            "paired_mse": float(np.mean((real - gen) ** 2)),
            "nn_mse": nearest_neighbour_mse(real, gen),
            "diversity": mean_pairwise_distance(gen),
            "frechet": frechet_distance(real, gen),
            "precision": prec,
            "recall": rec,
            "boot_mean": b_mean,
            "boot_std": b_std,
        }
    out["Real"] = {"diversity": mean_pairwise_distance(real)}
    return out


def generate_outputs(real, vae, gan, m):
    # Table 1: Fidelity and diversity benchmark
    t1 = pd.DataFrame(
        {
            "Model": ["Real Ground Truth", "Variational Autoencoder (VAE)", "Generative Adversarial Net (GAN)"],
            "Reconstruction_MSE (paired)": [0.0, round(m["VAE"]["paired_mse"], 4), round(m["GAN"]["paired_mse"], 4)],
            "Nearest_Neighbour_MSE": [0.0, round(m["VAE"]["nn_mse"], 4), round(m["GAN"]["nn_mse"], 4)],
            "Sample_Diversity": [round(m["Real"]["diversity"], 3), round(m["VAE"]["diversity"], 3), round(m["GAN"]["diversity"], 3)],
            "Frechet_Distance": [0.0, round(m["VAE"]["frechet"], 4), round(m["GAN"]["frechet"], 4)],
        }
    )
    t1.to_csv(get_path("q30_table1_model_comparison.csv"), index=False)

    # Table 2: Coverage and consistency metrics
    t2 = pd.DataFrame(
        {
            "Model": ["Variational Autoencoder (VAE)", "Generative Adversarial Net (GAN)"],
            "Precision (fidelity)": [round(m["VAE"]["precision"], 4), round(m["GAN"]["precision"], 4)],
            "Recall (coverage)": [round(m["VAE"]["recall"], 4), round(m["GAN"]["recall"], 4)],
            "Bootstrap_Frechet_Mean": [round(m["VAE"]["boot_mean"], 4), round(m["GAN"]["boot_mean"], 4)],
            "Bootstrap_Frechet_Std": [round(m["VAE"]["boot_std"], 4), round(m["GAN"]["boot_std"], 4)],
        }
    )
    t2.to_csv(get_path("q30_table2_coverage_consistency.csv"), index=False)

    # Table 3: Better model for each metric (computed from the values above)
    real_div = m["Real"]["diversity"]

    def pick(metric, rule):
        v, g = m["VAE"][metric], m["GAN"][metric]
        if rule == "low":
            return "VAE" if v < g else "GAN"
        if rule == "high":
            return "VAE" if v > g else "GAN"
        return "VAE" if abs(v - real_div) < abs(g - real_div) else "GAN"

    t3 = pd.DataFrame(
        {
            "Metric": [
                "Reconstruction MSE (paired)",
                "Nearest-neighbour MSE",
                "Sample diversity (closest to real)",
                "Frechet distance",
                "Precision",
                "Recall",
                "Bootstrap stability (lower std)",
            ],
            "Better_Model": [
                pick("paired_mse", "low"),
                pick("nn_mse", "low"),
                pick("diversity", "close"),
                pick("frechet", "low"),
                pick("precision", "high"),
                pick("recall", "high"),
                pick("boot_std", "low"),
            ],
        }
    )
    t3.to_csv(get_path("q30_table3_best_model_per_metric.csv"), index=False)

    # Plot 1: 2D Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    axes[0].scatter(real[:, 0], real[:, 1], color="darkblue", alpha=0.6)
    axes[0].set_title("Ground Truth Distribution")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[1].scatter(vae[:, 0], vae[:, 1], color="teal", alpha=0.6)
    axes[1].set_title("VAE Generated")
    axes[1].set_xlabel("Feature 1")
    axes[2].scatter(gan[:, 0], gan[:, 1], color="crimson", alpha=0.6)
    axes[2].set_title("GAN Generated")
    axes[2].set_xlabel("Feature 1")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q30_plot1_distribution_scatter.png"))
    plt.close()

    # Plot 2: Frechet distance benchmark
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    scores = [m["VAE"]["frechet"], m["GAN"]["frechet"]]
    ax2.bar(["VAE", "GAN"], scores, color=["teal", "crimson"], width=0.45)
    ax2.set_ylim(0, max(scores) * 1.25)
    ax2.set_ylabel("Frechet Distance to Real (Lower is Better)")
    ax2.set_title("Frechet Distance Benchmark: VAE vs GAN")
    for i, v in enumerate(scores):
        ax2.text(i, v + max(scores) * 0.02, f"{v:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(get_path("q30_plot2_metric_benchmark.png"))
    plt.close()

    # Plot 3: Precision and recall
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    xs = np.arange(2)
    ax3.bar(xs - 0.18, [m["VAE"]["precision"], m["GAN"]["precision"]], width=0.36, label="Precision (fidelity)", color="steelblue")
    ax3.bar(xs + 0.18, [m["VAE"]["recall"], m["GAN"]["recall"]], width=0.36, label="Recall (coverage)", color="darkorange")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["VAE", "GAN"])
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel("Score (Higher is Better)")
    ax3.set_title("k-NN Precision and Recall: VAE vs GAN")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(get_path("q30_plot3_precision_recall.png"))
    plt.close()


if __name__ == "__main__":
    r_data, v_data, g_data = load_generative_benchmark()
    metrics = compute_evaluation_metrics(r_data, v_data, g_data)
    generate_outputs(r_data, v_data, g_data, metrics)
    print("Lab 30 complete: 3 tables and 3 plots saved in Exp.No.30.")
