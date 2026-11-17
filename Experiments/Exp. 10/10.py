import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_dataset(filename="q10_optimizer_stability.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[["Feature_1", "Feature_2", "Feature_3"]].values
    y = df["Target"].values.reshape(-1, 1)
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b, y


def run_experiment(X, y, method="batch", lr=0.01, epochs=100, batch_size=32):
    np.random.seed(42)
    m = len(y)
    weights = np.array([[3.0], [-2.0], [4.0], [-1.0]])
    losses = []

    for _ in range(epochs):
        if method == "batch":
            preds = X.dot(weights)
            loss = (1 / (2 * m)) * np.sum((preds - y) ** 2)
            grads = (1 / m) * X.T.dot(preds - y)
            weights -= lr * grads
            losses.append(loss)
        elif method == "sgd":
            epoch_loss = 0
            for i in range(m):
                xi = X[i : i + 1]
                yi = y[i : i + 1]
                pred = xi.dot(weights)
                epoch_loss += (0.5 * (pred - yi) ** 2)[0, 0]
                grad = xi.T.dot(pred - yi)
                weights -= lr * grad
            losses.append(epoch_loss / m)
        elif method == "mini_batch":
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start in range(0, m, batch_size):
                xb = X_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                pred = xb.dot(weights)
                grad = (1 / len(yb)) * xb.T.dot(pred - yb)
                weights -= lr * grad
            current_preds = X.dot(weights)
            losses.append((1 / (2 * m)) * np.sum((current_preds - y) ** 2))

    return losses, weights


def generate_outputs(results_dict):
    df_losses = pd.DataFrame(results_dict)
    df_losses.index.name = "Epoch"
    df_losses.iloc[::5].to_csv(get_path("q10_table1_losses_comparison.csv"))

    stability_stats = []
    for col in df_losses.columns:
        series = df_losses[col]
        volatility = np.std(np.diff(series))
        stability_stats.append({
            "Configuration": col,
            "Initial_Loss": round(series.iloc[0], 3),
            "Final_Loss": round(series.iloc[-1], 3),
            "Loss_Variance (Stability Metric)": round(volatility, 4),
            "Status": "Unstable/Oscillating" if volatility > 0.8 else "Stable",
        })
    pd.DataFrame(stability_stats).to_csv(
        get_path("q10_table2_stability_assessment.csv"), index=False
    )

    # Plot 1: Optimization Curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, series in results_dict.items():
        style = "--" if "lr=0.08" in name else "-"
        ax.plot(series, label=name, linestyle=style)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Log Scale)")
    ax.set_title(
        "Optimization Stability across GD Variants and Learning Rates"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q10_plot1_stability_curves.png"))
    plt.close()

    # Plot 2: Volatility Bar Chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    vols = [st["Loss_Variance (Stability Metric)"] for st in stability_stats]
    labels = [st["Configuration"] for st in stability_stats]
    ax2.barh(labels, vols, color=["red" if v > 0.8 else "teal" for v in vols])
    ax2.set_xlabel("Step-to-Step Oscillation / Volatility")
    ax2.set_title("Optimizer Instability Metric Comparison")
    plt.tight_layout()
    plt.savefig(get_path("q10_plot2_volatility_comparison.png"))
    plt.close()


if __name__ == "__main__":
    X_data, y_data = load_dataset()

    configs = {
        "BGD (lr=0.01) - Smooth": run_experiment(
            X_data, y_data, method="batch", lr=0.01
        )[0],
        "BGD (lr=0.08) - High Rate": run_experiment(
            X_data, y_data, method="batch", lr=0.08
        )[0],
        "SGD (lr=0.01) - Noisy": run_experiment(
            X_data, y_data, method="sgd", lr=0.01
        )[0],
        "MiniBatch-32 (lr=0.01) - Balanced": run_experiment(
            X_data, y_data, method="mini_batch", lr=0.01, batch_size=32
        )[0],
    }

    generate_outputs(configs)
    print("Lab 10 complete: 2 tables and 2 plots saved in Exp.No.10.")