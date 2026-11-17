import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_digit_data(filename="q26_digits_dbn.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"p_{i}" for i in range(64)]].values.astype(np.float32)
    y = df["Target_Digit"].values
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def build_and_train_dbn(X_train, y_train, h1=100, h2=50, lr=0.04, n_iter=200):
    """Greedy layer-wise unsupervised pretraining of two stacked RBMs,
    followed by a supervised logistic-regression (softmax) classifier."""
    # Layer 1 RBM (64 inputs -> h1 hidden units), trained on the raw pixels
    rbm1 = BernoulliRBM(n_components=h1, learning_rate=lr, n_iter=n_iter, random_state=42)
    rbm1.fit(X_train)
    H1 = rbm1.transform(X_train)

    # Layer 2 RBM (h1 inputs -> h2 hidden units), trained on layer-1 features
    rbm2 = BernoulliRBM(n_components=h2, learning_rate=lr, n_iter=n_iter, random_state=42)
    rbm2.fit(H1)
    H2 = rbm2.transform(H1)

    # Supervised classification head on the top-layer features
    dbn_clf = LogisticRegression(C=100, max_iter=4000)
    dbn_clf.fit(H2, y_train)

    # Single-layer RBM model for comparison (64 -> h1 -> softmax)
    single_clf = LogisticRegression(C=100, max_iter=4000)
    single_clf.fit(H1, y_train)

    # Baseline: logistic regression on the raw pixels
    raw_clf = LogisticRegression(max_iter=4000)
    raw_clf.fit(X_train, y_train)

    return rbm1, rbm2, dbn_clf, single_clf, raw_clf


def generate_outputs(rbm1, rbm2, dbn_clf, single_clf, raw_clf, X_test, y_test, h1, h2):
    H1_test = rbm1.transform(X_test)
    H2_test = rbm2.transform(H1_test)

    y_pred_dbn = dbn_clf.predict(H2_test)
    y_pred_single = single_clf.predict(H1_test)
    y_pred_raw = raw_clf.predict(X_test)

    acc_dbn = accuracy_score(y_test, y_pred_dbn)
    acc_single = accuracy_score(y_test, y_pred_single)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # Table 1: Model Comparison
    t1_df = pd.DataFrame(
        {
            "Architecture": [
                "Stacked Deep Belief Network (DBN)",
                "Single-layer RBM + Classifier",
                "Baseline Logistic Regression",
            ],
            "Layer_Structure": [
                f"64 -> {h1} -> {h2} -> Softmax",
                f"64 -> {h1} -> Softmax",
                "64 -> Softmax",
            ],
            "Classification_Accuracy": [
                f"{acc_dbn * 100:.2f}%",
                f"{acc_single * 100:.2f}%",
                f"{acc_raw * 100:.2f}%",
            ],
        }
    )
    t1_df.to_csv(get_path("q26_table1_dbn_performance.csv"), index=False)

    # Table 2: Detailed DBN Classification Report
    rep_dict = classification_report(y_test, y_pred_dbn, output_dict=True, zero_division=0)
    pd.DataFrame(rep_dict).transpose().to_csv(get_path("q26_table2_classification_report.csv"))

    # Plot 1: Accuracy comparison (axis starts at 0 so every bar is visible)
    names = ["Baseline\n(Raw Pixels)", "Single-layer\nRBM", "Stacked DBN\n(2 RBM layers)"]
    values = [acc_raw, acc_single, acc_dbn]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, values, color=["salmon", "goldenrod", "teal"], width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Digit Classification: Baseline vs Single RBM vs Stacked DBN")
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v*100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(get_path("q26_plot1_accuracy_comparison.png"))
    plt.close()

    # Plot 2: First-Layer RBM Receptive Fields
    fig2, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(rbm1.components_[i].reshape(8, 8), cmap="Greys")
        ax.axis("off")
        ax.set_title(f"Feature {i+1}")
    plt.suptitle("Layer 1 Learned Visual Receptive Fields")
    plt.tight_layout()
    plt.savefig(get_path("q26_plot2_receptive_fields.png"))
    plt.close()


if __name__ == "__main__":
    H1_UNITS, H2_UNITS = 100, 50
    X_tr, X_te, y_tr, y_te = load_digit_data()
    r1, r2, dbn, single, baseline = build_and_train_dbn(X_tr, y_tr, h1=H1_UNITS, h2=H2_UNITS)
    generate_outputs(r1, r2, dbn, single, baseline, X_te, y_te, H1_UNITS, H2_UNITS)
    print("Lab 26 complete: 2 tables and 2 plots saved in Exp.No.26.")
