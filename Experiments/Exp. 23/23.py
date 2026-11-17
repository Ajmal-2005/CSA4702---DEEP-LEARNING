import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_cardio_dataset(filename="q23_cardio_risk_markers.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[["Age", "Cholesterol", "Resting_BP", "Fasting_BS"]].values
    y = df["Cardio_Risk"].values
    return train_test_split(X, y, test_size=0.25, random_state=42)


def train_naive_bayes(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb


def generate_outputs(nb, X_test, y_test):
    y_pred = nb.predict(X_test)
    y_probs = nb.predict_proba(X_test)
    labels = ["Low_Risk", "High_Risk"]

    # Table 1: Classification Report
    rep_dict = classification_report(
        y_test, y_pred, target_names=labels, output_dict=True
    )
    pd.DataFrame(rep_dict).transpose().to_csv(
        get_path("q23_table1_classification_report.csv")
    )

    # Table 2: Prior and Class Feature Parameters
    feat_cols = ["Age", "Cholesterol", "Resting_BP", "Fasting_BS"]
    param_df = pd.DataFrame(
        nb.theta_, index=labels, columns=[f"{c}_Mean" for c in feat_cols]
    )
    param_df["Class_Prior"] = nb.class_prior_.round(4)
    param_df.to_csv(get_path("q23_table2_gaussian_parameters.csv"))

    # Plot 1: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Cardio Risk Naive Bayes Confusion Matrix", pad=15)
    plt.tight_layout()
    plt.savefig(get_path("q23_plot1_confusion_matrix.png"))
    plt.close()

    # Plot 2: Posterior Probabilities vs Actual Risk
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(
        range(len(y_test)),
        y_probs[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="k",
    )
    ax2.axhline(0.5, color="black", linestyle="--", label="Threshold (0.5)")
    ax2.set_xlabel("Test Patient Record Index")
    ax2.set_ylabel("Posterior P(High Risk | Biomarkers)")
    ax2.set_title("Gaussian Naive Bayes Posterior Probability Spectrum")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q23_plot2_posteriors.png"))
    plt.close()


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_cardio_dataset()
    nb_model = train_naive_bayes(X_tr, y_tr)
    generate_outputs(nb_model, X_te, y_te)
    print("Lab 23 complete: 2 tables and 2 plots saved in Exp.No.23.")