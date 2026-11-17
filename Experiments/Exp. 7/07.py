import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_student_data(filename="q7_student_pass_fail.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[["Attendance", "Internal_Exam", "Assignment_Score"]].values
    y = df["Result"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_feedforward_model():
    model = Sequential([
        Input(shape=(3,)),
        Dense(8, activation="relu"),
        Dense(4, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def generate_outputs(history, y_test, y_pred, y_prob):
    # Tables
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    comp_df = pd.DataFrame(
        {
            "Actual_Result": y_test[:15],
            "Predicted_Probability": y_prob[:15].flatten().round(4),
            "Predicted_Result": y_pred_binary[:15],
            "Status": [
                "Correct" if a == p else "Incorrect"
                for a, p in zip(y_test[:15], y_pred_binary[:15])
            ],
        }
    )
    comp_df.to_csv(
        get_path("q7_table1_predictions_comparison.csv"), index=False
    )

    report_dict = classification_report(
        y_test, y_pred_binary, output_dict=True
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        get_path("q7_table2_classification_report.csv")
    )

    cm = confusion_matrix(y_test, y_pred_binary)
    pd.DataFrame(
        cm,
        columns=["Pred_Fail", "Pred_Pass"],
        index=["Actual_Fail", "Actual_Pass"],
    ).to_csv(get_path("q7_table3_confusion_matrix.csv"))

    # Plot 1: Training History
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="Train Loss", color="royalblue")
    ax[0].plot(
        history.history["val_loss"],
        label="Val Loss",
        color="crimson",
        linestyle="--",
    )
    ax[0].set_title("Binary Cross-Entropy Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(
        history.history["accuracy"], label="Train Accuracy", color="teal"
    )
    ax[1].plot(
        history.history["val_accuracy"],
        label="Val Accuracy",
        color="darkorange",
        linestyle="--",
    )
    ax[1].set_title("Training vs Validation Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q7_plot1_training_history.png"))
    plt.close()

    # Plot 2: Confusion Matrix Heatmap
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    cax = ax2.matshow(cm, cmap="Blues", alpha=0.8)
    fig2.colorbar(cax)
    for i in range(2):
        for j in range(2):
            ax2.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Fail", "Pass"])
    ax2.set_yticklabels(["Fail", "Pass"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix Heatmap", pad=12)
    plt.tight_layout()
    plt.savefig(get_path("q7_plot2_confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_student_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_feedforward_model()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=70,
        batch_size=8,
        verbose=0,
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)
    generate_outputs(history, y_test, y_pred, y_prob)
    print("Lab 7 complete: 3 tables and 2 plots saved in Exp.No.07.")