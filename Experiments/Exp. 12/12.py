import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_traffic_images(filename="q12_traffic_vehicles.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"p_{i}" for i in range(256)]].values.reshape(-1, 16, 16, 1) / 255.0
    y = df["Vehicle_Type"].values
    y_cat = to_categorical(y, num_classes=4)
    return train_test_split(X, y_cat, y, test_size=0.25, random_state=42)


def build_cnn_classifier():
    model = Sequential([
        Input(shape=(16, 16, 1)),
        Conv2D(16, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_outputs(history, y_raw_test, y_pred_classes):
    class_names = ["Car", "Bus", "Motorcycle", "Truck"]

    # Table 1: Classification Report
    rep_dict = classification_report(
        y_raw_test, y_pred_classes, target_names=class_names, output_dict=True
    )
    pd.DataFrame(rep_dict).transpose().to_csv(
        get_path("q12_table1_classification_report.csv")
    )

    # Table 2: Confusion Matrix
    cm = confusion_matrix(y_raw_test, y_pred_classes)
    pd.DataFrame(cm, columns=class_names, index=class_names).to_csv(
        get_path("q12_table2_confusion_matrix.csv")
    )

    # Plot 1: CNN Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="YlGnBu", alpha=0.85)
    fig.colorbar(cax)
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Vehicle Type")
    ax.set_ylabel("True Vehicle Type")
    ax.set_title("Traffic System Classification: Confusion Matrix", pad=15)
    plt.tight_layout()
    plt.savefig(get_path("q12_plot1_confusion_matrix.png"))
    plt.close()

    # Plot 2: Training & Validation Curves
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    ax2[0].plot(
        history.history["loss"], label="Train Loss", color="royalblue"
    )
    ax2[0].plot(
        history.history["val_loss"],
        label="Val Loss",
        color="crimson",
        linestyle="--",
    )
    ax2[0].set_title("CNN Cross-Entropy Loss")
    ax2[0].set_xlabel("Epoch")
    ax2[0].legend()
    ax2[0].grid(alpha=0.3)

    ax2[1].plot(
        history.history["accuracy"], label="Train Accuracy", color="teal"
    )
    ax2[1].plot(
        history.history["val_accuracy"],
        label="Val Accuracy",
        color="darkorange",
        linestyle="--",
    )
    ax2[1].set_title("CNN Accuracy")
    ax2[1].set_xlabel("Epoch")
    ax2[1].legend()
    ax2[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q12_plot2_cnn_performance.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_cat_train, y_cat_test, y_raw_train, y_raw_test = (
        load_traffic_images()
    )
    model = build_cnn_classifier()

    history = model.fit(
        X_train,
        y_cat_train,
        validation_data=(X_test, y_cat_test),
        epochs=35,
        batch_size=16,
        verbose=0,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    generate_outputs(history, y_raw_test, y_pred_classes)
    print("Lab 12 complete: 2 tables and 2 plots saved in Exp.No.12.")