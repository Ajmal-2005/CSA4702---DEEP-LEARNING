import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_data(filename="q11_financial_risk_data.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"ratio_{i+1}" for i in range(20)]].values
    y = df["Financial_Default"].values
    return train_test_split(X, y, test_size=0.25, random_state=42)


def build_model(reg_type=None, reg_val=0.01):
    reg = None
    if reg_type == "l1":
        reg = l1(reg_val)
    elif reg_type == "l2":
        reg = l2(reg_val)

    model = Sequential([
        Input(shape=(20,)),
        Dense(32, activation="relu", kernel_regularizer=reg),
        Dense(16, activation="relu", kernel_regularizer=reg),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def generate_outputs(histories, models, X_test, y_test):
    results = []
    first_layer_weights = {}

    for name, m in models.items():
        loss_val, acc_val = m.evaluate(X_test, y_test, verbose=0)
        w = m.layers[0].get_weights()[0].flatten()
        first_layer_weights[name] = w

        # Measure sparsity: fraction of weights near zero
        sparsity = np.mean(np.abs(w) < 0.01)

        results.append({
            "Model": name,
            "Test_Loss": round(loss_val, 4),
            "Test_Accuracy": round(acc_val, 4),
            "Weight_Sparsity (<0.01)": f"{sparsity*100:.1f}%",
            "Weight_L2_Norm": round(np.linalg.norm(w), 4),
        })

    pd.DataFrame(results).to_csv(
        get_path("q11_table1_regularization_metrics.csv"), index=False
    )

    # Plot 1: Weight distribution comparing L1, L2, Unregularized
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, w) in zip(axes, first_layer_weights.items()):
        ax.hist(w, bins=25, color="teal", edgecolor="black", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Weight Value")
    axes[0].set_ylabel("Count")
    plt.suptitle("Layer 1 Weight Distributions Under Different Regularizers")
    plt.tight_layout()
    plt.savefig(get_path("q11_plot1_weight_sparsity.png"))
    plt.close()

    # Plot 2: Validation Loss comparison
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for name, h in histories.items():
        ax2.plot(h.history["val_loss"], label=f"{name} Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Binary Cross-Entropy Loss")
    ax2.set_title("Overfitting Control: Validation Loss across Regularizers")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q11_plot2_val_loss_comparison.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_configs = {
        "Unregularized (Baseline)": build_model(None),
        "L1 Regularization (Lasso)": build_model("l1", reg_val=0.01),
        "L2 Regularization (Ridge)": build_model("l2", reg_val=0.01),
    }

    histories = {}
    for name, model in model_configs.items():
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=60,
            batch_size=16,
            verbose=0,
        )
        histories[name] = hist

    generate_outputs(histories, model_configs, X_test, y_test)
    print("Lab 11 complete: 1 table and 2 plots saved in Exp.No.11.")