import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_dataset(filename="q8_activation_benchmark.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"f_{i}" for i in range(1, 7)]].values
    y = df["Target"].values
    return train_test_split(X, y, test_size=0.25, random_state=42)


def build_deep_model(activation_func):
    # 5-layer deep architecture to demonstrate gradient flow
    model = Sequential([
        Input(shape=(6,)),
        Dense(16, activation=activation_func),
        Dense(16, activation=activation_func),
        Dense(16, activation=activation_func),
        Dense(16, activation=activation_func),
        Dense(1, activation="sigmoid"),
    ])
    return model


def compute_layer_gradients(model, X_sample, y_sample):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        preds = model(X_sample, training=True)
        loss = loss_fn(y_sample, preds)
    grads = tape.gradient(loss, model.trainable_weights)

    # Average gradient magnitude per weight matrix (skip biases)
    layer_grad_norms = [
        tf.norm(g).numpy() for i, g in enumerate(grads) if i % 2 == 0
    ]
    return layer_grad_norms


def generate_outputs(grad_results, training_histories):
    # Tables
    df_grads = pd.DataFrame(grad_results)
    df_grads.index = [f"Hidden_Layer_{i+1}" for i in range(len(df_grads))]
    df_grads.to_csv(get_path("q8_table1_gradient_norms.csv"))

    summary_df = pd.DataFrame(
        {
            "Activation": ["sigmoid", "tanh", "relu"],
            "Initial_Layer1_Grad": [
                grad_results["sigmoid"][0],
                grad_results["tanh"][0],
                grad_results["relu"][0],
            ],
            "Final_Layer_Grad": [
                grad_results["sigmoid"][-1],
                grad_results["tanh"][-1],
                grad_results["relu"][-1],
            ],
            "Gradient_Decay_Ratio (L1/L_final)": [
                grad_results["sigmoid"][0] / (grad_results["sigmoid"][-1] + 1e-8),
                grad_results["tanh"][0] / (grad_results["tanh"][-1] + 1e-8),
                grad_results["relu"][0] / (grad_results["relu"][-1] + 1e-8),
            ],
        }
    )
    summary_df.to_csv(
        get_path("q8_table2_vanishing_gradient_summary.csv"), index=False
    )

    # Plot 1: Gradient Norm across depth
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = [f"L{i+1}" for i in range(len(df_grads))]
    ax.plot(
        layers,
        grad_results["sigmoid"],
        marker="o",
        label="Sigmoid (Vanishing Flow)",
        color="crimson",
        linewidth=2,
    )
    ax.plot(
        layers,
        grad_results["tanh"],
        marker="s",
        label="Tanh (Moderate Flow)",
        color="darkorange",
        linewidth=2,
    )
    ax.plot(
        layers,
        grad_results["relu"],
        marker="^",
        label="ReLU (Robust Flow)",
        color="teal",
        linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Layer Index (Input to Output)")
    ax.set_ylabel("Average Gradient Norm (Log Scale)")
    ax.set_title("Backpropagation Gradient Propagation Across Deep Layers")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(get_path("q8_plot1_gradient_decay.png"))
    plt.close()

    # Plot 2: Convergence Loss across Activations
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for act, hist in training_histories.items():
        ax2.plot(hist.history["loss"], label=f"{act.upper()} Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss Convergence with Varying Activations")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q8_plot2_activation_loss_curves.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    activations = ["sigmoid", "tanh", "relu"]
    grad_results = {}
    histories = {}

    for act in activations:
        m = build_deep_model(act)
        m.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        grad_results[act] = compute_layer_gradients(
            m, tf.constant(X_train[:64]), tf.constant(y_train[:64])
        )
        h = m.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
        )
        histories[act] = h

    generate_outputs(grad_results, histories)
    print("Lab 8 complete: 2 tables and 2 plots saved in Exp.No.08.")