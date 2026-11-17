import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_digit_data(filename="q9_digits_sample.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"pixel_{i}" for i in range(64)]].values / 16.0
    y = df["label"].values
    # One-hot encode targets
    num_classes = 10
    Y_onehot = np.eye(num_classes)[y]
    return train_test_split(X, Y_onehot, y, test_size=0.2, random_state=42)


# Hand-crafted 2-layer neural network with explicit forward & backward formulas
class ManualNeuralNetwork:

    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10, lr=0.05):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.05
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.05
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_deriv(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Track weight updates magnitude
        return np.linalg.norm(dW1), np.linalg.norm(dW2)


def generate_outputs(epoch_logs, sample_predictions):
    log_df = pd.DataFrame(epoch_logs)
    log_df.to_csv(get_path("q9_table1_training_log.csv"), index=False)

    pred_df = pd.DataFrame(sample_predictions)
    pred_df.to_csv(get_path("q9_table2_sample_digit_preds.csv"), index=False)

    # Plot 1: Loss & Accuracy Curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(log_df["Epoch"], log_df["Loss"], color="crimson", linewidth=2)
    ax[0].set_title("Cross-Entropy Loss vs Epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(alpha=0.3)

    ax[1].plot(log_df["Epoch"], log_df["Accuracy"], color="teal", linewidth=2)
    ax[1].set_title("Classification Accuracy vs Epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q9_plot1_loss_accuracy.png"))
    plt.close()

    # Plot 2: Gradient Norm Updates
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(
        log_df["Epoch"],
        log_df["dW1_Norm"],
        label="Hidden Layer Grad Norm (dW1)",
        color="royalblue",
    )
    ax2.plot(
        log_df["Epoch"],
        log_df["dW2_Norm"],
        label="Output Layer Grad Norm (dW2)",
        color="darkorange",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Norm Magnitude")
    ax2.set_title("Weight Update Magnitude via Backpropagation")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q9_plot2_gradient_magnitudes.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, y_raw_train, y_raw_test = (
        load_digit_data()
    )
    nn = ManualNeuralNetwork(
        input_dim=64, hidden_dim=32, output_dim=10, lr=0.1
    )

    epochs = 120
    epoch_logs = []

    for ep in range(1, epochs + 1):
        probs = nn.forward(X_train)
        loss = -np.mean(np.sum(Y_train * np.log(probs + 1e-12), axis=1))
        dw1_n, dw2_n = nn.backward(X_train, Y_train)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y_raw_train)

        if ep % 5 == 0 or ep == 1:
            epoch_logs.append({
                "Epoch": ep,
                "Loss": round(loss, 4),
                "Accuracy": round(acc, 4),
                "dW1_Norm": round(dw1_n, 5),
                "dW2_Norm": round(dw2_n, 5),
            })

    test_probs = nn.forward(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    sample_preds = [
        {
            "Sample_ID": i + 1,
            "True_Digit": y_raw_test[i],
            "Predicted_Digit": test_preds[i],
            "Confidence": round(test_probs[i, test_preds[i]], 4),
            "Correct": y_raw_test[i] == test_preds[i],
        }
        for i in range(12)
    ]

    generate_outputs(epoch_logs, sample_preds)
    print("Lab 9 complete: 2 tables and 2 plots saved in Exp.No.09.")