import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def create_sequences(data, window=15):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def load_stock_data(filename="q16_stock_price_history.csv", window=15):
    df = pd.read_csv(get_path(filename))
    prices = df["Stock_Close_Price"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, window)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler


def create_model(cell_type="lstm", window=15):
    model = Sequential([Input(shape=(window, 1))])
    if cell_type == "rnn":
        model.add(SimpleRNN(24))
    elif cell_type == "lstm":
        model.add(LSTM(24))
    elif cell_type == "gru":
        model.add(GRU(24))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def generate_outputs(histories, predictions, y_true):
    records = []
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        final_val_loss = histories[name].history["val_loss"][-1]
        records.append(
            {
                "Architecture": name.upper(),
                "Final_Val_Loss": round(final_val_loss, 5),
                "Test_RMSE ($)": round(rmse, 3),
            }
        )

    # Table 1: Model Comparison
    comp_df = pd.DataFrame(records)
    comp_df.to_csv(get_path("q16_table1_model_comparison.csv"), index=False)

    # Plot 1: Prediction Trajectories
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        y_true, label="Ground Truth Price", color="black", linewidth=2, alpha=0.7
    )
    colors = {"rnn": "crimson", "lstm": "teal", "gru": "darkorange"}
    for name, pred in predictions.items():
        ax.plot(
            pred,
            label=f"{name.upper()} Prediction",
            color=colors[name],
            linestyle="--",
        )
    ax.set_title("Stock Price Forecasting: Simple RNN vs LSTM vs GRU")
    ax.set_xlabel("Test Timeline (Trading Days)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q16_plot1_price_comparisons.png"))
    plt.close()

    # Plot 2: Convergence Loss
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        ax2.plot(hist.history["loss"], label=f"{name.upper()} Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Optimization Convergence Dynamics Across Cells")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q16_plot2_loss_curves.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_stock_data()
    architectures = ["rnn", "lstm", "gru"]

    histories = {}
    preds = {}

    for arch in architectures:
        m = create_model(cell_type=arch, window=15)
        h = m.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=35,
            batch_size=16,
            verbose=0,
        )
        histories[arch] = h
        p_scaled = m.predict(X_test, verbose=0)
        preds[arch] = scaler.inverse_transform(p_scaled)

    y_true_orig = scaler.inverse_transform(y_test)
    generate_outputs(histories, preds, y_true_orig)
    print("Lab 16 complete: 1 table and 2 plots saved in Exp.No.16.")