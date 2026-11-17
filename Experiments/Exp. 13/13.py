import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def load_furnace_data(filename="q13_furnace_temperature.csv", window_size=10):
    df = pd.read_csv(get_path(filename))
    values = df["Furnace_Temperature"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    X, y = create_sequences(scaled_values, window_size)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler, df


def build_rnn_model(window_size=10):
    model = Sequential([
        Input(shape=(window_size, 1)),
        SimpleRNN(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def generate_outputs(history, y_test, y_pred, df):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Table 1: Metrics
    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE", "MSE", "RMSE"],
            "Value": [round(mae, 3), round(mse, 3), round(rmse, 3)],
            "Unit": ["°C", "°C²", "°C"],
        }
    )
    metrics_df.to_csv(get_path("q13_table1_error_metrics.csv"), index=False)

    # Table 2: Sample predictions
    sample_df = pd.DataFrame(
        {
            "Actual_Temperature_C": y_test[:15].flatten().round(2),
            "Predicted_Temperature_C": y_pred[:15].flatten().round(2),
            "Absolute_Error_C": np.abs(
                y_test[:15].flatten() - y_pred[:15].flatten()
            ).round(2),
        }
    )
    sample_df.to_csv(
        get_path("q13_table2_sample_predictions.csv"), index=False
    )

    # Plot 1: Prediction Tracking
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y_test, label="Actual Temperature", color="royalblue", linewidth=1.5)
    ax.plot(
        y_pred,
        label="RNN Prediction",
        color="crimson",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_title("Industrial Furnace Temperature Forecast (Test Window)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q13_plot1_temperature_tracking.png"))
    plt.close()

    # Plot 2: Training Loss Curve
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(history.history["loss"], label="Train MSE", color="teal")
    ax2.plot(
        history.history["val_loss"],
        label="Val MSE",
        color="darkorange",
        linestyle="--",
    )
    ax2.set_title("Simple RNN Convergence Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Mean Squared Error")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q13_plot2_training_loss.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = load_furnace_data()
    model = build_rnn_model(window_size=10)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=16,
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(pred_scaled)
    y_test_orig = scaler.inverse_transform(y_test)

    generate_outputs(history, y_test_orig, y_pred, df)
    print("Lab 13 complete: 2 tables and 2 plots saved in Exp.No.13.")