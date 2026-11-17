import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def create_sliding_windows(series, window_size=14):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def load_energy_data(filename="q15_daily_electricity.csv", window=14):
    df = pd.read_csv(get_path(filename))
    vals = df["Daily_Consumption_kWh"].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(vals)

    X, y = create_sliding_windows(scaled_vals, window_size=window)
    train_size = int(len(X) * 0.8)
    return (
        X[:train_size],
        X[train_size:],
        y[:train_size],
        y[train_size:],
        scaler,
    )


def build_forecaster(window=14):
    model = Sequential([
        Input(shape=(window, 1)),
        SimpleRNN(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def generate_outputs(history, y_true_orig, y_pred_orig):
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

    # Table 1: Performance Summary
    summary_df = pd.DataFrame(
        {
            "Forecast_Horizon": ["Next-Day Load"],
            "MAE_kWh": [round(mae, 3)],
            "RMSE_kWh": [round(rmse, 3)],
            "Mean_Daily_Demand": [round(np.mean(y_true_orig), 2)],
        }
    )
    summary_df.to_csv(get_path("q15_table1_forecast_metrics.csv"), index=False)

    # Table 2: Step-by-Step Log
    log_df = pd.DataFrame(
        {
            "Day": range(1, len(y_true_orig) + 1),
            "Actual_Consumption": y_true_orig.flatten().round(2),
            "Forecasted_Consumption": y_pred_orig.flatten().round(2),
            "Residual": (
                y_true_orig.flatten() - y_pred_orig.flatten()
            ).round(2),
        }
    )
    log_df.head(20).to_csv(
        get_path("q15_table2_daily_forecast_log.csv"), index=False
    )

    # Plot 1: Time Series Forecast
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        y_true_orig,
        label="Actual Consumption",
        color="steelblue",
        linewidth=1.8,
    )
    ax.plot(
        y_pred_orig,
        label="RNN Forecast",
        color="crimson",
        linestyle="--",
        linewidth=1.8,
    )
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Electricity Demand (kWh)")
    ax.set_title("Daily Household Power Consumption: Actual vs RNN Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q15_plot1_load_forecast.png"))
    plt.close()

    # Plot 2: Residual Distribution
    residuals = y_true_orig - y_pred_orig
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(residuals, bins=15, color="darkseagreen", edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Prediction Residual (kWh)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Forecast Error Residual Distribution")
    plt.tight_layout()
    plt.savefig(get_path("q15_plot2_residuals.png"))
    plt.close()


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, scaler = load_energy_data()
    model = build_forecaster(window=14)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_te, y_te),
        epochs=45,
        batch_size=16,
        verbose=0,
    )

    preds_scaled = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_te)

    generate_outputs(history, y_true, y_pred)
    print("Lab 15 complete: 2 tables and 2 plots saved in Exp.No.15.")