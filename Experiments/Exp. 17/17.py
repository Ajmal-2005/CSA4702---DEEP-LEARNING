import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def create_multivariate_windows(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window, :])
        y.append(data[i + window, :])
    return np.array(X), np.array(y)


def load_trajectory_data(filename="q17_flight_trajectories.csv", window=10):
    df = pd.read_csv(get_path(filename))
    features = ["Lat_Disp", "Lon_Disp", "Altitude_hft"]
    values = df[features].values

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)

    X, y = create_multivariate_windows(scaled_values, window)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler, features


def build_stacked_lstm(window=10, n_features=3):
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(32, return_sequences=True),
        LSTM(16),
        Dense(16, activation="relu"),
        Dense(n_features),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def generate_outputs(history, y_true, y_pred, features):
    # Table 1: Feature-wise Errors
    feature_metrics = []
    for idx, col in enumerate(features):
        rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
        feature_metrics.append({
            "Trajectory_Feature": col,
            "RMSE": round(rmse, 4),
            "Baseline_Mean": round(np.mean(y_true[:, idx]), 2),
        })
    pd.DataFrame(feature_metrics).to_csv(
        get_path("q17_table1_trajectory_errors.csv"), index=False
    )

    # Plot 1: 2D Spatial Waypoint Forecast (Lat vs Lon)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        y_true[:, 0],
        y_true[:, 1],
        "o-",
        label="Actual Path",
        color="darkblue",
        markersize=4,
    )
    ax.plot(
        y_pred[:, 0],
        y_pred[:, 1],
        "s--",
        label="Stacked LSTM Forecast",
        color="crimson",
        markersize=4,
    )
    ax.set_xlabel("Latitude Displacement Coordinate")
    ax.set_ylabel("Longitude Displacement Coordinate")
    ax.set_title("Aircraft 2D Flight Trajectory Tracking")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q17_plot1_2d_trajectory.png"))
    plt.close()

    # Plot 2: Altitude Time Series Profile
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(y_true[:, 2], label="Actual Altitude", color="teal", linewidth=1.8)
    ax2.plot(
        y_pred[:, 2],
        label="Forecast Altitude",
        color="darkorange",
        linestyle="--",
        linewidth=1.8,
    )
    ax2.set_xlabel("Sequential Waypoint Step")
    ax2.set_ylabel("Altitude (hundreds of ft)")
    ax2.set_title("Vertical Flight Path Profile Forecast")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q17_plot2_altitude_profile.png"))
    plt.close()


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, scaler, features = load_trajectory_data()
    model = build_stacked_lstm()

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_te, y_te),
        epochs=40,
        batch_size=16,
        verbose=0,
    )

    p_scaled = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(p_scaled)
    y_true = scaler.inverse_transform(y_te)

    generate_outputs(history, y_true, y_pred, features)
    print("Lab 17 complete: 1 table and 2 plots saved in Exp.No.17.")