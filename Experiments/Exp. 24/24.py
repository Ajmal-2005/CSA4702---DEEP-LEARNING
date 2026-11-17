import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_hmm_sequence(filename="q24_tool_wear_sequence.csv"):
    df = pd.read_csv(get_path(filename))
    observations = df["Observed_Vibration"].values
    true_states = df["True_Tool_State"].values
    return df, observations, true_states


def viterbi_decode(obs, pi, A, B):
    n_states = A.shape[0]
    T = len(obs)
    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)

    # Initialization
    delta[0, :] = np.log(pi + 1e-12) + np.log(B[:, obs[0]] + 1e-12)

    # Recursion
    for t in range(1, T):
        for j in range(n_states):
            trans_prob = delta[t - 1, :] + np.log(A[:, j] + 1e-12)
            best_prev = np.argmax(trans_prob)
            psi[t, j] = best_prev
            delta[t, j] = trans_prob[best_prev] + np.log(B[j, obs[t]] + 1e-12)

    # Backtracking path
    q_opt = np.zeros(T, dtype=int)
    q_opt[-1] = np.argmax(delta[-1, :])
    for t in range(T - 2, -1, -1):
        q_opt[t] = psi[t + 1, q_opt[t + 1]]

    return q_opt


def generate_outputs(df, decoded_states, true_states):
    acc = np.mean(decoded_states == true_states)
    state_names = ["Healthy", "Degraded", "Failing"]

    # Table 1: Decoding Summary
    sum_df = pd.DataFrame(
        {
            "Metric": ["Total Waypoints", "Exact State Match Rate", "Error Steps"],
            "Value": [len(true_states), f"{acc * 100:.2f}%", np.sum(decoded_states != true_states)],
        }
    )
    sum_df.to_csv(get_path("q24_table1_viterbi_summary.csv"), index=False)

    # Table 2: Step-by-Step Sample Log
    log_df = pd.DataFrame(
        {
            "Step": df["Step"][:25],
            "Observed_Signal": df["Observed_Vibration"][:25],
            "True_Wear_State": [state_names[s] for s in true_states[:25]],
            "Decoded_Wear_State": [state_names[s] for s in decoded_states[:25]],
            "Match": true_states[:25] == decoded_states[:25],
        }
    )
    log_df.to_csv(get_path("q24_table2_sample_state_sequence.csv"), index=False)

    # Plot 1: Decoded vs True Hidden States
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(
        range(1, 101),
        true_states[:100],
        where="mid",
        label="True Latent Wear State",
        color="darkblue",
        linewidth=2,
    )
    ax.step(
        range(1, 101),
        decoded_states[:100],
        where="mid",
        label="Viterbi Decoded State",
        color="crimson",
        linestyle="--",
        linewidth=2,
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Time Step")
    ax.set_title("Viterbi Hidden State Decoding: Tool Mechanical Degradation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q24_plot1_viterbi_sequence.png"))
    plt.close()

    # Plot 2: Observation vs Latent State Overlap
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        df["Observed_Vibration"][:60],
        "o-",
        label="Observed Vibration (Low=0, Mid=1, High=2)",
        color="goldenrod",
        alpha=0.7,
    )
    ax2.step(
        range(60),
        decoded_states[:60],
        where="mid",
        label="Decoded Wear Regime",
        color="black",
        linewidth=2,
    )
    ax2.set_yticks([0, 1, 2])
    ax2.set_xlabel("Time Step")
    ax2.set_title("Observed Sensor Emissions vs Inferred Tool Degradation")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q24_plot2_observations_vs_states.png"))
    plt.close()


if __name__ == "__main__":
    df_seq, obs_series, true_s = load_hmm_sequence()

    # HMM Matrices
    pi = np.array([0.9, 0.08, 0.02])
    A = np.array([[0.85, 0.12, 0.03], [0.0, 0.80, 0.20], [0.0, 0.0, 1.0]])
    B = np.array([[0.70, 0.25, 0.05], [0.15, 0.65, 0.20], [0.05, 0.25, 0.70]])

    decoded_s = viterbi_decode(obs_series, pi, A, B)
    generate_outputs(df_seq, decoded_s, true_s)
    print("Lab 24 complete: 2 tables and 2 plots saved in Exp.No.24.")