import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Sequential

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tf.keras.utils.set_random_seed(42)


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_silhouettes(filename="q28_product_silhouettes_gan.csv"):
    df = pd.read_csv(get_path(filename))
    return df.values.astype(np.float32)


def build_generator(latent_dim=16):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(64),
        LeakyReLU(negative_slope=0.2),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(64, activation="sigmoid"),
    ])
    return model


def build_discriminator(input_dim=64):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(64),
        LeakyReLU(negative_slope=0.2),
        Dense(1, activation="sigmoid"),
    ])
    return model


def train_gan(X_real, steps=3000, batch_size=32, latent_dim=16):
    """Adversarial training with an explicit GradientTape loop. The discriminator update
    changes only the discriminator weights and the generator update changes only the
    generator weights, so no trainable-flag switching is needed."""
    gen = build_generator(latent_dim)
    disc = build_discriminator()
    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    real_data = tf.constant(X_real, dtype=tf.float32)
    m = len(X_real)

    @tf.function
    def train_step():
        idx = tf.random.uniform((batch_size,), 0, m, dtype=tf.int32)
        real_imgs = tf.gather(real_data, idx)

        # Discriminator step (one-sided label smoothing: real label 0.9)
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as d_tape:
            fake_imgs = gen(noise, training=True)
            d_real = disc(real_imgs, training=True)
            d_fake = disc(fake_imgs, training=True)
            d_loss = 0.5 * (
                bce(tf.ones_like(d_real) * 0.9, d_real) + bce(tf.zeros_like(d_fake), d_fake)
            )
        d_grads = d_tape.gradient(d_loss, disc.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))

        # Generator step (non-saturating loss: generator wants D(fake) -> 1)
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as g_tape:
            fake_imgs = gen(noise, training=True)
            g_out = disc(fake_imgs, training=False)
            g_loss = bce(tf.ones_like(g_out), g_out)
        g_grads = g_tape.gradient(g_loss, gen.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))
        return d_loss, g_loss

    d_losses, g_losses = [], []
    for _ in range(steps):
        d_l, g_l = train_step()
        d_losses.append(float(d_l))
        g_losses.append(float(g_l))
    return gen, disc, d_losses, g_losses


def trend_label(series, window=500):
    first, last = np.mean(series[:window]), np.mean(series[-window:])
    if last < first * 0.95:
        return "Decreasing"
    if last > first * 1.05:
        return "Increasing"
    return "Stable"


def generate_outputs(gen, disc, X_real, d_losses, g_losses, latent_dim=16):
    d_arr, g_arr = np.array(d_losses), np.array(g_losses)
    steps = len(d_arr)

    # Table 1: Training Progression (every 100th step and the final step)
    keep = sorted(set([0] + list(range(99, steps, 100)) + [steps - 1]))
    loss_df = pd.DataFrame(
        {
            "Epoch": [k + 1 for k in keep],
            "Discriminator_Loss": d_arr[keep].round(4),
            "Generator_Loss": g_arr[keep].round(4),
        }
    )
    loss_df.to_csv(get_path("q28_table1_adversarial_losses.csv"), index=False)

    # Table 2: Final Model Status
    mean_d_last = float(np.mean(d_arr[-500:]))
    summary_df = pd.DataFrame(
        {
            "Component": ["Discriminator", "Generator"],
            "Final_Loss": [round(float(d_arr[-1]), 4), round(float(g_arr[-1]), 4)],
            "Mean_Loss_Last_500": [round(mean_d_last, 4), round(float(np.mean(g_arr[-500:])), 4)],
            "Trend": [trend_label(d_arr), trend_label(g_arr)],
            "Status": [
                "Near equilibrium" if abs(mean_d_last - 0.693) < 0.15 else "Not at equilibrium",
                "Active gradient",
            ],
        }
    )
    summary_df.to_csv(get_path("q28_table2_equilibrium_summary.csv"), index=False)

    # Table 3: Real vs generated image statistics
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(len(X_real), latent_dim)).astype(np.float32)
    fake = gen(noise, training=False).numpy()
    d_real = float(np.mean(disc(X_real, training=False).numpy()))
    d_fake = float(np.mean(disc(fake, training=False).numpy()))
    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Mean pixel value",
                "Pixel standard deviation",
                "Mean discriminator output",
                "Mean absolute difference of mean images",
            ],
            "Real": [round(float(X_real.mean()), 4), round(float(X_real.std()), 4), round(d_real, 4), 0.0],
            "Generated": [
                round(float(fake.mean()), 4),
                round(float(fake.std()), 4),
                round(d_fake, 4),
                round(float(np.mean(np.abs(X_real.mean(axis=0) - fake.mean(axis=0)))), 4),
            ],
        }
    )
    stats_df.to_csv(get_path("q28_table3_real_vs_generated.csv"), index=False)

    # Plot 1: Adversarial Loss Curves (raw + moving average)
    def smooth(a, w=50):
        return np.convolve(a, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d_arr, color="crimson", alpha=0.2)
    ax.plot(g_arr, color="teal", alpha=0.2)
    ax.plot(np.arange(49, steps), smooth(d_arr), label="Discriminator Loss", color="crimson", linewidth=2)
    ax.plot(np.arange(49, steps), smooth(g_arr), label="Generator Loss", color="teal", linewidth=2)
    ax.axhline(0.693, color="gray", linestyle="--", label="Theoretical Equilibrium (-ln 0.5)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("GAN Adversarial Optimization Dynamics")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q28_plot1_loss_dynamics.png"))
    plt.close()

    # Plot 2: Real vs generated silhouette patches
    fig2, axes = plt.subplots(4, 8, figsize=(10, 5.4))
    real_idx = rng.choice(len(X_real), 16, replace=False)
    for i in range(16):
        r, c = divmod(i, 4)
        axes[r, c].imshow(X_real[real_idx[i]].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[r, c + 4].imshow(fake[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].set_title("Real patches", loc="left", fontsize=10)
    axes[0, 4].set_title("GAN generated patches", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(get_path("q28_plot2_generated_images.png"))
    plt.close()


if __name__ == "__main__":
    X_data = load_silhouettes()
    g_net, d_net, d_hist, g_hist = train_gan(X_data, steps=3000)
    generate_outputs(g_net, d_net, X_data, d_hist, g_hist)
    print("Lab 28 complete: 3 tables and 2 plots saved in Exp.No.28.")
