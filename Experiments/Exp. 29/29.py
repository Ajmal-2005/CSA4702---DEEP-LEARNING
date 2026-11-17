import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LeakyReLU, Reshape
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tf.keras.utils.set_random_seed(42)

DEFAULT_CLASS_NAMES = ["Speed_Limit (Ring)", "Caution (Triangle)", "Stop (Block)"]


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_hazard_signs(filename="q29_hazard_signs_cgan.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"p_{i}" for i in range(64)]].values.astype(np.float32)
    y = df["Sign_Class"].values.astype(np.int32)
    return X, y


def build_cgan_models(latent_dim=16, num_classes=3):
    # Generator: noise + class embedding -> 8x8 image (64 values)
    noise_in = Input(shape=(latent_dim,))
    label_in = Input(shape=(1,), dtype="int32")
    label_emb = Reshape((latent_dim,))(Embedding(num_classes, latent_dim)(label_in))
    gen_in = Concatenate()([noise_in, label_emb])
    h_g = LeakyReLU(negative_slope=0.2)(Dense(64)(gen_in))
    h_g = LeakyReLU(negative_slope=0.2)(Dense(128)(h_g))
    out_g = Dense(64, activation="sigmoid")(h_g)
    generator = Model([noise_in, label_in], out_g, name="cgen")

    # Discriminator: image + class embedding -> real/fake probability
    img_in = Input(shape=(64,))
    disc_label_in = Input(shape=(1,), dtype="int32")
    disc_label_emb = Reshape((64,))(Embedding(num_classes, 64)(disc_label_in))
    disc_in = Concatenate()([img_in, disc_label_emb])
    h_d = LeakyReLU(negative_slope=0.2)(Dense(128)(disc_in))
    h_d = LeakyReLU(negative_slope=0.2)(Dense(64)(h_d))
    out_d = Dense(1, activation="sigmoid")(h_d)
    discriminator = Model([img_in, disc_label_in], out_d, name="cdisc")
    return generator, discriminator


def train_cgan(X_real, y_real, steps=5000, batch_size=32, latent_dim=16):
    """Conditional GAN trained with an explicit GradientTape loop. The discriminator update
    changes only the discriminator weights and the generator update changes only the
    generator weights, so no trainable-flag switching is needed."""
    num_classes = int(y_real.max()) + 1
    gen, disc = build_cgan_models(latent_dim, num_classes)
    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    real_data = tf.constant(X_real, dtype=tf.float32)
    real_labels = tf.constant(y_real.reshape(-1, 1), dtype=tf.int32)
    m = len(X_real)

    @tf.function
    def train_step():
        idx = tf.random.uniform((batch_size,), 0, m, dtype=tf.int32)
        real_imgs = tf.gather(real_data, idx)
        labels = tf.gather(real_labels, idx)

        # Discriminator step (real label smoothed to 0.9, fakes use the same class labels)
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as d_tape:
            fake_imgs = gen([noise, labels], training=True)
            d_real = disc([real_imgs, labels], training=True)
            d_fake = disc([fake_imgs, labels], training=True)
            d_loss = 0.5 * (
                bce(tf.ones_like(d_real) * 0.9, d_real) + bce(tf.zeros_like(d_fake), d_fake)
            )
        d_grads = d_tape.gradient(d_loss, disc.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))

        # Generator step with random class labels and fresh noise
        noise = tf.random.normal((batch_size, latent_dim))
        sampled = tf.random.uniform((batch_size, 1), 0, num_classes, dtype=tf.int32)
        with tf.GradientTape() as g_tape:
            fake_imgs = gen([noise, sampled], training=True)
            g_out = disc([fake_imgs, sampled], training=False)
            g_loss = bce(tf.ones_like(g_out), g_out)
        g_grads = g_tape.gradient(g_loss, gen.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))
        return d_loss, g_loss

    d_losses, g_losses = [], []
    for _ in range(steps):
        d_l, g_l = train_step()
        d_losses.append(float(d_l))
        g_losses.append(float(g_l))
    return gen, d_losses, g_losses, num_classes


def generate_outputs(gen, X_real, y_real, d_losses, g_losses, num_classes, latent_dim=16):
    class_names = DEFAULT_CLASS_NAMES if num_classes == len(DEFAULT_CLASS_NAMES) else [f"Class_{i}" for i in range(num_classes)]
    rng = np.random.default_rng(42)
    steps = len(d_losses)

    # Table 1: Loss Progression (every 250th step and the final step)
    keep = sorted(set([0] + list(range(249, steps, 250)) + [steps - 1]))
    loss_df = pd.DataFrame(
        {
            "Epoch": [k + 1 for k in keep],
            "D_Loss": np.round(np.array(d_losses)[keep], 4),
            "G_Loss": np.round(np.array(g_losses)[keep], 4),
        }
    )
    loss_df.to_csv(get_path("q29_table1_cgan_loss.csv"), index=False)

    # Table 2: Class-control check. For each class, the mean generated image should
    # match the mean real image of the SAME class better than the other classes.
    real_means = [X_real[y_real == c].mean(axis=0) for c in range(num_classes)]
    gen_samples = {}
    for c in range(num_classes):
        noise = rng.normal(size=(200, latent_dim)).astype(np.float32)
        labels = np.full((200, 1), c, dtype=np.int32)
        gen_samples[c] = gen([noise, labels], training=False).numpy()

    rows = []
    for c in range(num_classes):
        g_mean = gen_samples[c].mean(axis=0)
        corrs = [np.corrcoef(g_mean, real_means[k])[0, 1] for k in range(num_classes)]
        own = corrs[c]
        other = max(v for k, v in enumerate(corrs) if k != c)
        rows.append(
            {
                "Class": class_names[c],
                "Real_Samples": int((y_real == c).sum()),
                "Own_Class_Correlation": round(float(own), 4),
                "Best_Other_Class_Correlation": round(float(other), 4),
                "Class_Controlled": "Yes" if own > other else "No",
            }
        )
    pd.DataFrame(rows).to_csv(get_path("q29_table2_class_control.csv"), index=False)

    # Plot 1: Loss Progression Curve (raw + moving average)
    def smooth(a, w=50):
        return np.convolve(a, np.ones(w) / w, mode="valid")

    d_arr, g_arr = np.array(d_losses), np.array(g_losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d_arr, color="royalblue", alpha=0.2)
    ax.plot(g_arr, color="darkorange", alpha=0.2)
    ax.plot(np.arange(49, steps), smooth(d_arr), label="Conditional Discriminator Loss", color="royalblue", linewidth=2)
    ax.plot(np.arange(49, steps), smooth(g_arr), label="Conditional Generator Loss", color="darkorange", linewidth=2)
    ax.axhline(0.693, color="gray", linestyle="--", label="Equilibrium (-ln 0.5)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("cGAN Loss Dynamics for Class-Conditioned Synthesis")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q29_plot1_cgan_loss.png"))
    plt.close()

    # Plot 2: Real (left 4) vs generated (right 4) signs for every class
    fig2, axes = plt.subplots(num_classes, 8, figsize=(11, 2.1 * num_classes + 0.8))
    for c in range(num_classes):
        real_idx = rng.choice(np.where(y_real == c)[0], 4, replace=False)
        for col in range(4):
            axes[c, col].imshow(X_real[real_idx[col]].reshape(8, 8), cmap="magma", vmin=0, vmax=1)
            axes[c, col + 4].imshow(gen_samples[c][col].reshape(8, 8), cmap="magma", vmin=0, vmax=1)
        axes[c, 0].set_title(f"{class_names[c]} - Real", loc="left", fontsize=9, fontweight="bold")
        axes[c, 4].set_title(f"{class_names[c]} - Generated", loc="left", fontsize=9, fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle("cGAN Synthesized Signs Conditioned on Category")
    plt.tight_layout()
    plt.savefig(get_path("q29_plot2_class_samples.png"))
    plt.close()


if __name__ == "__main__":
    X_signs, y_signs = load_hazard_signs()
    generator, d_loss_hist, g_loss_hist, n_classes = train_cgan(X_signs, y_signs)
    generate_outputs(generator, X_signs, y_signs, d_loss_hist, g_loss_hist, n_classes)
    print("Lab 29 complete: 2 tables and 2 plots saved in Exp.No.29.")
