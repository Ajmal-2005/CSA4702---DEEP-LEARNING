import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tf.keras.utils.set_random_seed(42)


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


class Sampling(Layer):
    """Reparameterization trick: sample z = z_mean + exp(0.5 * z_log_var) * eps"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "recon_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}


def load_texture_data(filename="q27_microstructure_vae.csv"):
    df = pd.read_csv(get_path(filename))
    return df.values.astype(np.float32)


def build_vae(input_dim=64, latent_dim=2):
    # Encoder
    enc_inputs = Input(shape=(input_dim,))
    h = Dense(32, activation="relu")(enc_inputs)
    z_mean = Dense(latent_dim, name="z_mean")(h)
    z_log_var = Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    h_dec = Dense(32, activation="relu")(latent_inputs)
    dec_outputs = Dense(input_dim, activation="sigmoid")(h_dec)
    decoder = Model(latent_inputs, dec_outputs, name="decoder")

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(0.002))
    return vae, encoder, decoder


def generate_outputs(vae, encoder, decoder, X, history):
    rng = np.random.default_rng(42)

    # Table 1: Training Loss Breakdown (every 10th epoch and the final epoch)
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "Epoch"
    keep = sorted(set(range(0, len(hist_df), 10)) | {len(hist_df) - 1})
    hist_df.iloc[keep].round(4).to_csv(get_path("q27_table1_training_loss.csv"))

    # Table 2: Latent space distribution statistics (all training samples)
    z_mean_all, z_log_var_all, _ = encoder.predict(X, verbose=0)
    stats_df = pd.DataFrame(
        {
            "Latent_Dimension": ["z1", "z2"],
            "Mean_of_z_mean": z_mean_all.mean(axis=0).round(4),
            "Std_of_z_mean": z_mean_all.std(axis=0).round(4),
            "Mean_Posterior_Std": np.exp(0.5 * z_log_var_all).mean(axis=0).round(4),
            "Prior_Mean": [0.0, 0.0],
            "Prior_Std": [1.0, 1.0],
        }
    )
    stats_df.to_csv(get_path("q27_table2_latent_statistics.csv"), index=False)

    # Table 3: Latent coordinates of the first 10 textures
    z_df = pd.DataFrame(z_mean_all[:10], columns=["Latent_Dim_1 (z1)", "Latent_Dim_2 (z2)"])
    z_df.to_csv(get_path("q27_table3_latent_samples.csv"), index=False)

    # Table 4: Reconstruction quality
    recon = decoder.predict(z_mean_all, verbose=0)
    mse = float(np.mean((X - recon) ** 2))
    prior_samples = rng.normal(size=(len(X), 2)).astype(np.float32)
    gen = decoder.predict(prior_samples, verbose=0)
    quality_df = pd.DataFrame(
        {
            "Metric": [
                "Reconstruction MSE (pixel level)",
                "Mean pixel value: real textures",
                "Mean pixel value: generated textures",
                "Pixel std: real textures",
                "Pixel std: generated textures",
            ],
            "Value": [
                round(mse, 4),
                round(float(X.mean()), 4),
                round(float(gen.mean()), 4),
                round(float(X.std()), 4),
                round(float(gen.std()), 4),
            ],
        }
    )
    quality_df.to_csv(get_path("q27_table4_reconstruction_quality.csv"), index=False)

    # Plot 1: 2D Latent Space Projection
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z_mean_all[:, 0], z_mean_all[:, 1], c=z_mean_all[:, 0], cmap="viridis", alpha=0.7)
    fig.colorbar(scatter, label="z1 value")
    ax.set_xlabel("Latent Coordinate $z_1$")
    ax.set_ylabel("Latent Coordinate $z_2$")
    ax.set_title("VAE Latent Space Distribution")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q27_plot1_latent_space.png"))
    plt.close()

    # Plot 2: Synthetic Samples Generated by Sampling Prior N(0, 1)
    grid_size = 4
    norm_samples = rng.normal(size=(grid_size * grid_size, 2)).astype(np.float32)
    generated_imgs = decoder.predict(norm_samples, verbose=0)
    fig2, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_imgs[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(r"Synthetic Textures Sampled from VAE Prior $\mathcal{N}(0, I)$")
    plt.tight_layout()
    plt.savefig(get_path("q27_plot2_synthetic_textures.png"))
    plt.close()

    # Plot 3: Original vs Reconstructed textures
    n_show = 6
    idx = rng.choice(len(X), n_show, replace=False)
    fig3, axes3 = plt.subplots(2, n_show, figsize=(10, 3.6))
    for j, k in enumerate(idx):
        axes3[0, j].imshow(X[k].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes3[0, j].axis("off")
        axes3[1, j].imshow(recon[k].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes3[1, j].axis("off")
    axes3[0, 0].set_title("Original", loc="left", fontsize=10)
    axes3[1, 0].set_title("Reconstructed", loc="left", fontsize=10)
    plt.suptitle("VAE Reconstruction of Surface Textures")
    plt.tight_layout()
    plt.savefig(get_path("q27_plot3_reconstruction.png"))
    plt.close()


if __name__ == "__main__":
    X_data = load_texture_data()
    vae_model, enc, dec = build_vae()
    hist = vae_model.fit(X_data, epochs=60, batch_size=16, verbose=0)
    generate_outputs(vae_model, enc, dec, X_data, hist)
    print("Lab 27 complete: 4 tables and 3 plots saved in Exp.No.27.")
