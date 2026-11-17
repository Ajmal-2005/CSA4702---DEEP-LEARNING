import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def prepare_seq2seq_data(filename="q18_query_translation_pairs.csv"):
    df = pd.read_csv(get_path(filename))
    src_texts = df["Source_Query"].tolist()
    tgt_texts = df["Target_Translation"].tolist()

    src_tok = Tokenizer(filters="")
    src_tok.fit_on_texts(src_texts)
    src_seq = pad_sequences(
        src_tok.texts_to_sequences(src_texts), padding="post"
    )

    tgt_tok = Tokenizer(filters="")
    tgt_tok.fit_on_texts(tgt_texts)
    tgt_seq = pad_sequences(
        tgt_tok.texts_to_sequences(tgt_texts), padding="post"
    )

    # Decoder targets shifted by 1 timestep forward
    decoder_target_data = np.zeros_like(tgt_seq)
    decoder_target_data[:, :-1] = tgt_seq[:, 1:]

    return src_seq, tgt_seq, decoder_target_data, src_tok, tgt_tok, df


def build_seq2seq(
    src_vocab_size, tgt_vocab_size, latent_dim=32, max_src=8, max_tgt=10
):
    # Encoder
    encoder_inputs = Input(shape=(None,), name="enc_input")
    enc_emb = Embedding(src_vocab_size, latent_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name="dec_input")
    dec_emb = Embedding(tgt_vocab_size, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(tgt_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_outputs(history, df, src_tok, tgt_tok):
    # Table 1: Training Progression
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "Epoch"
    hist_df.iloc[::5].to_csv(get_path("q18_table1_training_progression.csv"))

    # Table 2: Sample Parallel Corpus
    df.head(10).to_csv(get_path("q18_table2_sample_corpus.csv"), index=False)

    # Plot 1: Seq2Seq Accuracy Curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        history.history["accuracy"],
        label="Decoder Token Accuracy",
        color="teal",
        linewidth=2,
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Categorical Accuracy")
    ax.set_title("Seq2Seq Helpdesk Query Translation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q18_plot1_seq2seq_accuracy.png"))
    plt.close()

    # Plot 2: Seq2Seq Loss Curve
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(
        history.history["loss"],
        label="Cross-Entropy Loss",
        color="crimson",
        linewidth=2,
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Token Loss")
    ax2.set_title("Seq2Seq Translation Loss Optimization")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q18_plot2_seq2seq_loss.png"))
    plt.close()


if __name__ == "__main__":
    src_seq, tgt_seq, dec_tgt, src_tok, tgt_tok, df = prepare_seq2seq_data()
    src_v = len(src_tok.word_index) + 1
    tgt_v = len(tgt_tok.word_index) + 1

    model = build_seq2seq(src_v, tgt_v)

    history = model.fit(
        [src_seq, tgt_seq],
        np.expand_dims(dec_tgt, -1),
        epochs=60,
        batch_size=16,
        verbose=0,
    )

    generate_outputs(history, df, src_tok, tgt_tok)
    print("Lab 18 complete: 2 tables and 2 plots saved in Exp.No.18.")