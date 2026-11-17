import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def load_sentiment_corpus(filename="q14_financial_sentiment.csv"):
    df = pd.read_csv(get_path(filename))
    texts = df["Headline"].values
    labels = df["Sentiment_Label"].values

    tokenizer = Tokenizer(num_words=500, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=15, padding="post")
    y = to_categorical(labels, num_classes=3)

    return train_test_split(X, y, labels, test_size=0.25, random_state=42)


def build_text_rnn(vocab_size=500, maxlen=15):
    model = Sequential([
        Input(shape=(maxlen,)),
        Embedding(vocab_size, 16),
        SimpleRNN(32),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_outputs(history, y_test_raw, y_pred_classes):
    target_names = ["Bullish", "Bearish", "Neutral"]

    # Table 1: Classification Metrics
    rep_dict = classification_report(
        y_test_raw, y_pred_classes, target_names=target_names, output_dict=True
    )
    pd.DataFrame(rep_dict).transpose().to_csv(
        get_path("q14_table1_classification_report.csv")
    )

    # Table 2: Confusion Matrix
    cm = confusion_matrix(y_test_raw, y_pred_classes)
    pd.DataFrame(cm, columns=target_names, index=target_names).to_csv(
        get_path("q14_table2_confusion_matrix.csv")
    )

    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    fig.colorbar(cax)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontweight="bold",
            )
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Financial Sentiment Confusion Matrix", pad=15)
    plt.tight_layout()
    plt.savefig(get_path("q14_plot1_confusion_matrix.png"))
    plt.close()

    # Plot 2: Training History
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(
        history.history["accuracy"], label="Train Accuracy", color="teal"
    )
    ax2.plot(
        history.history["val_accuracy"],
        label="Val Accuracy",
        color="crimson",
        linestyle="--",
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Text RNN Sentiment Classification Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q14_plot2_accuracy_history.png"))
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = (
        load_sentiment_corpus()
    )
    model = build_text_rnn()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=8,
        verbose=0,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    generate_outputs(history, y_test_raw, y_pred_classes)
    print("Lab 14 complete: 2 tables and 2 plots saved in Exp.No.14.")