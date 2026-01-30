import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Dataset Loading & Preprocessing
def load_and_preprocess_text(file_path):
    """
    Loads a text file, converts text to lowercase,
    and removes punctuation.

    Args:
        file_path (str): Path to text dataset

    Returns:
        str: Cleaned text
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Dataset not found. Download from "
            "https://www.gutenberg.org/files/100/100-0.txt"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    text = re.sub(r"[^a-z\s]", "", text)
    return text



# Sequence Generation
def create_sequences(text, seq_length=30, step=8):
    """
    Creates input-output sequences for character-level prediction.

    Args:
        text (str): Cleaned dataset text
        seq_length (int): Length of input sequences
        step (int): Step size between sequences

    Returns:
        X, y, char_to_idx, idx_to_char, vocab_size
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    sequences = []
    next_chars = []

    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])

    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool_)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1

    return X, y, char_to_idx, idx_to_char, vocab_size



# Model Definition
def build_model(seq_length, vocab_size):
    """
    Builds and compiles an LSTM model.

    Args:
        seq_length (int): Input sequence length
        vocab_size (int): Number of unique characters

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(64, input_shape=(seq_length, vocab_size)),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    return model



# Model Training
def train_model(model, X, y, epochs=3, batch_size=128):
    """
    Trains the LSTM model with early stopping.

    Args:
        model: Keras model
        X: Input sequences
        y: Target outputs
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stop]
    )


# Text Generation
def sample(preds, temperature=0.8):
    """
    Samples the next character index using temperature scaling.
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_text(model, seed, char_to_idx, idx_to_char,
                  seq_length=30, length=120):
    """
    Generates text from a trained model.

    Args:
        model: Trained Keras model
        seed (str): Seed text
        length (int): Number of characters to generate

    Returns:
        str: Generated text
    """
    generated = seed.lower()

    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(char_to_idx)))

        for t, char in enumerate(generated[-seq_length:]):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds)
        next_char = idx_to_char[next_idx]
        generated += next_char

    return generated



# Main Execution
def main():
    text = load_and_preprocess_text("shakespeare.txt")

    X, y, char_to_idx, idx_to_char, vocab_size = create_sequences(text)

    model = build_model(seq_length=30, vocab_size=vocab_size)
    model.summary()

    train_model(model, X, y)

    print("\nTraining completed. Starting text generation...\n")

    # Sample Outputs
    seeds = [
        "to be or not to be ",
        "all the worlds a stage ",
        "love looks not with the eyes "
    ]

    for seed in seeds:
        print("\nSeed:", seed)
        print(generate_text(model, seed, char_to_idx, idx_to_char))


if __name__ == "__main__":
    main()
