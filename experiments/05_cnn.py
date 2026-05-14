"""
Experiment 05 — CNN speaker identification with logging.

Treats each utterance as a (MAX_FRAMES x 40) MFCC+delta spectrogram and passes
it through a small 2-D convolutional network built with Keras/TensorFlow.

Requirements: pip install tensorflow
"""
import sys
import os
import numpy as np
import time
from scipy.io.wavfile import read
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "code"))

from speakerfeatures import extract_features
from logging_utils import ExperimentLogger
from metrics import compute_open_set_metrics

MAX_FRAMES = 300
N_FEATURES = 40
EPOCHS = 30
BATCH_SIZE = 8


def load_dataset(root, le=None, fit_encoder=False, logger=None, encode_labels=True):
    """Load dataset with optional logging."""
    X, y = [], []
    file_count = 0

    for speaker in os.listdir(root):
        path = os.path.join(root, speaker)
        if not os.path.isdir(path):
            continue

        speaker_files = []
        for fname in os.listdir(path):
            if not fname.endswith(".wav"):
                continue
            try:
                sr, audio = read(os.path.join(path, fname))
                feats = extract_features(audio, sr)

                if feats.shape[0] < MAX_FRAMES:
                    feats = np.vstack((feats, np.zeros((MAX_FRAMES - feats.shape[0], N_FEATURES))))
                else:
                    feats = feats[:MAX_FRAMES]

                X.append(feats)
                y.append(speaker)
                speaker_files.append(fname)
                file_count += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load {fname}: {e}")

        if speaker_files and logger:
            logger.debug(f"Loaded {len(speaker_files)} files for {speaker}")

    X = np.array(X)[..., np.newaxis]

    if le is None or not encode_labels:
        return X, np.array(y), file_count

    if fit_encoder:
        le.fit(y)
    y_enc = le.transform(y)

    return X, np.array(y_enc), file_count


def build_model(n_classes):
    """Build CNN model."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    inp = keras.Input(shape=(MAX_FRAMES, N_FEATURES, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_cnn(root_dir, dataset_name, logger=None):
    """Train CNN model."""
    if logger is None:
        logger = ExperimentLogger(f"cnn_{dataset_name}")

    logger.info("=" * 60)
    logger.info("CNN MODEL TRAINING")
    logger.info("=" * 60)

    start_time = time.time()
    train_dir = os.path.join(root_dir, "data", dataset_name, "train")

    le = LabelEncoder()
    X_train, y_train, file_count = load_dataset(train_dir, le=le, fit_encoder=True, logger=logger)

    if len(X_train) == 0:
        logger.error("No training data loaded")
        return None

    n_classes = len(le.classes_)
    logger.info(f"Loaded {len(X_train)} samples from {file_count} files")
    logger.info(f"Classes: {n_classes}")
    logger.info(f"Input shape: {X_train.shape}")

    logger.info("Building CNN model...")
    model = build_model(n_classes)

    logger.info(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(X_train, y_train,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=0.1,
                       verbose=0)

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Classes:               {n_classes}")
    logger.info(f"Training samples:      {len(X_train)}")
    logger.info(f"Input shape:           {X_train.shape}")
    logger.info(f"Epochs:                {EPOCHS}")
    logger.info(f"Batch size:            {BATCH_SIZE}")
    logger.info(f"Final train loss:      {history.history['loss'][-1]:.4f}")
    logger.info(f"Final train accuracy:  {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Training time:         {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "classes": n_classes,
        "training_samples": len(X_train),
        "epochs": EPOCHS,
        "training_time_seconds": elapsed,
    }

    logger.save_metrics("training_complete", metrics)

    return model, le, X_train.shape, logger


def evaluate_cnn(model, le, X_shape, root_dir, dataset_name, logger):
    """Evaluate CNN model."""
    logger.info("=" * 60)
    logger.info("CNN MODEL EVALUATION")
    logger.info("=" * 60)

    start_time = time.time()
    test_dir = os.path.join(root_dir, "data", dataset_name, "test")

    X_test, y_test, file_count = load_dataset(
        test_dir,
        le=le,
        fit_encoder=False,
        logger=logger,
        encode_labels=False,
    )

    if len(X_test) == 0:
        logger.error("No test data loaded")
        return None

    logger.info(f"Loaded {len(X_test)} test samples")

    logger.info("Making predictions...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_pred_labels = le.inverse_transform(y_pred)

    metrics = compute_open_set_metrics(y_test, y_pred_labels, le.classes_)

    elapsed = time.time() - start_time

    metrics["evaluation_time_seconds"] = elapsed

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed:       {metrics['total']}")
    logger.info(f"Known speaker samples: {metrics['known_total']}")
    logger.info(f"Unknown speaker samples: {metrics['unknown_total']}")
    logger.info(f"Correct predictions:   {metrics['correct']}")
    logger.info(f"Incorrect predictions: {metrics['incorrect']}")
    logger.info(f"False accepts:         {metrics['false_accepts']}")
    logger.info(f"False rejects:         {metrics['false_rejects']}")
    logger.info(f"Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"FNMR:                  {metrics['fnmr']:.4f}")
    logger.info(f"FMR:                   {metrics['fmr']:.4f}")
    logger.info(f"Evaluation time:       {elapsed:.2f}s")
    logger.info("=" * 60)

    logger.save_metrics("evaluation_complete", metrics)
    logger.save_results_summary(metrics)

    return metrics


if __name__ == "__main__":
    dataset_name = "1-to-1"
    root_dir = os.path.join(os.path.dirname(__file__), "..")

    logger = ExperimentLogger(f"cnn_{dataset_name}")
    logger.save_config({
        "dataset": dataset_name,
        "algorithm": "CNN",
        "max_frames": MAX_FRAMES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    })

    result = train_cnn(root_dir, dataset_name, logger=logger)
    if result:
        model, le, X_shape, logger = result
        evaluate_cnn(model, le, X_shape, root_dir, dataset_name, logger)
        logger.info("CNN EXPERIMENT COMPLETE")
