"""
Experiment 04 — k-Nearest Neighbours speaker identification with logging.

Represents each utterance as the mean of its MFCC+delta frames and classifies
using Euclidean-distance KNN.
"""
import sys
import os
import numpy as np
import time
from scipy.io.wavfile import read
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "code"))

from speakerfeatures import extract_features
from logging_utils import ExperimentLogger
from metrics import compute_open_set_metrics

K = 5


def load_dataset_with_logging(root, logger):
    """Load dataset and log progress."""
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
                X.append(feats.mean(axis=0))
                y.append(speaker)
                speaker_files.append(fname)
                file_count += 1
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

        if speaker_files:
            logger.debug(f"Loaded {len(speaker_files)} files for {speaker}")

    return np.array(X), np.array(y), file_count


def train_knn(root_dir, dataset_name, k=K, logger=None):
    """Train KNN model."""
    if logger is None:
        logger = ExperimentLogger(f"knn_{dataset_name}")

    logger.info("=" * 60)
    logger.info("K-NEAREST NEIGHBORS MODEL TRAINING")
    logger.info("=" * 60)

    start_time = time.time()
    train_dir = os.path.join(root_dir, "data", dataset_name, "train")

    X_train, y_train, file_count = load_dataset_with_logging(train_dir, logger)

    if len(X_train) == 0:
        logger.error("No training data loaded")
        return None

    logger.info(f"Loaded {len(X_train)} samples from {file_count} files")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    unique_speakers = len(np.unique(y_train_enc))

    logger.info(f"Training KNN with k={k}...")
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    clf.fit(X_train, y_train_enc)

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Speakers:              {unique_speakers}")
    logger.info(f"Training samples:      {len(X_train)}")
    logger.info(f"Feature dimension:     {X_train.shape[1]}")
    logger.info(f"K value:               {k}")
    logger.info(f"Training time:         {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "speakers": unique_speakers,
        "training_samples": len(X_train),
        "feature_dimension": X_train.shape[1],
        "training_time_seconds": elapsed,
    }

    logger.save_metrics("training_complete", metrics)

    return clf, le, logger


def evaluate_knn(clf, le, root_dir, dataset_name, logger):
    """Evaluate KNN model."""
    logger.info("=" * 60)
    logger.info("K-NEAREST NEIGHBORS MODEL EVALUATION")
    logger.info("=" * 60)

    start_time = time.time()
    test_dir = os.path.join(root_dir, "data", dataset_name, "test")

    X_test, y_test, file_count = load_dataset_with_logging(test_dir, logger)

    if len(X_test) == 0:
        logger.error("No test data loaded")
        return None

    logger.info(f"Loaded {len(X_test)} test samples")

    y_pred = clf.predict(X_test)
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

    logger = ExperimentLogger(f"knn_{dataset_name}")
    logger.save_config({
        "dataset": dataset_name,
        "algorithm": "KNN",
        "k": K,
        "metric": "euclidean"
    })

    clf, le, logger = train_knn(root_dir, dataset_name, k=K, logger=logger)
    if clf:
        evaluate_knn(clf, le, root_dir, dataset_name, logger)
        logger.info("KNN EXPERIMENT COMPLETE")
