"""
Experiment 06 — Unsupervised clustering (K-Means) for speaker grouping with logging.

Unlike the other experiments this is unsupervised: no labels are used during
fitting. Performance is measured with Adjusted Rand Index (ARI) and
Normalised Mutual Information (NMI), which compare the discovered clusters to
the known speaker labels.
"""
import sys
import os
import csv
import numpy as np
import time
from scipy.io.wavfile import read
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "code"))

from speakerfeatures import extract_features
from logging_utils import ExperimentLogger
from audio_segments import iter_segment_features


def save_cluster_assignments(assignments, output_file):
    """Save segment clustering assignments to CSV."""
    if not assignments:
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fieldnames = list(assignments[0].keys())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)


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


def cluster_with_logging(root_dir, dataset_name, logger=None):
    """Perform clustering on dataset."""
    if logger is None:
        logger = ExperimentLogger(f"clustering_{dataset_name}")

    logger.info("=" * 60)
    logger.info("CLUSTERING (K-MEANS) MODEL")
    logger.info("=" * 60)

    start_time = time.time()

    # Load data
    data_split = "test" if dataset_name in ("1-to-1", "1-to-n") else "train"
    data_dir = os.path.join(root_dir, "data", dataset_name, data_split)
    if not os.path.isdir(data_dir):
        return cluster_audio_segments_with_logging(root_dir, dataset_name, logger=logger)

    X, y_true, file_count = load_dataset_with_logging(data_dir, logger)

    if len(X) == 0:
        logger.error("No data loaded")
        return None

    logger.info(f"Loaded {len(X)} samples from {file_count} files from {data_split}")

    # Encode true labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    n_clusters = len(le.classes_)

    logger.info(f"Ground truth speakers: {n_clusters}")
    logger.info(f"Feature dimension: {X.shape[1]}")

    # Perform clustering
    logger.info(f"Running K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    elapsed = time.time() - start_time

    # Compute metrics
    ari = adjusted_rand_score(y_enc, y_pred)
    nmi = normalized_mutual_info_score(y_enc, y_pred)

    try:
        silhouette = silhouette_score(X, y_pred)
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")
        silhouette = 0.0

    logger.info("=" * 60)
    logger.info("CLUSTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clusters:                {n_clusters}")
    logger.info(f"Samples:                 {len(X)}")
    logger.info(f"Feature dimension:       {X.shape[1]}")
    logger.info(f"Adjusted Rand Index:     {ari:.4f}  (1.0=perfect, 0.0=random)")
    logger.info(f"Normalized Mutual Info:  {nmi:.4f}  (1.0=perfect, 0.0=random)")
    logger.info(f"Silhouette Score:        {silhouette:.4f} (-1 to 1, higher is better)")
    logger.info(f"Clustering time:         {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "n_clusters": n_clusters,
        "n_samples": len(X),
        "feature_dimension": X.shape[1],
        "ari": ari,
        "nmi": nmi,
        "silhouette_score": silhouette,
        "clustering_time_seconds": elapsed,
        "data_split": data_split,
    }

    logger.save_metrics("clustering_complete", metrics)
    logger.save_results_summary(metrics)

    return metrics


def cluster_audio_segments_with_logging(root_dir, dataset_name, logger=None, n_clusters=3):
    """Cluster raw/combined audio datasets that do not have train/test folders."""
    if logger is None:
        logger = ExperimentLogger(f"clustering_segments_{dataset_name}")

    logger.info("=" * 60)
    logger.info("SEGMENT-LEVEL CLUSTERING (K-MEANS)")
    logger.info("=" * 60)

    start_time = time.time()
    data_dir = os.path.join(root_dir, "data", dataset_name)

    X, segment_rows, skipped = iter_segment_features(data_dir, logger=logger)
    if len(X) == 0:
        logger.error("No segment features loaded")
        if skipped:
            logger.error(f"Skipped files/segments: {skipped}")
        return None

    actual_clusters = min(n_clusters, len(X))
    if actual_clusters < 1:
        logger.error("No clusters can be formed")
        return None

    logger.info(f"Loaded {len(X)} segments")
    logger.info(f"Feature dimension: {X.shape[1]}")
    logger.info(f"Running K-Means with k={actual_clusters}...")

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    try:
        silhouette = silhouette_score(X, y_pred) if actual_clusters > 1 else 0.0
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")
        silhouette = 0.0

    elapsed = time.time() - start_time
    assignments = []
    for row, cluster_id in zip(segment_rows, y_pred):
        assignment = dict(row)
        assignment["cluster"] = int(cluster_id)
        assignments.append(assignment)

    save_cluster_assignments(
        assignments,
        os.path.join(logger.get_experiment_dir(), "segment_clusters.csv"),
    )

    metrics = {
        "n_clusters": actual_clusters,
        "n_samples": len(X),
        "feature_dimension": X.shape[1],
        "ari": None,
        "nmi": None,
        "silhouette_score": silhouette,
        "clustering_time_seconds": elapsed,
        "segment_seconds": 5.0,
        "source_files": len({row["source_file"] for row in segment_rows}),
        "skipped_files": len(skipped),
        "evaluation_note": "segment-level clustering without ground-truth labels",
    }

    logger.info("=" * 60)
    logger.info("SEGMENT CLUSTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clusters:                {actual_clusters}")
    logger.info(f"Segments:                {len(X)}")
    logger.info(f"Source files:            {metrics['source_files']}")
    logger.info(f"Skipped files/segments:  {len(skipped)}")
    logger.info(f"Silhouette Score:        {silhouette:.4f}")
    logger.info(f"Clustering time:         {elapsed:.2f}s")
    logger.info("=" * 60)

    logger.save_metrics("segment_clustering_complete", metrics)
    logger.save_results_summary(metrics)
    return metrics


def cluster_test_set(root_dir, dataset_name, n_clusters, logger=None):
    """Cluster test set using K-means with k=n_clusters."""
    if logger is None:
        logger = ExperimentLogger(f"clustering_test_{dataset_name}")

    logger.info("=" * 60)
    logger.info("CLUSTERING TEST SET EVALUATION")
    logger.info("=" * 60)

    start_time = time.time()

    # Load test data
    test_dir = os.path.join(root_dir, "data", dataset_name, "test")
    X_test, y_test, file_count = load_dataset_with_logging(test_dir, logger)

    if len(X_test) == 0:
        logger.error("No test data loaded")
        return None

    logger.info(f"Loaded {len(X_test)} test samples")

    # Encode true labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_test)

    # Perform clustering
    logger.info(f"Running K-Means clustering on test set with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_test)

    elapsed = time.time() - start_time

    # Compute metrics
    ari = adjusted_rand_score(y_enc, y_pred)
    nmi = normalized_mutual_info_score(y_enc, y_pred)

    try:
        silhouette = silhouette_score(X_test, y_pred)
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")
        silhouette = 0.0

    logger.info("=" * 60)
    logger.info("TEST SET CLUSTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clusters:                {n_clusters}")
    logger.info(f"Test samples:            {len(X_test)}")
    logger.info(f"Adjusted Rand Index:     {ari:.4f}")
    logger.info(f"Normalized Mutual Info:  {nmi:.4f}")
    logger.info(f"Silhouette Score:        {silhouette:.4f}")
    logger.info(f"Evaluation time:         {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "n_clusters": n_clusters,
        "n_test_samples": len(X_test),
        "ari": ari,
        "nmi": nmi,
        "silhouette_score": silhouette,
        "evaluation_time_seconds": elapsed,
    }

    logger.save_metrics("test_clustering_complete", metrics)
    logger.save_results_summary(metrics)

    return metrics


if __name__ == "__main__":
    dataset_name = "1-to-1"
    root_dir = os.path.join(os.path.dirname(__file__), "..")

    logger = ExperimentLogger(f"clustering_{dataset_name}")
    logger.save_config({
        "dataset": dataset_name,
        "algorithm": "KMeans",
        "approach": "unsupervised"
    })

    # Cluster training set
    metrics = cluster_with_logging(root_dir, dataset_name, logger=logger)

    # Cluster test set if available
    if metrics and dataset_name != "1-to-1":  # Simplified test set handling
        cluster_test_set(root_dir, dataset_name, metrics["n_clusters"], logger=logger)

    logger.info("CLUSTERING EXPERIMENT COMPLETE")
