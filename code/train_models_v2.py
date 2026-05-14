"""
Enhanced training script for GMM-UBM baseline with comprehensive logging.
"""
import os
import sys
import pickle
import numpy as np
import time
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")

# Add code directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from speakerfeatures import extract_features
from logging_utils import ExperimentLogger


def remove_previous_models(models_folder, ubm_folder, logger):
    """Delete previous models."""
    try:
        files = os.listdir(models_folder)
        for file in files:
            file_path = os.path.join(models_folder, file)
            os.remove(file_path)
        logger.info(f"Cleared previous models from {models_folder}")

        ubm_path = os.path.join(ubm_folder, "ubm.gmm")
        if os.path.isfile(ubm_path):
            os.remove(ubm_path)
            logger.info(f"Cleared previous UBM from {ubm_folder}")
    except Exception as e:
        logger.error(f"Error clearing previous models: {e}")


def create_gmm_models(audio_files, ubm_folder, models_folder, n_components=16, logger=None):
    """
    Create GMM models for each speaker and UBM from training data.

    Args:
        audio_files: str, path to root folder containing speaker subfolders
        ubm_folder: str, path to folder where UBM will be saved
        models_folder: str, path to folder where speaker models will be saved
        n_components: int, number of Gaussian components
        logger: ExperimentLogger instance
    """
    if logger is None:
        logger = ExperimentLogger("gmm_training")

    logger.info("=" * 60)
    logger.info("GMM-UBM MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Audio files path: {audio_files}")
    logger.info(f"N components: {n_components}")

    start_time = time.time()
    dir_list = os.listdir(audio_files)
    ubm_train = np.asarray(())
    speaker_count = 0
    file_count = 0
    feature_count = 0

    # Step 1: Process each speaker folder
    for folder in dir_list:
        folder_path = os.path.join(audio_files, folder)

        if os.path.isdir(folder_path):
            logger.debug(f"Processing speaker: {folder}")
            file_list = os.listdir(folder_path)
            features = np.asarray(())  # Empty feature set
            audio_files_for_speaker = 0

            # Step 2: Extract features from each audio file for this speaker
            for file in file_list:
                if not file.endswith(".wav"):
                    continue

                file_path = os.path.join(folder_path, file)
                try:
                    sr, audio = read(file_path)
                    vector = extract_features(audio, sr)

                    if features.size == 0:
                        features = vector
                    else:
                        features = np.vstack((features, vector))

                    if ubm_train.size == 0:
                        ubm_train = vector
                    else:
                        ubm_train = np.vstack((ubm_train, vector))

                    audio_files_for_speaker += 1
                    file_count += 1
                    feature_count += features.shape[0] if len(features.shape) > 1 else 1

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue

            if features.size == 0:
                logger.warning(f"No valid features extracted for speaker {folder}")
                continue

            # Step 3: Train GMM for this speaker
            try:
                gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
                gmm.fit(features)

                picklefile = folder + ".gmm"
                pickle.dump(gmm, open(os.path.join(models_folder, picklefile), "wb"))

                logger.info(
                    f"[OK] Model trained: {folder} | "
                    f"Features: {features.shape} | "
                    f"Audio files: {audio_files_for_speaker}"
                )
                speaker_count += 1

            except Exception as e:
                logger.error(f"Failed to train model for {folder}: {e}")
                continue
        else:
            if folder.endswith(".wav"):
                logger.debug(f"Processing single file: {folder}")
                try:
                    sr, audio = read(folder_path)
                    vector = extract_features(audio, sr)

                    if ubm_train.size == 0:
                        ubm_train = vector
                    else:
                        ubm_train = np.vstack((ubm_train, vector))

                    features = vector
                    gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
                    gmm.fit(features.reshape(-1, 1) if len(features.shape) == 1 else features)

                    picklefile = folder + ".gmm"
                    pickle.dump(gmm, open(os.path.join(models_folder, picklefile), "wb"))

                    logger.info(f"✓ Model trained: {folder}")
                    speaker_count += 1
                    file_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process {folder}: {e}")

    # Step 4: Train UBM on all features
    if ubm_train.size == 0:
        logger.error("No training data available for UBM!")
        return None

    try:
        ubm = GaussianMixture(n_components=n_components, covariance_type="diag")
        ubm.fit(ubm_train)
        pickle.dump(ubm, open(os.path.join(ubm_folder, "ubm.gmm"), "wb"))
        logger.info(f"[OK] UBM trained | Features: {ubm_train.shape}")
    except Exception as e:
        logger.error(f"Failed to train UBM: {e}")
        return None

    elapsed = time.time() - start_time

    # Log summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Speakers trained:    {speaker_count}")
    logger.info(f"Audio files:         {file_count}")
    logger.info(f"Total features:      {feature_count}")
    logger.info(f"Training time:       {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "speakers_trained": speaker_count,
        "audio_files": file_count,
        "total_features": feature_count,
        "training_time_seconds": elapsed,
    }

    logger.save_metrics("training_complete", metrics)
    logger.save_results_summary(metrics)

    return logger


if __name__ == "__main__":
    # Configuration
    dataset = "1-to-1"
    audio_files_folder = os.path.join("..", "data", dataset, "train")
    models_folder = "speaker_models"
    ubm_folder = "ubm_model"

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(ubm_folder, exist_ok=True)

    # Initialize logger
    logger = ExperimentLogger(f"gmm_ubm_train_{dataset}")
    logger.save_config(
        {
            "dataset": dataset,
            "algorithm": "GMM-UBM",
            "n_components": 16,
            "covariance_type": "diag",
        }
    )

    # Run training
    remove_previous_models(models_folder, ubm_folder, logger)
    create_gmm_models(audio_files_folder, ubm_folder, models_folder, logger=logger)

    logger.info("TRAINING COMPLETE")
