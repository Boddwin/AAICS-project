"""
Enhanced testing script for GMM-UBM baseline with comprehensive logging.
"""
import os
import sys
import pickle
import csv
import numpy as np
import time
from scipy.io.wavfile import read
import warnings

warnings.filterwarnings("ignore")

# Add code directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from speakerfeatures import extract_features
from logging_utils import ExperimentLogger
from metrics import compute_open_set_metrics


def _load_models(models_folder, ubm_file, logger):
    """Load speaker GMMs and the UBM from disk."""
    gmm_files = [
        os.path.join(models_folder, fname)
        for fname in os.listdir(models_folder)
        if fname.endswith(".gmm")
    ]

    if not gmm_files:
        logger.error(f"No GMM files found in {models_folder}")
        return None, None, None

    with open(ubm_file, "rb") as f:
        ubm = pickle.load(f)

    models = []
    speakers = []
    for fname in sorted(gmm_files):
        with open(fname, "rb") as f:
            models.append(pickle.load(f))
        speakers.append(os.path.splitext(os.path.basename(fname))[0])

    logger.info(f"[OK] Loaded {len(models)} speaker models")
    logger.info("[OK] Loaded UBM model")
    return models, speakers, ubm


def _iter_wav_files(audio_files_folder):
    """Yield (true_speaker, sample_id, file_path) for direct or foldered WAV files."""
    for item in sorted(os.scandir(audio_files_folder), key=lambda entry: entry.name):
        if item.is_dir():
            true_speaker = item.name
            for fname in sorted(os.listdir(item.path)):
                if fname.lower().endswith(".wav"):
                    sample_id = os.path.splitext(fname)[0]
                    yield true_speaker, sample_id, os.path.join(item.path, fname)
        elif item.name.lower().endswith(".wav"):
            sample_id = os.path.splitext(item.name)[0]
            yield sample_id, sample_id, item.path


def _score_features(features, models, speakers, ubm):
    """Score one feature matrix against all speaker models."""
    gmm_scores = np.array([model.score(features) for model in models])
    ubm_score = float(ubm.score(features))
    llr_scores = gmm_scores - ubm_score
    best_idx = int(np.argmax(llr_scores))

    return {
        "matched_speaker": speakers[best_idx],
        "llr_score": float(llr_scores[best_idx]),
        "gmm_score": float(gmm_scores[best_idx]),
        "ubm_score": ubm_score,
    }


def score_gmm_ubm_utterances(models_folder, audio_files_folder, ubm_file, logger=None):
    """Score every WAV utterance in a labeled test folder."""
    if logger is None:
        logger = ExperimentLogger("gmm_ubm_scoring")

    try:
        models, speakers, ubm = _load_models(models_folder, ubm_file, logger)
        if models is None:
            return None, []
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None, []

    rows = []
    known_speakers = set(speakers)
    for true_speaker, sample_id, file_path in _iter_wav_files(audio_files_folder):
        try:
            sr, audio = read(file_path)
            features = extract_features(audio, sr)
            scores = _score_features(features, models, speakers, ubm)
            rows.append(
                {
                    "true_speaker": true_speaker,
                    "sample_id": sample_id,
                    "file_path": file_path,
                    "known_speaker": true_speaker in known_speakers,
                    **scores,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to score {file_path}: {e}")

    return rows, speakers


def _decision_from_row(row, threshold, threshold_metric, require_positive_llr=True):
    score = row[threshold_metric]
    accepted = score > threshold
    if require_positive_llr:
        accepted = accepted and row["llr_score"] > 0
    return row["matched_speaker"] if accepted else None


def compute_gmm_metrics_from_scores(
    score_rows,
    known_speakers,
    threshold,
    threshold_metric="gmm_score",
    require_positive_llr=True,
):
    """Compute open-set metrics from previously scored utterances."""
    y_true = [row["true_speaker"] for row in score_rows]
    y_pred = [
        _decision_from_row(row, threshold, threshold_metric, require_positive_llr)
        for row in score_rows
    ]

    metrics = compute_open_set_metrics(y_true, y_pred, known_speakers)
    metrics["threshold"] = threshold
    metrics["threshold_metric"] = threshold_metric
    metrics["require_positive_llr"] = require_positive_llr
    return metrics


def sweep_gmm_thresholds(
    score_rows,
    known_speakers,
    threshold_metric="gmm_score",
    require_positive_llr=True,
):
    """Sweep thresholds and select the point closest to EER."""
    if not score_rows:
        return [], None

    known_speakers = set(known_speakers)
    has_known = any(row["true_speaker"] in known_speakers for row in score_rows)
    has_unknown = any(row["true_speaker"] not in known_speakers for row in score_rows)
    if not (has_known and has_unknown):
        return [], None

    scores = sorted({float(row[threshold_metric]) for row in score_rows})
    thresholds = [scores[0] - 1e-6] + scores + [scores[-1] + 1e-6]
    sweep_rows = []

    for threshold in thresholds:
        metrics = compute_gmm_metrics_from_scores(
            score_rows,
            known_speakers,
            threshold,
            threshold_metric=threshold_metric,
            require_positive_llr=require_positive_llr,
        )
        eer_gap = abs(metrics["fmr"] - metrics["fnmr"])
        eer_estimate = (metrics["fmr"] + metrics["fnmr"]) / 2
        sweep_rows.append(
            {
                "threshold": threshold,
                "threshold_metric": threshold_metric,
                "accuracy": metrics["accuracy"],
                "fnmr": metrics["fnmr"],
                "fmr": metrics["fmr"],
                "eer_gap": eer_gap,
                "eer_estimate": eer_estimate,
                "known_total": metrics["known_total"],
                "unknown_total": metrics["unknown_total"],
                "false_accepts": metrics["false_accepts"],
                "false_rejects": metrics["false_rejects"],
            }
        )

    best = min(sweep_rows, key=lambda row: (row["eer_gap"], row["eer_estimate"]))
    return sweep_rows, best


def _save_csv(rows, output_file):
    """Save a list of dictionaries with unioned fieldnames."""
    if not rows:
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gmm_ubm_utterances(
    models_folder,
    audio_files_folder,
    ubm_file,
    threshold=None,
    threshold_metric="gmm_score",
    require_positive_llr=True,
    logger=None,
    output_dir=None,
):
    """Evaluate GMM-UBM using one row per WAV utterance and threshold sweep."""
    if logger is None:
        logger = ExperimentLogger("gmm_ubm_utterance_test")

    logger.info("=" * 60)
    logger.info("GMM-UBM UTTERANCE-LEVEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models folder: {models_folder}")
    logger.info(f"Test folder: {audio_files_folder}")
    logger.info(f"Threshold metric: {threshold_metric}")

    start_time = time.time()
    score_rows, speakers = score_gmm_ubm_utterances(
        models_folder, audio_files_folder, ubm_file, logger=logger
    )

    if not score_rows:
        logger.error("No utterance scores generated")
        return None

    sweep_rows, best_threshold = sweep_gmm_thresholds(
        score_rows,
        speakers,
        threshold_metric=threshold_metric,
        require_positive_llr=require_positive_llr,
    )

    selected_threshold = threshold
    threshold_selection = "provided"
    if selected_threshold is None and best_threshold is not None:
        selected_threshold = best_threshold["threshold"]
        threshold_selection = "eer_sweep"
    elif selected_threshold is None:
        selected_threshold = -25 if threshold_metric == "gmm_score" else 0
        threshold_selection = "fallback"

    metrics = compute_gmm_metrics_from_scores(
        score_rows,
        speakers,
        selected_threshold,
        threshold_metric=threshold_metric,
        require_positive_llr=require_positive_llr,
    )

    elapsed = time.time() - start_time
    metrics["evaluation_time_seconds"] = elapsed
    metrics["threshold_selection"] = threshold_selection
    if best_threshold is not None:
        metrics["eer"] = best_threshold["eer_estimate"]
        metrics["eer_threshold"] = best_threshold["threshold"]
    else:
        metrics["eer"] = None
        metrics["eer_threshold"] = None

    for row in score_rows:
        predicted = _decision_from_row(
            row,
            selected_threshold,
            threshold_metric,
            require_positive_llr=require_positive_llr,
        )
        row["decision"] = "ACCEPT" if predicted is not None else "REJECT"
        row["predicted_speaker"] = predicted or ""
        row["correct"] = predicted == row["true_speaker"]

    if output_dir:
        _save_csv(score_rows, os.path.join(output_dir, "predictions.csv"))
        _save_csv(sweep_rows, os.path.join(output_dir, "threshold_sweep.csv"))

    logger.info("=" * 60)
    logger.info("UTTERANCE EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed:       {metrics['total']}")
    logger.info(f"Known speaker samples: {metrics['known_total']}")
    logger.info(f"Unknown speaker samples: {metrics['unknown_total']}")
    logger.info(f"Correct:               {metrics['correct']}")
    logger.info(f"Incorrect:             {metrics['incorrect']}")
    logger.info(f"False accepts:         {metrics['false_accepts']}")
    logger.info(f"False rejects:         {metrics['false_rejects']}")
    logger.info(f"Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"FNMR:                  {metrics['fnmr']:.4f}")
    logger.info(f"FMR:                   {metrics['fmr']:.4f}")
    if metrics["eer"] is not None:
        logger.info(f"EER:                   {metrics['eer']:.4f}")
        logger.info(f"EER threshold:         {metrics['eer_threshold']:.4f}")
    logger.info(f"Selected threshold:    {selected_threshold:.4f} ({threshold_selection})")
    logger.info(f"Evaluation time:       {elapsed:.2f}s")
    logger.info("=" * 60)

    logger.save_metrics("evaluation_complete", metrics)
    logger.save_results_summary(metrics)
    return metrics


def evaluate_gmm_ubm(
    models_folder, audio_files_folder, ubm_file, threshold=-25, logger=None
):
    """
    Evaluate GMM-UBM models on test data with labeled folders.

    Args:
        models_folder: str, path to folder with trained .gmm models
        audio_files_folder: str, path to test data (subfolders = speaker names)
        ubm_file: str, path to UBM model file
        threshold: float, decision threshold for acceptance
        logger: ExperimentLogger instance

    Returns:
        dict with evaluation metrics
    """
    if logger is None:
        logger = ExperimentLogger("gmm_ubm_test")

    logger.info("=" * 60)
    logger.info("GMM-UBM MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models folder: {models_folder}")
    logger.info(f"Test folder: {audio_files_folder}")
    logger.info(f"Threshold: {threshold}")

    start_time = time.time()

    # Step 1: Load trained models
    try:
        gmm_files = [
            os.path.join(models_folder, fname)
            for fname in os.listdir(models_folder)
            if fname.endswith(".gmm")
        ]

        if not gmm_files:
            logger.error(f"No GMM files found in {models_folder}")
            return None

        ubm = pickle.load(open(ubm_file, "rb"))
        models = [pickle.load(open(fname, "rb")) for fname in gmm_files]
        speakers = [os.path.splitext(os.path.basename(fname))[0] for fname in gmm_files]

        logger.info(f"[OK] Loaded {len(models)} speaker models")
        logger.info(f"[OK] Loaded UBM model")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None

    # Step 2: Test each speaker in test folder
    correct = 0
    incorrect = 0
    notmatched = 0
    total_processed = 0
    test_results = []

    for it in os.scandir(audio_files_folder):
        features = np.asarray(())
        test_speaker_name = it.name
        audio_count = 0

        # Extract features from all audio files for this test speaker
        if it.is_dir():
            logger.debug(f"Processing speaker: {test_speaker_name}")
            file_list = os.listdir(it.path)

            for file in file_list:
                if not file.endswith(".wav"):
                    continue

                file_path = os.path.join(it.path, file)
                try:
                    sr, audio = read(file_path)
                    vector = extract_features(audio, sr)

                    if features.size == 0:
                        features = vector
                    else:
                        features = np.vstack((features, vector))
                    audio_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue

            if features.size == 0:
                logger.warning(f"No valid audio features for {test_speaker_name}")
                continue

        else:
            # Single file matching
            if not it.name.endswith(".wav"):
                continue

            test_speaker_name = os.path.splitext(it.name)[0]
            try:
                sr, audio = read(it.path)
                vector = extract_features(audio, sr)
                features = vector
                audio_count = 1
            except Exception as e:
                logger.warning(f"Failed to process {it.path}: {e}")
                continue

        # Step 3: Score against all models
        gmm_scores = np.zeros(len(models))
        ubm_scores = np.zeros(len(models))
        llr_scores = np.zeros(len(models))

        for i in range(len(models)):
            gmm_scores[i] = models[i].score(features)
            ubm_scores[i] = ubm.score(features)
            llr_scores[i] = gmm_scores[i] - ubm_scores[i]

        # Step 4: Make decision based on best score
        best_idx = np.argmax(llr_scores)
        best_speaker = speakers[best_idx]
        best_score = llr_scores[best_idx]
        best_gmm_score = gmm_scores[best_idx]

        total_processed += 1
        decision = "REJECT"

        if best_score > 0 and best_gmm_score > threshold:
            decision = "ACCEPT"

            if test_speaker_name == best_speaker:
                correct += 1
                result_status = "[OK] CORRECT"
            else:
                incorrect += 1
                result_status = "[ERR] INCORRECT"
        else:
            notmatched += 1
            result_status = "- NO MATCH"

        logger.info(
            f"{result_status}: {test_speaker_name:15} -> {best_speaker:15} "
            f"(LLR={best_score:7.2f}, GMM={best_gmm_score:7.2f})"
        )

        test_results.append(
            {
                "test_speaker": test_speaker_name,
                "matched_speaker": best_speaker,
                "decision": decision,
                "llr_score": best_score,
                "gmm_score": best_gmm_score,
                "correct": test_speaker_name == best_speaker,
            }
        )

    elapsed = time.time() - start_time

    # Step 5: Compute metrics
    total = correct + incorrect + notmatched
    accuracy = correct / total if total > 0 else 0.0
    fnmr = notmatched / total if total > 0 else 0.0
    fmr = incorrect / total if total > 0 else 0.0

    # Log summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed:       {total_processed}")
    logger.info(f"Correct matches:       {correct}")
    logger.info(f"Incorrect matches:     {incorrect}")
    logger.info(f"No match found:        {notmatched}")
    logger.info(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"FNMR:                  {fnmr:.4f}")
    logger.info(f"FMR:                   {fmr:.4f}")
    logger.info(f"Evaluation time:       {elapsed:.2f}s")
    logger.info("=" * 60)

    metrics = {
        "total": total_processed,
        "correct": correct,
        "incorrect": incorrect,
        "notmatched": notmatched,
        "accuracy": accuracy,
        "fnmr": fnmr,
        "fmr": fmr,
        "evaluation_time_seconds": elapsed,
        "threshold": threshold,
    }

    logger.save_metrics("evaluation_complete", metrics)
    logger.save_results_summary(metrics)

    return metrics


if __name__ == "__main__":
    # Configuration
    dataset = "1-to-1"
    audio_files_folder = os.path.join("..", "data", dataset, "test")
    models_folder = "speaker_models"
    ubm_file = os.path.join("ubm_model", "ubm.gmm")

    # Initialize logger
    logger = ExperimentLogger(f"gmm_ubm_test_{dataset}")
    logger.save_config(
        {
            "dataset": dataset,
            "algorithm": "GMM-UBM",
            "threshold": -25,
        }
    )

    # Run evaluation
    results = evaluate_gmm_ubm(
        models_folder, audio_files_folder, ubm_file, threshold=-25, logger=logger
    )

    if results:
        logger.info("EVALUATION COMPLETE")
    else:
        logger.error("EVALUATION FAILED")
