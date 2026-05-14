"""
Experiment 01 — GMM-UBM speaker identification.

Trains one 16-component diagonal GMM per speaker plus a Universal Background
Model from all training data, then evaluates on the test set using a
likelihood-ratio threshold.
"""
import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "utils"))

import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture

from features import extract_features
from metrics import compute_metrics, print_metrics, save_metrics_csv

TRAIN_DIR    = os.path.join(ROOT, "data", "1-to-1", "train")
TEST_DIR     = os.path.join(ROOT, "data", "1-to-1", "test")
MODELS_DIR   = os.path.join(ROOT, "speaker_models")
UBM_DIR      = os.path.join(ROOT, "ubm_model")
RESULTS_PATH = os.path.join(ROOT, "results", "01_gmm_ubm.csv")
THRESHOLD    = -25
N_COMPONENTS = 16


def _load_speaker_features(path):
    feats = np.asarray(())
    for fname in os.listdir(path):
        if not fname.endswith(".wav"):
            continue
        sr, audio = read(os.path.join(path, fname))
        vec = extract_features(audio, sr)
        feats = vec if feats.size == 0 else np.vstack((feats, vec))
    return feats


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(UBM_DIR, exist_ok=True)

    ubm_data = np.asarray(())
    for speaker in os.listdir(TRAIN_DIR):
        speaker_path = os.path.join(TRAIN_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue
        feats = _load_speaker_features(speaker_path)
        gmm = mixture.GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag")
        gmm.fit(feats)
        pickle.dump(gmm, open(os.path.join(MODELS_DIR, f"{speaker}.gmm"), "wb"))
        print(f"Trained GMM: {speaker}  shape={feats.shape}")
        ubm_data = feats if ubm_data.size == 0 else np.vstack((ubm_data, feats))

    ubm = mixture.GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag")
    ubm.fit(ubm_data)
    pickle.dump(ubm, open(os.path.join(UBM_DIR, "ubm.gmm"), "wb"))
    print(f"UBM trained on {ubm_data.shape[0]} frames from all speakers.")


def evaluate():
    gmm_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gmm")]
    models    = [pickle.load(open(os.path.join(MODELS_DIR, f), "rb")) for f in gmm_files]
    speakers  = [os.path.splitext(f)[0] for f in gmm_files]
    ubm       = pickle.load(open(os.path.join(UBM_DIR, "ubm.gmm"), "rb"))

    correct = incorrect = not_matched = 0

    for entry in os.scandir(TEST_DIR):
        if entry.is_dir():
            feats = _load_speaker_features(entry.path)
        elif entry.name.endswith(".wav"):
            sr, audio = read(entry.path)
            feats = extract_features(audio, sr)
        else:
            continue

        gmm_scores = np.array([m.score(feats) for m in models])
        ubm_score  = ubm.score(feats)
        sc         = gmm_scores - ubm_score
        best       = np.argmax(sc)

        if sc[best] > 0 and gmm_scores[best] > THRESHOLD:
            if entry.name == speakers[best]:
                correct += 1
                print(f"Correct:   {entry.name} -> {speakers[best]}  score={sc[best]:.2f}")
            else:
                incorrect += 1
                print(f"Incorrect: {entry.name} -> {speakers[best]}  score={sc[best]:.2f}")
        else:
            not_matched += 1
            print(f"No match:  {entry.name}")

    m = compute_metrics(correct, incorrect, not_matched)
    m["model"] = "GMM-UBM"
    print("\n--- Results ---")
    print_metrics(m)
    save_metrics_csv(m, RESULTS_PATH)


if __name__ == "__main__":
    train()
    evaluate()
