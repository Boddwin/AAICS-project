import os
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read

from speakerfeatures import extract_features


SUPPORTED_EXTENSIONS = {".wav"}


def iter_segment_features(
    root_dir,
    segment_seconds=5.0,
    min_segment_seconds=1.0,
    logger=None,
):
    """
    Yield mean-pooled MFCC features for fixed-length audio segments.

    Unsupported files are skipped and returned through the final skipped list.
    """
    root_path = Path(root_dir)
    skipped = []
    segment_rows = []
    feature_rows = []

    audio_files = sorted(path for path in root_path.rglob("*") if path.is_file())
    for audio_path in audio_files:
        ext = audio_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(
                {
                    "file_path": str(audio_path),
                    "reason": f"unsupported audio format: {ext}",
                }
            )
            if logger:
                logger.warning(f"Skipping unsupported audio file: {audio_path}")
            continue

        try:
            sample_rate, audio = read(str(audio_path))
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
        except Exception as e:
            skipped.append({"file_path": str(audio_path), "reason": str(e)})
            if logger:
                logger.warning(f"Failed to read {audio_path}: {e}")
            continue

        segment_samples = int(sample_rate * segment_seconds)
        min_samples = int(sample_rate * min_segment_seconds)
        if segment_samples <= 0:
            raise ValueError("segment_seconds must be positive")

        segment_index = 0
        for start in range(0, len(audio), segment_samples):
            segment = audio[start:start + segment_samples]
            if len(segment) < min_samples:
                continue

            try:
                features = extract_features(segment, sample_rate)
                feature_rows.append(features.mean(axis=0))
                segment_rows.append(
                    {
                        "source_file": str(audio_path),
                        "source_group": audio_path.parent.name,
                        "segment_index": segment_index,
                        "start_seconds": start / sample_rate,
                        "duration_seconds": len(segment) / sample_rate,
                    }
                )
                segment_index += 1
            except Exception as e:
                skipped.append(
                    {
                        "file_path": str(audio_path),
                        "segment_index": segment_index,
                        "reason": str(e),
                    }
                )
                if logger:
                    logger.warning(f"Failed to extract segment from {audio_path}: {e}")

    if feature_rows:
        X = np.vstack(feature_rows)
    else:
        X = np.empty((0, 0))

    return X, segment_rows, skipped
