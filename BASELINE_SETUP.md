# Speaker Identification Baseline - Setup & Usage Guide

## Overview

The baseline code has been refactored with:
- **Comprehensive logging** for all experiments
- **Structured metrics collection** saved to CSV
- **Organized results** with per-experiment directories
- **Master experiment runner** for systematic evaluation

## New Scripts

### 1. `train_models_v2.py` - Training with Logging

Enhanced version of the original `train_models.py` with:
- Proper error handling
- Detailed logging to file + console
- Metrics saved to CSV
- Configuration tracking

**Usage:**
```bash
cd code
python train_models_v2.py
```

**Configuration** (edit at bottom of script):
- `dataset`: Choose dataset name (e.g., "1-to-1")
- Models saved to: `speaker_models/`
- UBM saved to: `ubm_model/`
- Logs saved to: `results/gmm_ubm_train_1-to-1_TIMESTAMP/`

**Output files:**
- `experiment.log` - Detailed console output
- `config.json` - Configuration used
- `metrics.csv` - Metrics for each stage
- `summary.json` - Final summary

### 2. `test_speaker_v2.py` - Testing with Logging

Enhanced version of the original `test_speaker.py` with:
- Fixed file validation bugs
- Proper feature extraction error handling
- Detailed per-match logging
- Comprehensive metrics computation
- Returns structured results

**Usage:**
```bash
cd code
python test_speaker_v2.py
```

**Output:**
```
[OK] CORRECT: Action-potential -> Action-potential (LLR=   0.08, GMM= -23.26)

EVALUATION SUMMARY
============================================================
Total processed:       1
Correct matches:       1
Incorrect matches:     0
No match found:        0
Accuracy:              1.0000 (100.00%)
FNMR:                  0.0000
FMR:                   0.0000
Evaluation time:       1.69s
```

### 3. `run_experiments.py` - Master Experiment Runner

Systematically runs GMM-UBM on all datasets and aggregates results.

**Usage - All datasets:**
```bash
cd code
python run_experiments.py
```

**Usage - Single dataset:**
```bash
python run_experiments.py --dataset 1-to-1
python run_experiments.py --dataset 1-to-n
python run_experiments.py --dataset n-to-n
```

**Output:**
- `results/all_results.csv` - Aggregated metrics across all runs
- `results/gmm_ubm_[dataset]_TIMESTAMP/` - Per-dataset results

## Logging Infrastructure

### ExperimentLogger (in `utils/logging_utils.py`)

Centralized logging with automatic directory structure:

```python
from utils.logging_utils import ExperimentLogger

logger = ExperimentLogger("my_experiment_name")
logger.info("Starting experiment...")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred!")

# Save metrics at checkpoints
metrics = {"accuracy": 0.95, "fnmr": 0.05}
logger.save_metrics("stage_name", metrics)

# Save config
logger.save_config({"algorithm": "GMM-UBM", "n_components": 16})

# Get output directory
exp_dir = logger.get_experiment_dir()
```

**Output structure:**
```
results/
  gmm_ubm_train_1-to-1_20260510_164535/
    experiment.log       # Detailed logs
    config.json          # Configuration
    metrics.csv          # Timestamped metrics
    summary.json         # Final summary
```

### ResultsAggregator (in `utils/logging_utils.py`)

Aggregate results from multiple experiments:

```python
from utils.logging_utils import ResultsAggregator

agg = ResultsAggregator("results/all_experiments.csv")

# Add individual results
agg.add_result("GMM-UBM", "1-to-1", "verification", {
    "accuracy": 0.95,
    "fnmr": 0.05,
    "fmr": 0.02
})

# Save aggregated CSV
agg.save()
agg.print_summary()
```

## Metrics Collected

### Primary Metrics (computed for all tasks)
- **Accuracy**: Correct / Total
- **FNMR** (False Non-Match Rate): No-Match / Total
- **FMR** (False Match Rate): Incorrect / Total

### Performance Metrics
- **Training time** (seconds)
- **Evaluation time** (seconds)
- **Per-match scores**: LLR, GMM score, UBM score

## Fixed Issues

### Bug Fixes in Baseline Code

1. **File validation (test_speaker.py:36)**
   - **Before**: Checked `file_path.endswith('.wav')` before validating path exists
   - **After**: Checks `file.endswith('.wav')` before constructing path

2. **Feature extraction error handling**
   - Added try-catch blocks for audio file reading
   - Graceful handling of corrupted files

3. **Score computation fix**
   - Fixed using undefined `vector` variable
   - Now properly accumulates and scores all features

4. **Metrics calculation**
   - Fixed division by zero when no matches found
   - Proper metrics returned as dictionary

## Directory Structure

```
project/
  code/
    train_models_v2.py       ← Enhanced training (NEW)
    test_speaker_v2.py        ← Enhanced testing (NEW)
    run_experiments.py        ← Master runner (NEW)
    train_models.py           ← Original (deprecated)
    test_speaker.py           ← Original (deprecated)
    speakerfeatures.py        ← Feature extraction (unchanged)
  utils/
    logging_utils.py          ← Logging infrastructure (NEW)
    metrics.py                ← Metrics functions (existing)
    features.py               ← Feature utils
  results/
    all_results.csv           ← Aggregated results
    gmm_ubm_train_1-to-1_*/   ← Individual run results
    gmm_ubm_test_1-to-1_*/
  data/
    1-to-1/train/
    1-to-1/test/
    1-to-n/train/
    1-to-n/test/
    n-to-n/train/
    n-to-n/test/
    large_combined/
    large_mixed/
```

## Workflow Example

### Single dataset (1-to-1)
```bash
cd code

# Train GMM-UBM model
python train_models_v2.py

# Test on 1-to-1 dataset
python test_speaker_v2.py

# Check results
cat ../results/gmm_ubm_test_1-to-1_*/summary.json
```

### All datasets
```bash
cd code

# Run everything with master runner
python run_experiments.py

# View aggregated results
cat ../results/all_results.csv
```

### Custom experiment
```python
# my_experiment.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from logging_utils import ExperimentLogger
from train_models_v2 import create_gmm_models

logger = ExperimentLogger("my_custom_experiment")
logger.save_config({"custom_param": "value"})

# Your code here
create_gmm_models("../data/1-to-1/train", "ubm_model", "speaker_models", logger=logger)

logger.info("Experiment complete")
```

## Next Steps

1. **Verify baseline works** on all 5 datasets
2. **Implement alternative algorithms** in `experiments/` directory
3. **Run all algorithms** with `run_experiments.py`
4. **Compare results** in `results/all_results.csv`
5. **Generate visualizations** (ROC curves, confusion matrices)
6. **Write critical evaluation** based on findings

## Troubleshooting

**Issue**: Unicode encoding errors in console output
- **Cause**: Windows console doesn't support some Unicode characters
- **Solution**: Already fixed (using [OK], [ERR] instead of ✓, ✗)

**Issue**: Models not found after training
- **Cause**: Running from wrong directory or paths misconfigured
- **Solution**: Ensure you're in `code/` directory; check relative paths

**Issue**: "No GMM files found"
- **Cause**: Training didn't complete successfully
- **Solution**: Check training logs in `results/` for errors

**Issue**: Features dimension mismatch
- **Cause**: MFCC extraction differs between modules
- **Solution**: Ensure all use `speakerfeatures.extract_features()` consistently
