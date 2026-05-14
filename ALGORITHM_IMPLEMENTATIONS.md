# Algorithm Implementations - Summary

## Overview

All 5 alternative algorithms have been implemented with full logging infrastructure compatible with the experiment runner. Each algorithm:
- Uses `ExperimentLogger` for structured logging and metrics collection
- Exports results to JSON + CSV
- Follows same train/test pattern as baseline GMM-UBM
- Generates per-experiment timestamped result directories

## Algorithms Implemented

### 1. SVM (Support Vector Machine) - `02_svm.py`
**Approach**: One-vs-rest binary classifiers for each speaker

**Key Features**:
- Mean-pooled MFCC features (40-dim)
- RBF kernel with C=10
- Feature standardization (StandardScaler)
- Probability-based matching with threshold

**Usage**:
```python
from experiments.svm_v2 import train_svm, evaluate_svm

clf, scaler, le, logger = train_svm(root_dir, "1-to-1", logger)
metrics = evaluate_svm(clf, scaler, le, root_dir, "1-to-1", logger)
```

**Metrics**:
- Accuracy, FNMR, FMR
- Training/evaluation time

---

### 2. Random Forest - `03_random_forest.py`
**Approach**: Ensemble decision trees for speaker classification

**Key Features**:
- 200 decision trees
- Mean-pooled MFCC features
- Out-of-bag error estimation
- Feature importance available

**Usage**:
```python
from experiments.rf_v2 import train_rf, evaluate_rf

clf, le, logger = train_rf(root_dir, "1-to-1", logger)
metrics = evaluate_rf(clf, le, root_dir, "1-to-1", logger)
```

**Metrics**:
- Accuracy, FNMR, FMR
- Training time (typically fast with n_jobs=-1)

---

### 3. KNN (K-Nearest Neighbors) - `04_knn.py`
**Approach**: Distance-based similarity to training samples

**Key Features**:
- K=5 neighbors
- Euclidean distance metric
- No training phase (lazy learner)
- Fast inference

**Usage**:
```python
from experiments.knn_v2 import train_knn, evaluate_knn

clf, le, logger = train_knn(root_dir, "1-to-1", k=5, logger)
metrics = evaluate_knn(clf, le, root_dir, "1-to-1", logger)
```

**Metrics**:
- Accuracy, FNMR, FMR
- Evaluation time (longer for large datasets)

**Note**: Training time is minimal since KNN stores training data

---

### 4. CNN (Convolutional Neural Network) - `05_cnn.py`
**Approach**: Deep learning on spectrogram-like features

**Key Features**:
- 2D convolutions: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
- Fully connected layers: Dense(128) → Dropout(0.3) → Dense(n_classes)
- Input: (300 frames × 40 features × 1 channel)
- Padding/truncation to fixed length
- 30 epochs, batch size 8

**Architecture**:
```
Input (300×40×1)
  ↓
Conv2D(32, 3×3) + ReLU + MaxPool(2×2)
  ↓
Conv2D(64, 3×3) + ReLU + MaxPool(2×2)
  ↓
Flatten
  ↓
Dense(128) + ReLU + Dropout(0.3)
  ↓
Dense(n_classes) + Softmax
```

**Usage**:
```python
from experiments.cnn_v2 import train_cnn, evaluate_cnn

model, le, X_shape, logger = train_cnn(root_dir, "1-to-1", logger)
metrics = evaluate_cnn(model, le, X_shape, root_dir, "1-to-1", logger)
```

**Requirements**:
```bash
pip install tensorflow
```

**Metrics**:
- Accuracy, FNMR, FMR
- Training loss/accuracy per epoch
- Training/evaluation time

---

### 5. Clustering (K-Means) - `06_clustering.py`
**Approach**: Unsupervised speaker grouping (no labels used during training)

**Key Features**:
- K-Means with k=number of true speakers
- Mean-pooled MFCC features
- No training phase needed for evaluation
- Clustering quality metrics

**Metrics** (Unsupervised):
- **ARI** (Adjusted Rand Index): -1 to 1 (1 = perfect clustering)
- **NMI** (Normalized Mutual Information): 0 to 1 (1 = perfect clustering)
- **Silhouette Score**: -1 to 1 (higher is better)

**Usage**:
```python
from experiments.clustering_v2 import cluster_with_logging

metrics = cluster_with_logging(root_dir, "1-to-1", logger)
```

**Note**: Best for n-to-n datasets where clustering quality can be assessed

---

## Common Patterns

All algorithms follow this pattern:

```python
# 1. Load and preprocess
data = load_dataset_with_logging(path, logger)

# 2. Train
model_data = train_algo(root_dir, dataset, logger)

# 3. Evaluate
metrics = evaluate_algo(model_data, root_dir, dataset, logger)

# 4. Results saved to:
# - results/[algorithm]_[dataset]_TIMESTAMP/experiment.log
# - results/[algorithm]_[dataset]_TIMESTAMP/summary.json
# - results/[algorithm]_[dataset]_TIMESTAMP/metrics.csv
# - results/[algorithm]_[dataset]_TIMESTAMP/config.json
```

## Running Individual Algorithms

### From experiments directory:
```bash
cd experiments

python 02_svm.py              # Run SVM on 1-to-1
python 03_random_forest.py    # Run Random Forest
python 04_knn.py              # Run KNN
python 05_cnn.py              # Run CNN (requires TensorFlow)
python 06_clustering.py       # Run Clustering
```

### From code directory (planned integration):
Will be integrated into master runner for systematic evaluation across all datasets.

## Hyperparameter Configuration

Edit the parameters at the top of each file:

**SVM** (`02_svm.py`):
- Kernel: 'rbf' (line ~50)
- C: 10 (regularization parameter)
- Gamma: 'scale'

**Random Forest** (`03_random_forest.py`):
- n_estimators: 200 trees
- random_state: 42 (reproducibility)

**KNN** (`04_knn.py`):
- K: 5 neighbors
- metric: 'euclidean'

**CNN** (`05_cnn.py`):
- MAX_FRAMES: 300 (temporal dimension)
- EPOCHS: 30
- BATCH_SIZE: 8
- Layers: 2 Conv + 1 Dense layers

**Clustering** (`06_clustering.py`):
- n_clusters: automatically set to true speaker count
- random_state: 42
- n_init: 10 (random initializations)

## Performance Comparison Matrix

| Algorithm | Training | Inference | Memory | Interpretability | Best For |
|-----------|----------|-----------|--------|------------------|----------|
| GMM-UBM   | Fast     | Slow      | Low    | High             | Baseline |
| SVM       | Medium   | Fast      | High   | Medium           | Verification |
| RF        | Fast     | Fast      | Medium | High             | Grouping |
| KNN       | None     | Slow      | High   | High             | Small datasets |
| CNN       | Slow     | Fast      | High   | Low              | Large datasets |
| Clustering| Fast     | N/A       | Low    | High             | Discovery |

## Next Steps

1. **Integrate into master runner**:
   - Modify `code/run_experiments.py` to run all algorithms
   - Aggregate results across all datasets and algorithms
   - Generate comparison visualizations

2. **Test on all 5 datasets**:
   - 1-to-1 (verification) ✓
   - 1-to-n (identification)
   - n-to-n (grouping)
   - large_combined (complex)
   - large_mixed (discovery)

3. **Analyze results**:
   - Compare accuracy across algorithms and datasets
   - Evaluate computational efficiency
   - Assess suitability for each task

4. **Generate reports**:
   - Performance comparison tables
   - ROC curves (if probability scores available)
   - Confusion matrices per algorithm
   - Critical evaluation and recommendations
