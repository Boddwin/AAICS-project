# Voice Biometrics ML Strategy Document

## 1. Algorithm Selection Strategy

### Algorithms to Evaluate (6 total)
1. **GMM-UBM** (Gaussian Mixture Model + Universal Background Model)
   - Baseline, already provided
   - Probabilistic approach, standard for speaker recognition
   
2. **Support Vector Machine (SVM)**
   - Binary/multi-class classification
   - Good for verification tasks (1-to-1)
   - Efficient with high-dimensional features
   
3. **Random Forest**
   - Ensemble method, robust to overfitting
   - Handles feature interactions well
   - Interpretable feature importance
   
4. **K-Nearest Neighbors (KNN)**
   - Simple baseline for comparison
   - Distance-based similarity matching
   - Lightweight, good for real-time matching
   
5. **Convolutional Neural Network (CNN)**
   - Can work with spectrograms/MFCC features
   - Deep learning approach for complex patterns
   - Evaluate if deep learning improves over classical methods
   
6. **Clustering** (K-means, DBSCAN variants)
   - For n-to-n grouping tasks
   - Unsupervised learning evaluation
   - Compare with supervised methods

**Rationale**: Mix of classical (GMM, SVM, RF) and modern (CNN) methods, plus clustering for unsupervised scenarios.

---

## 2. Data Splitting & Cross-Validation Strategy

### Dataset-Specific Approach

#### 1-to-1 (Verification)
- **Split**: Use provided train/test folders as-is
- **Validation**: 5-fold cross-validation on training data
- **Task**: Binary classification (same person vs. different)
- **Metric Focus**: Equal Error Rate (EER), accuracy, FNMR, FMR

#### 1-to-n (Identification)
- **Split**: Use provided train/test folders
- **Validation**: Train on 1 person, test on 17 candidates
- **Task**: Multi-class classification (identify 1 out of 17)
- **Metric Focus**: Accuracy, Top-N accuracy

#### n-to-n (Grouping)
- **Split**: Stratified split (80% train, 20% test from provided folders)
- **Validation**: 5-fold cross-validation
- **Task**: Speaker clustering and identification
- **Metric Focus**: Accuracy, F1-score, silhouette score

#### large_combined & large_mixed
- **Preliminary Analysis**: Determine number of speakers first
- **Adaptive Split**: Based on dataset size
- **Task**: Complex speaker diarization/matching
- **Metric Focus**: Accuracy, recall per speaker class

### General Cross-Validation Approach
- **5-fold stratified CV** for small datasets (1-to-1, 1-to-n)
- **Time-series aware splitting** for speaker-level separation (different speakers in train/test)
- **Hold-out test set** for final evaluation

---

## 3. Evaluation & Success Metrics

### Primary Metrics (All Tasks)
1. **Accuracy**: Overall correctness
2. **FNMR (False Non-Match Rate)**: % of genuine speakers rejected
3. **FMR (False Match Rate)**: % of impostor speakers accepted
4. **Equal Error Rate (EER)**: Threshold where FNMR = FMR

### Secondary Metrics (Task-Specific)
- **1-to-1**: FAR (False Accept Rate), FRR (False Reject Rate)
- **1-to-n**: Top-1 and Top-5 accuracy
- **n-to-n**: F1-score, Silhouette score, Purity
- **large_* tasks**: Recall per speaker, Speaker-wise accuracy

### Performance Metrics (Non-Accuracy)
- **Training Time**: Wall-clock time to train models
- **Inference Time**: Per-audio matching speed
- **Memory Usage**: Model size and memory footprint
- **Scalability**: Performance degradation with dataset size

### Success Criteria
- **Verification (1-to-1)**: Achieve >95% accuracy, EER <5%
- **Identification (1-to-n)**: Achieve >90% Top-1 accuracy
- **Grouping (n-to-n)**: Achieve >85% accuracy with reasonable cluster purity
- **Practicality**: Training time <5 min for small datasets, <1 min inference per audio

---

## 4. Experimental Framework

### Testing Pipeline
```
For each dataset:
  For each algorithm:
    1. Load & preprocess data
    2. Train model with cross-validation
    3. Evaluate on test set
    4. Record: accuracy, FNMR, FMR, time, memory
    5. Generate ROC curves (for applicable algorithms)
    6. Log results to CSV
```

### Hyperparameter Tuning
- **Grid Search** on training set via cross-validation
- **Search Space** (per algorithm):
  - GMM: n_components in {4, 8, 16, 32}
  - SVM: C in {0.1, 1, 10}, kernel in {'rbf', 'linear'}
  - RF: n_estimators in {50, 100, 200}, max_depth in {10, 20, None}
  - KNN: k in {3, 5, 7, 11}
  - CNN: layers, filters, dropout rates
  - Clustering: n_clusters (adaptive), eps for DBSCAN

### Threshold Optimization
- For verification/identification tasks, sweep decision thresholds
- Report metrics at optimal threshold (maximizing F1 or minimizing EER)

---

## 5. Execution Plan

### Phase 1: Setup & Baseline (Week 1)
- [ ] Fix test_speaker.py bugs
- [ ] Implement comprehensive logging framework
- [ ] Verify feature extraction pipeline (MFCC 40-dim)
- [ ] Run GMM-UBM on all 5 datasets as baseline
- [ ] Document results

### Phase 2: Algorithm Implementation (Week 2)
- [ ] Implement/complete SVM experiment
- [ ] Implement/complete Random Forest
- [ ] Implement/complete KNN
- [ ] Implement/complete CNN (may use pre-trained embeddings)
- [ ] Implement/complete Clustering approaches
- [ ] Ensure consistent feature extraction across all

### Phase 3: Systematic Evaluation (Week 3)
- [ ] Run all algorithms on 1-to-1 dataset
- [ ] Run all algorithms on 1-to-n dataset
- [ ] Run all algorithms on n-to-n dataset
- [ ] Analyze large_combined and large_mixed
- [ ] Aggregate metrics to CSV

### Phase 4: Analysis & Reporting (Week 4)
- [ ] Compare performance across algorithms
- [ ] Critical evaluation: suitability for each task
- [ ] Identify best algorithm per dataset/task
- [ ] Visualize results (ROC, confusion matrices, bar charts)
- [ ] Write conclusions

---

## 6. Output & Reporting

### Deliverables
1. **results/metrics_summary.csv**: All metrics across all configs
2. **results/[dataset]_[algorithm]_results.txt**: Detailed per-run logs
3. **figures/**: ROC curves, confusion matrices, performance comparison
4. **CRITICAL_EVALUATION.md**: Final suitability assessment

### Report Structure
- Abstract: Summary of findings
- Methodology: Algorithms, datasets, evaluation approach
- Results: Metrics per task, visual comparisons
- Critical Evaluation:
  - What works well and why?
  - What fails and why?
  - Practical feasibility for investigative use
  - Recommendations for each use case
- Limitations & Future Work

---

## Key Decision Points & Open Questions

1. **Feature Engineering**: Should we explore augmenting MFCC (e.g., i-vector, x-vector)?
2. **Dataset Imbalance**: How to handle if large_* datasets have class imbalance?
3. **Threshold Strategy**: Optimize globally or per-algorithm?
4. **Computational Constraints**: CPU-only or GPU acceleration available?
5. **Ensemble Methods**: Should we try voting/stacking multiple algorithms?

*Next Step: Begin Phase 1 — fix baseline and establish logging framework.*
