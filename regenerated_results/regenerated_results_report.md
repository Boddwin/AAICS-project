# Regenerated Results Report

This draft uses only files regenerated into `regenerated_results/`.

## Results Narrative

### GMM-UBM 1-to-n

The regenerated GMM-UBM 1-to-n run uses utterance-level scoring. It processed 694 utterances, including 10 enrolled-speaker samples and 684 unknown-speaker samples. The threshold sweep selected `-28.0404` using `gmm_score`, giving accuracy 0.7997, FNMR 0.2000, FMR 0.2003, and EER 0.2001.

### n-to-n Supervised Classification

The best regenerated n-to-n supervised classifier is `CNN` with accuracy 0.2790 over 810.0000 test utterances. All supervised n-to-n models remain well below the GMM-UBM historical baseline, supporting the discussion that mean-pooled MFCC vectors are a weak representation for large closed-set speaker identification.

### 1-to-n Open-set Classifiers

RF, KNN, and CNN now complete under open-set scoring, but they do not have a rejection threshold. Each predicts the sole enrolled speaker for unknown utterances, producing FMR 1.0000 and accuracy 0.0144. This is a useful failure finding: closed-set classifiers are unsuitable for 1-to-n surveillance-style screening unless calibrated with rejection logic.

### Large Raw-audio Datasets

`large_combined` was evaluated with segment-level clustering: 1293.0000 five-second segments, 3.0000 clusters, silhouette 0.0665, and 0.0000 skipped unsupported files/segments.
`large_mixed` was evaluated with segment-level clustering: 2004.0000 five-second segments, 3.0000 clusters, silhouette 0.0667, and 1.0000 skipped unsupported files/segments.

These large-dataset results are diagnostic rather than accuracy claims because no segment-level ground-truth labels are present.

### Explicit Exclusions

The regenerated table records 10 failed or excluded algorithm/dataset combinations. These are retained as rows in `all_algorithms.csv` so the report can distinguish unavailable tasks from missing data.

## GMM-UBM Results

| algorithm | dataset | task | status | accuracy | fnmr | fmr | eer | threshold | threshold_selection | total | known_total | unknown_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM-UBM | 1-to-n | identification | PASS | 0.7997 | 0.2000 | 0.2003 | 0.2001 | -28.0404 | eer_sweep | 694 | 10 | 684 |

## Supervised And Clustering Results

| algorithm | dataset | task | status | reason | accuracy | fnmr | fmr | ari | nmi | silhouette_score | total | known_total | unknown_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM | n-to-n | multi-class | PASS |  | 0.2605 | 0.0000 | 0.7395 |  |  |  | 810.0000 | 810.0000 | 0.0000 |
| RandomForest | n-to-n | multi-class | PASS |  | 0.2630 | 0.0000 | 0.7370 |  |  |  | 810.0000 | 810.0000 | 0.0000 |
| KNN | n-to-n | multi-class | PASS |  | 0.1605 | 0.0000 | 0.8395 |  |  |  | 810.0000 | 810.0000 | 0.0000 |
| CNN | n-to-n | deep-learning | PASS |  | 0.2790 | 0.0000 | 0.7210 |  |  |  | 810.0000 | 810.0000 | 0.0000 |
| Clustering | n-to-n | unsupervised | PASS |  |  |  |  | 0.0405 | 0.4959 | -0.0018 |  |  |  |
| SVM | 1-to-1 | multi-class | FAIL | training returned no model; SVM requires at least 2 classes |  |  |  |  |  |  |  |  |  |
| RandomForest | 1-to-1 | multi-class | PASS |  | 1.0000 | 0.0000 | 0.0000 |  |  |  | 10.0000 | 10.0000 | 0.0000 |
| KNN | 1-to-1 | multi-class | PASS |  | 1.0000 | 0.0000 | 0.0000 |  |  |  | 10.0000 | 10.0000 | 0.0000 |
| CNN | 1-to-1 | deep-learning | PASS |  | 1.0000 | 0.0000 | 0.0000 |  |  |  | 10.0000 | 10.0000 | 0.0000 |
| Clustering | 1-to-1 | unsupervised | PASS |  |  |  |  | 1.0000 | 1.0000 | 0.0000 |  |  |  |
| SVM | 1-to-n | multi-class | FAIL | training returned no model; SVM requires at least 2 classes |  |  |  |  |  |  |  |  |  |
| RandomForest | 1-to-n | multi-class | PASS |  | 0.0144 | 0.0000 | 1.0000 |  |  |  | 694.0000 | 10.0000 | 684.0000 |
| KNN | 1-to-n | multi-class | PASS |  | 0.0144 | 0.0000 | 1.0000 |  |  |  | 694.0000 | 10.0000 | 684.0000 |
| CNN | 1-to-n | deep-learning | PASS |  | 0.0144 | 0.0000 | 1.0000 |  |  |  | 694.0000 | 10.0000 | 684.0000 |
| Clustering | 1-to-n | unsupervised | PASS |  |  |  |  | 0.0703 | 0.2370 | 0.0339 |  |  |  |
| SVM | large_combined | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_combined\\train' |  |  |  |  |  |  |  |  |  |
| RandomForest | large_combined | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_combined\\train' |  |  |  |  |  |  |  |  |  |
| KNN | large_combined | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_combined\\train' |  |  |  |  |  |  |  |  |  |
| CNN | large_combined | deep-learning | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_combined\\train' |  |  |  |  |  |  |  |  |  |
| Clustering | large_combined | unsupervised | PASS |  |  |  |  |  |  | 0.0665 |  |  |  |
| SVM | large_mixed | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_mixed\\train' |  |  |  |  |  |  |  |  |  |
| RandomForest | large_mixed | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_mixed\\train' |  |  |  |  |  |  |  |  |  |
| KNN | large_mixed | multi-class | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_mixed\\train' |  |  |  |  |  |  |  |  |  |
| CNN | large_mixed | deep-learning | FAIL | FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\Chris Boddy\\OneDrive\\Documents\\Cyber Security and Digital Forensics MSc\\Applications of Artificial Intelligence in Cyber Security\\Assessment\\project\\data\\large_mixed\\train' |  |  |  |  |  |  |  |  |  |
| Clustering | large_mixed | unsupervised | PASS |  |  |  |  |  |  | 0.0667 |  |  |  |

## Generated Figures

- `regenerated_results\figures\algorithm_accuracy_comparison.png`
- `regenerated_results\figures\gmm_1-to-n_score_distribution.png`
- `regenerated_results\figures\gmm_1-to-n_far_frr.png`

## Notes For Final Write-up

- Treat any `FAIL` row as an explicit exclusion, not missing data.
- For 1-to-n, open-set metrics distinguish known speaker samples from unknown/impostor samples.
- For large raw-audio datasets, segment-level clustering has no ground-truth ARI/NMI unless labels are later supplied.
- Historical files under `code/results/` are not used in this draft.

Latest GMM 1-to-n artefacts: `regenerated_results\gmm_ubm_1-to-n_20260512_214128`