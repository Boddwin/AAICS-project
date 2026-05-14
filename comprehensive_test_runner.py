"""
Comprehensive test runner: Test all algorithms on all datasets
Direct import approach - starts with n-to-n dataset
"""
import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))

from logging_utils import ExperimentLogger, ResultsAggregator

# Datasets in order (n-to-n first)
DATASETS = [
    "n-to-n",           # 0: Start here - 100 speakers
    "1-to-1",           # 1: Single speaker verification
    "1-to-n",           # 2: Single speaker identification
    "large_combined",   # 3: Complex splitting
    "large_mixed",      # 4: Mixed grouping
]

ALGORITHM_TASKS = {
    "SVM": "multi-class",
    "RandomForest": "multi-class",
    "KNN": "multi-class",
    "CNN": "deep-learning",
    "Clustering": "unsupervised",
}

DEFAULT_RESULTS_DIR = "regenerated_results"

def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def import_module_from_file(name, file_path):
    """Import a module from a numbered experiment file."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_svm(dataset_name, root_dir, results_dir):
    """Test SVM on dataset."""
    try:
        svm_module = import_module_from_file(
            "svm_module",
            os.path.join(root_dir, "experiments", "02_svm.py"),
        )

        logger = ExperimentLogger(f"svm_{dataset_name}", results_dir)
        logger.save_config({"dataset": dataset_name, "algorithm": "SVM"})

        result = svm_module.train_svm(root_dir, dataset_name, logger=logger)
        if result:
            clf, scaler, le, logger = result
            metrics = svm_module.evaluate_svm(clf, scaler, le, root_dir, dataset_name, logger)
            if metrics:
                return True, metrics, ""
            return False, None, "evaluation returned no metrics"
        return False, None, "training returned no model; SVM requires at least 2 classes"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] SVM on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, reason


def test_rf(dataset_name, root_dir, results_dir):
    """Test Random Forest on dataset."""
    try:
        rf_module = import_module_from_file(
            "rf_module",
            os.path.join(root_dir, "experiments", "03_random_forest.py"),
        )

        logger = ExperimentLogger(f"rf_{dataset_name}", results_dir)
        logger.save_config({"dataset": dataset_name, "algorithm": "RandomForest"})

        result = rf_module.train_rf(root_dir, dataset_name, logger=logger)
        if result:
            clf, le, logger = result
            metrics = rf_module.evaluate_rf(clf, le, root_dir, dataset_name, logger)
            if metrics:
                return True, metrics, ""
            return False, None, "evaluation returned no metrics"
        return False, None, "training returned no model"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] RandomForest on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, reason


def test_knn(dataset_name, root_dir, results_dir):
    """Test KNN on dataset."""
    try:
        knn_module = import_module_from_file(
            "knn_module",
            os.path.join(root_dir, "experiments", "04_knn.py"),
        )

        logger = ExperimentLogger(f"knn_{dataset_name}", results_dir)
        logger.save_config({"dataset": dataset_name, "algorithm": "KNN"})

        result = knn_module.train_knn(root_dir, dataset_name, k=5, logger=logger)
        if result:
            clf, le, logger = result
            metrics = knn_module.evaluate_knn(clf, le, root_dir, dataset_name, logger)
            if metrics:
                return True, metrics, ""
            return False, None, "evaluation returned no metrics"
        return False, None, "training returned no model"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] KNN on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, reason


def test_clustering(dataset_name, root_dir, results_dir):
    """Test Clustering on dataset."""
    try:
        cluster_module = import_module_from_file(
            "cluster_module",
            os.path.join(root_dir, "experiments", "06_clustering.py"),
        )

        logger = ExperimentLogger(f"clustering_{dataset_name}", results_dir)
        logger.save_config({"dataset": dataset_name, "algorithm": "Clustering"})

        metrics = cluster_module.cluster_with_logging(root_dir, dataset_name, logger=logger)
        if metrics:
            return True, metrics, ""
        return False, None, "clustering returned no metrics"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] Clustering on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, reason


def test_cnn(dataset_name, root_dir, results_dir):
    """Test CNN on dataset."""
    try:
        if importlib.util.find_spec("tensorflow") is None:
            reason = "TensorFlow not installed"
            print(f"[SKIP] CNN on {dataset_name}: {reason}")
            return False, None, reason

        cnn_module = import_module_from_file(
            "cnn_module",
            os.path.join(root_dir, "experiments", "05_cnn.py"),
        )

        logger = ExperimentLogger(f"cnn_{dataset_name}", results_dir)
        logger.save_config({"dataset": dataset_name, "algorithm": "CNN"})

        result = cnn_module.train_cnn(root_dir, dataset_name, logger=logger)
        if result:
            model, le, X_shape, logger = result
            metrics = cnn_module.evaluate_cnn(model, le, X_shape, root_dir, dataset_name, logger)
            if metrics:
                return True, metrics, ""
            return False, None, "evaluation returned no metrics"
        return False, None, "training returned no model"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] CNN on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, reason


TEST_FUNCTIONS = [
    ("SVM", test_svm),
    ("RandomForest", test_rf),
    ("KNN", test_knn),
    ("CNN", test_cnn),
    ("Clustering", test_clustering),
]


def main(results_dir=None):
    """Run all algorithms on all datasets."""
    root_dir = get_root_dir()
    if results_dir is None:
        results_dir = os.path.join(root_dir, DEFAULT_RESULTS_DIR)
    elif not os.path.isabs(results_dir):
        results_dir = os.path.join(root_dir, results_dir)

    aggregator = ResultsAggregator(os.path.join(results_dir, "all_algorithms.csv"))
    total_tests = len(DATASETS) * len(TEST_FUNCTIONS)
    start_time = datetime.now()

    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM TEST SUITE")
    print("="*70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(DATASETS)} | Algorithms: {len(TEST_FUNCTIONS)}")
    print(f"Total Tests: {total_tests}")
    print("="*70)

    results_summary = {
        "start_time": start_time.isoformat(),
        "total_tests": total_tests,
        "datasets": DATASETS,
        "algorithms": [name for name, _ in TEST_FUNCTIONS],
        "dataset_results": {}
    }

    test_count = 0
    success_count = 0
    failed_tests = []

    for dataset_idx, dataset_name in enumerate(DATASETS):
        print(f"\n\n{'#'*70}")
        print(f"DATASET {dataset_idx + 1}/{len(DATASETS)}: {dataset_name.upper()}")
        print(f"{'#'*70}")

        dataset_results = {"algorithms": {}}

        for algo_name, test_func in TEST_FUNCTIONS:
            test_count += 1
            progress = f"[{test_count}/{total_tests}]"
            print(f"\n{progress} Testing {algo_name} on {dataset_name}...")

            success, metrics, reason = test_func(dataset_name, root_dir, results_dir)
            status = "PASS" if success else "FAIL"
            dataset_results["algorithms"][algo_name] = status

            aggregate_metrics = dict(metrics or {})
            aggregate_metrics["status"] = status
            aggregate_metrics["reason"] = reason
            aggregator.add_result(
                algo_name,
                dataset_name,
                ALGORITHM_TASKS.get(algo_name, "unknown"),
                aggregate_metrics,
            )

            if success:
                success_count += 1
                print(f"[OK] {algo_name} PASSED on {dataset_name}")
                if metrics:
                    print(f"     Metrics: {metrics}")
            else:
                failed_tests.append(f"{algo_name} on {dataset_name}")
                print(f"[FAIL] {algo_name} FAILED on {dataset_name}")
                if reason:
                    print(f"       Reason: {reason}")

        results_summary["dataset_results"][dataset_name] = dataset_results

    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests Run: {test_count}")
    print(f"Passed: {success_count}")
    print(f"Failed: {test_count - success_count}")
    success_rate = (success_count / test_count) * 100 if test_count > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    if failed_tests:
        print(f"\nFailed Tests:")
        for test in failed_tests:
            print(f"  - {test}")

    end_time = datetime.now()
    print(f"\nEnd Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary
    results_summary["end_time"] = end_time.isoformat()
    results_summary["success_count"] = success_count
    results_summary["failed_count"] = test_count - success_count
    results_summary["success_rate"] = success_rate

    os.makedirs(results_dir, exist_ok=True)
    aggregator.save()
    aggregator.print_summary()

    summary_file = os.path.join(results_dir, "comprehensive_test_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ML algorithms on all datasets")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for regenerated outputs",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir)
