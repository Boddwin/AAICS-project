"""
Master algorithm runner - systematically run all algorithms on all datasets.
Integrates with the existing logging and aggregation infrastructure.
"""
import os
import sys
import argparse
import importlib.util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "utils"))
sys.path.insert(0, os.path.join(ROOT_DIR, "code"))
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments"))

from utils.logging_utils import ExperimentLogger, ResultsAggregator

VALID_DATASETS = ("1-to-1", "1-to-n", "n-to-n", "large_combined", "large_mixed")
VALID_ALGORITHMS = ("svm", "rf", "knn", "cnn", "clustering")


# Dynamic import helper
def import_module_from_file(name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load algorithm modules dynamically
experiments_dir = os.path.join(ROOT_DIR, "experiments")

svm_module = import_module_from_file("svm_module", os.path.join(experiments_dir, "02_svm.py"))
train_svm = svm_module.train_svm
evaluate_svm = svm_module.evaluate_svm

rf_module = import_module_from_file("rf_module", os.path.join(experiments_dir, "03_random_forest.py"))
train_rf = rf_module.train_rf
evaluate_rf = rf_module.evaluate_rf

knn_module = import_module_from_file("knn_module", os.path.join(experiments_dir, "04_knn.py"))
train_knn = knn_module.train_knn
evaluate_knn = knn_module.evaluate_knn

cluster_module = import_module_from_file("cluster_module", os.path.join(experiments_dir, "06_clustering.py"))
cluster_with_logging = cluster_module.cluster_with_logging

# CNN requires TensorFlow - make optional
HAS_TF = False
CNN_UNAVAILABLE_REASON = None
try:
    cnn_module = import_module_from_file("cnn_module", os.path.join(experiments_dir, "05_cnn.py"))
    train_cnn = cnn_module.train_cnn
    evaluate_cnn = cnn_module.evaluate_cnn
    if importlib.util.find_spec("tensorflow") is None:
        CNN_UNAVAILABLE_REASON = "TensorFlow not installed"
    else:
        HAS_TF = True
except ImportError as e:
    CNN_UNAVAILABLE_REASON = f"CNN dependencies unavailable: {e}"
except Exception as e:
    CNN_UNAVAILABLE_REASON = f"CNN module import failed: {type(e).__name__}: {e}"


class AlgorithmRunner:
    """Run all algorithms on specified datasets."""

    def __init__(self, base_dir, results_dir="results"):
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.aggregator = ResultsAggregator(os.path.join(results_dir, "all_algorithms.csv"))

    def run_svm(self, dataset_name):
        """Run SVM on dataset."""
        logger = ExperimentLogger(f"svm_{dataset_name}", self.results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"SVM: {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            result = train_svm(self.base_dir, dataset_name, logger=logger)
            if result:
                clf, scaler, le, logger = result
                metrics = evaluate_svm(clf, scaler, le, self.base_dir, dataset_name, logger)
                if metrics:
                    self.aggregator.add_result("SVM", dataset_name, "multi-class", metrics)
                    return True
        except Exception as e:
            logger.error(f"SVM failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def run_rf(self, dataset_name):
        """Run Random Forest on dataset."""
        logger = ExperimentLogger(f"rf_{dataset_name}", self.results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"Random Forest: {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            result = train_rf(self.base_dir, dataset_name, logger=logger)
            if result:
                clf, le, logger = result
                metrics = evaluate_rf(clf, le, self.base_dir, dataset_name, logger)
                if metrics:
                    self.aggregator.add_result("RandomForest", dataset_name, "multi-class", metrics)
                    return True
        except Exception as e:
            logger.error(f"Random Forest failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def run_knn(self, dataset_name):
        """Run KNN on dataset."""
        logger = ExperimentLogger(f"knn_{dataset_name}", self.results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"KNN: {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            result = train_knn(self.base_dir, dataset_name, k=5, logger=logger)
            if result:
                clf, le, logger = result
                metrics = evaluate_knn(clf, le, self.base_dir, dataset_name, logger)
                if metrics:
                    self.aggregator.add_result("KNN", dataset_name, "multi-class", metrics)
                    return True
        except Exception as e:
            logger.error(f"KNN failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def run_cnn(self, dataset_name):
        """Run CNN on dataset."""
        if not HAS_TF:
            reason = CNN_UNAVAILABLE_REASON or "TensorFlow not installed"
            print(f"[SKIP] CNN - {reason}")
            return False

        logger = ExperimentLogger(f"cnn_{dataset_name}", self.results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"CNN: {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            result = train_cnn(self.base_dir, dataset_name, logger=logger)
            if result:
                model, le, X_shape, logger = result
                metrics = evaluate_cnn(model, le, X_shape, self.base_dir, dataset_name, logger)
                if metrics:
                    self.aggregator.add_result("CNN", dataset_name, "deep-learning", metrics)
                    return True
        except Exception as e:
            logger.error(f"CNN failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def run_clustering(self, dataset_name):
        """Run Clustering on dataset."""
        logger = ExperimentLogger(f"clustering_{dataset_name}", self.results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"Clustering (K-Means): {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            metrics = cluster_with_logging(self.base_dir, dataset_name, logger=logger)
            if metrics:
                self.aggregator.add_result("KMeans-Clustering", dataset_name, "unsupervised", metrics)
                return True
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def run_all_algorithms(self, dataset_name):
        """Run all algorithms on a dataset."""
        print(f"\n{'='*70}")
        print(f"Testing all algorithms on {dataset_name}")
        print(f"{'='*70}")

        algorithms = [
            ("SVM", self.run_svm),
            ("Random Forest", self.run_rf),
            ("KNN", self.run_knn),
            ("CNN", self.run_cnn),
            ("Clustering", self.run_clustering),
        ]

        results = {}
        for name, func in algorithms:
            print(f"\n[*] Running {name}...")
            try:
                success = func(dataset_name)
                results[name] = "OK" if success else "FAILED"
            except KeyboardInterrupt:
                print("[INTERRUPTED]")
                raise

        return results

    def save_and_report(self):
        """Save aggregated results and print report."""
        self.aggregator.save()
        self.aggregator.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all algorithms on datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        default="n-to-n",
        help="Dataset to test"
    )
    parser.add_argument(
        "--algorithms",
        type=str.lower,
        nargs="+",
        choices=VALID_ALGORITHMS,
        help="Specific algorithms to run"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory"
    )

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    runner = AlgorithmRunner(root_dir, args.results_dir)

    if args.algorithms:
        print(f"Running specific algorithms: {', '.join(args.algorithms)}")
        algorithm_methods = {
            "svm": ("SVM", runner.run_svm),
            "rf": ("Random Forest", runner.run_rf),
            "knn": ("KNN", runner.run_knn),
            "cnn": ("CNN", runner.run_cnn),
            "clustering": ("Clustering", runner.run_clustering),
        }
        results = {}
        for algo in args.algorithms:
            display_name, run_algorithm = algorithm_methods[algo]
            success = run_algorithm(args.dataset)
            results[display_name] = "OK" if success else "FAILED"
    else:
        results = runner.run_all_algorithms(args.dataset)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for algo, status in results.items():
        print(f"{algo:20} {status}")

    runner.save_and_report()
