"""
Master experiment runner for systematic evaluation of all algorithms across all datasets.
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from logging_utils import ExperimentLogger, ResultsAggregator
from train_models_v2 import create_gmm_models, remove_previous_models
from test_speaker_v2 import evaluate_gmm_ubm_utterances


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "regenerated_results")


class ExperimentRunner:
    """Orchestrate all experiments across datasets and algorithms."""

    def __init__(self, base_results_dir="results"):
        self.base_results_dir = base_results_dir
        self.aggregator = ResultsAggregator(
            os.path.join(base_results_dir, "all_results.csv")
        )
        os.makedirs(base_results_dir, exist_ok=True)

    def run_gmm_ubm(self, dataset_name, dataset_path, models_folder, ubm_folder):
        """Run GMM-UBM training and testing on a dataset."""

        logger = ExperimentLogger(f"gmm_ubm_{dataset_name}", self.base_results_dir)
        logger.info(f"\n{'='*70}")
        logger.info(f"GMM-UBM EXPERIMENT: {dataset_name}")
        logger.info(f"{'='*70}\n")

        try:
            # Training
            logger.info("Starting training phase...")
            train_path = os.path.join(dataset_path, "train")
            if not os.path.isdir(train_path):
                logger.error(f"Training path not found: {train_path}")
                self.aggregator.add_result(
                    "GMM-UBM",
                    dataset_name,
                    self.get_task_type(dataset_name),
                    {
                        "status": "FAIL",
                        "reason": f"training path not found: {train_path}",
                    },
                )
                return False

            os.makedirs(models_folder, exist_ok=True)
            os.makedirs(ubm_folder, exist_ok=True)
            remove_previous_models(models_folder, ubm_folder, logger)

            if create_gmm_models(train_path, ubm_folder, models_folder, logger=logger) is None:
                self.aggregator.add_result(
                    "GMM-UBM",
                    dataset_name,
                    self.get_task_type(dataset_name),
                    {
                        "status": "FAIL",
                        "reason": "training returned no model",
                    },
                )
                return False

            # Testing
            logger.info("Starting testing phase...")
            test_path = os.path.join(dataset_path, "test")
            if not os.path.isdir(test_path):
                logger.error(f"Test path not found: {test_path}")
                self.aggregator.add_result(
                    "GMM-UBM",
                    dataset_name,
                    self.get_task_type(dataset_name),
                    {
                        "status": "FAIL",
                        "reason": f"test path not found: {test_path}",
                    },
                )
                return False

            ubm_file = os.path.join(ubm_folder, "ubm.gmm")

            test_results = evaluate_gmm_ubm_utterances(
                models_folder,
                test_path,
                ubm_file,
                threshold=None,
                threshold_metric="gmm_score",
                logger=logger,
                output_dir=logger.get_experiment_dir(),
            )

            if test_results:
                test_results["status"] = "PASS"
                task = self.get_task_type(dataset_name)
                self.aggregator.add_result(
                    "GMM-UBM", dataset_name, task, test_results
                )
                logger.info("\n[OK] GMM-UBM experiment completed successfully\n")
                return True

            self.aggregator.add_result(
                "GMM-UBM",
                dataset_name,
                self.get_task_type(dataset_name),
                {
                    "status": "FAIL",
                    "reason": "evaluation returned no metrics",
                },
            )
            return False

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.aggregator.add_result(
                "GMM-UBM",
                dataset_name,
                self.get_task_type(dataset_name),
                {
                    "status": "FAIL",
                    "reason": f"{type(e).__name__}: {e}",
                },
            )
            return False

    @staticmethod
    def get_task_type(dataset_name):
        """Return report task category for a dataset."""
        if dataset_name == "1-to-1":
            return "verification"
        if dataset_name == "1-to-n":
            return "identification"
        if dataset_name == "n-to-n":
            return "grouping"
        return "complex"

    def run_all_datasets(self):
        """Run GMM-UBM on all available datasets."""
        datasets = {
            "1-to-1": os.path.join(PROJECT_ROOT, "data", "1-to-1"),
            "1-to-n": os.path.join(PROJECT_ROOT, "data", "1-to-n"),
            "n-to-n": os.path.join(PROJECT_ROOT, "data", "n-to-n"),
            "large_combined": os.path.join(PROJECT_ROOT, "data", "large_combined"),
            "large_mixed": os.path.join(PROJECT_ROOT, "data", "large_mixed"),
        }

        results_summary = {
            "started_at": datetime.now().isoformat(),
            "datasets_tested": [],
            "experiments_passed": 0,
            "experiments_failed": 0,
        }

        for dataset_name, dataset_path in datasets.items():
            if not os.path.exists(dataset_path):
                print(f"⚠ Dataset not found: {dataset_path}")
                continue

            print(f"\n{'='*70}")
            print(f"Running GMM-UBM on {dataset_name}")
            print(f"{'='*70}")

            success = self.run_gmm_ubm(
                dataset_name,
                dataset_path,
                os.path.join(self.base_results_dir, "models", dataset_name, "speaker_models"),
                os.path.join(self.base_results_dir, "models", dataset_name, "ubm_model"),
            )

            results_summary["datasets_tested"].append(dataset_name)
            if success:
                results_summary["experiments_passed"] += 1
            else:
                results_summary["experiments_failed"] += 1

        results_summary["completed_at"] = datetime.now().isoformat()

        # Save aggregated results
        self.aggregator.save()
        self.aggregator.print_summary()

        print(f"\n{'='*70}")
        print("EXPERIMENT RUN SUMMARY")
        print(f"{'='*70}")
        print(f"Datasets tested:       {results_summary['experiments_passed'] + results_summary['experiments_failed']}")
        print(f"Experiments passed:    {results_summary['experiments_passed']}")
        print(f"Experiments failed:    {results_summary['experiments_failed']}")
        print(f"Aggregated results:    {self.aggregator.output_file}")
        print(f"{'='*70}\n")

        return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker identification experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Run specific dataset only (1-to-1, 1-to-n, n-to-n, large_combined, large_mixed)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.results_dir)

    if args.dataset:
        print(f"Running GMM-UBM on {args.dataset}")
        runner.run_gmm_ubm(
            args.dataset,
            os.path.join(PROJECT_ROOT, "data", args.dataset),
            os.path.join(runner.base_results_dir, "models", args.dataset, "speaker_models"),
            os.path.join(runner.base_results_dir, "models", args.dataset, "ubm_model"),
        )
        runner.aggregator.save()
    else:
        runner.run_all_datasets()
