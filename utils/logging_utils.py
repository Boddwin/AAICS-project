import logging
import os
from datetime import datetime
import json
import csv


def _ordered_fieldnames(records, preferred_fields=None):
    """Return stable CSV fieldnames using preferred fields first, then discovered keys."""
    fieldnames = []

    for field in preferred_fields or []:
        if any(field in record for record in records):
            fieldnames.append(field)

    for record in records:
        for field in record:
            if field not in fieldnames:
                fieldnames.append(field)

    return fieldnames


def _format_metric(value):
    """Format numeric metrics without crashing on missing or non-numeric values."""
    if value is None or value == "":
        return "N/A"

    if isinstance(value, int):
        return str(value)

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


class ExperimentLogger:
    """Centralized logging for ML experiments with file and console output."""

    def __init__(self, experiment_name, results_dir="results"):
        """
        Initialize logger for an experiment.

        Args:
            experiment_name: str, name of the experiment (e.g., "gmm_ubm_1to1")
            results_dir: str, directory to store results
        """
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            results_dir, f"{experiment_name}_{self.timestamp}"
        )

        os.makedirs(self.experiment_dir, exist_ok=True)

        # Setup file logger
        self.log_file = os.path.join(self.experiment_dir, "experiment.log")
        self.setup_file_logger()

        # Setup metrics CSV
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        self.metrics_data = []

    def setup_file_logger(self):
        """Configure file and console logging."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, level, message):
        """Log a message at specified level."""
        getattr(self.logger, level)(message)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def save_metrics(self, stage, metrics_dict):
        """Save metrics to CSV (stage-wise aggregation)."""
        record = dict(metrics_dict)
        record["stage"] = stage
        record["timestamp"] = datetime.now().isoformat()
        self.metrics_data.append(record)

        fieldnames = _ordered_fieldnames(
            self.metrics_data,
            preferred_fields=["stage", "timestamp"],
        )

        with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_data)

        self.info(f"Metrics saved for stage: {stage}")

    def save_config(self, config_dict):
        """Save experiment configuration."""
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        self.info(f"Config saved: {config_file}")

    def save_results_summary(self, summary_dict):
        """Save summary of results."""
        summary_file = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_dict, f, indent=2)
        self.info(f"Summary saved: {summary_file}")

    def get_experiment_dir(self):
        """Return the experiment output directory."""
        return self.experiment_dir


class ResultsAggregator:
    """Aggregate results across multiple experiments."""

    def __init__(self, output_file="results/aggregated_results.csv"):
        self.output_file = output_file
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def add_result(self, algorithm, dataset, task, metrics_dict):
        """Add a single experiment result."""
        result = {
            "algorithm": algorithm,
            "dataset": dataset,
            "task": task,
            "timestamp": datetime.now().isoformat(),
        }
        result.update(dict(metrics_dict))
        self.results.append(result)

    def save(self):
        """Save aggregated results to CSV."""
        if not self.results:
            print("No results to save")
            return

        fieldnames = _ordered_fieldnames(
            self.results,
            preferred_fields=["algorithm", "dataset", "task", "timestamp", "status", "reason"],
        )
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Aggregated results saved to {self.output_file}")

    def print_summary(self):
        """Print a summary of results grouped by algorithm and dataset."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY BY ALGORITHM AND DATASET")
        print("=" * 80)

        # Group by algorithm and dataset
        grouped = {}
        for result in self.results:
            algo = result["algorithm"]
            dataset = result["dataset"]
            key = f"{algo}_{dataset}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        for key, results in sorted(grouped.items()):
            print(f"\n{key}:")
            for r in results:
                if "status" in r:
                    print(f"  Status      {r.get('status')}")
                if r.get("reason"):
                    print(f"  Reason      {r.get('reason')}")

                summary_fields = [
                    ("Accuracy", "accuracy"),
                    ("FNMR", "fnmr"),
                    ("FMR", "fmr"),
                    ("ARI", "ari"),
                    ("NMI", "nmi"),
                    ("Silhouette", "silhouette_score"),
                ]

                printed = False
                for label, metric_key in summary_fields:
                    if metric_key in r:
                        print(f"  {label:12} {_format_metric(r.get(metric_key))}")
                        printed = True

                if not printed:
                    print("  No summary metrics available")
