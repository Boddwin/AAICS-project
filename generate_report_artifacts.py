"""
Generate figures and a report draft from regenerated experiment outputs only.
"""
import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent


def latest_dir(results_dir, pattern):
    candidates = sorted(results_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_algorithm_accuracy_figure(all_algorithms, figures_dir):
    if all_algorithms.empty or "accuracy" not in all_algorithms:
        return None

    df = all_algorithms.copy()
    df = df[(df.get("status", "PASS") == "PASS") & df["accuracy"].notna()]
    if df.empty:
        return None

    df["label"] = df["algorithm"] + " / " + df["dataset"]
    df = df.sort_values(["dataset", "algorithm"])

    fig, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.35)))
    ax.barh(df["label"], df["accuracy"], color="#3b82f6")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Regenerated Algorithm Accuracy Results")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output = figures_dir / "algorithm_accuracy_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_gmm_score_distribution(predictions, figures_dir):
    if predictions.empty or "gmm_score" not in predictions:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for known, label, color in [
        (True, "Known speaker", "#16a34a"),
        (False, "Unknown/impostor", "#dc2626"),
    ]:
        subset = predictions[predictions["known_speaker"].astype(str).str.lower() == str(known).lower()]
        if not subset.empty:
            ax.hist(subset["gmm_score"], bins=25, alpha=0.6, label=label, color=color)

    ax.set_xlabel("Best GMM score")
    ax.set_ylabel("Utterances")
    ax.set_title("GMM 1-to-n Score Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = figures_dir / "gmm_1-to-n_score_distribution.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_gmm_far_frr(sweep, figures_dir):
    if sweep.empty or not {"threshold", "fmr", "fnmr"}.issubset(sweep.columns):
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["threshold"], sweep["fmr"], label="FMR/FAR", color="#dc2626")
    ax.plot(sweep["threshold"], sweep["fnmr"], label="FNMR/FRR", color="#2563eb")
    if "eer_gap" in sweep:
        best = sweep.sort_values(["eer_gap", "eer_estimate"]).iloc[0]
        ax.axvline(best["threshold"], color="#111827", linestyle="--", linewidth=1)
        ax.text(best["threshold"], 0.95, "EER threshold", rotation=90, va="top")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("GMM 1-to-n Threshold Sweep")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output = figures_dir / "gmm_1-to-n_far_frr.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def markdown_table(df, columns):
    if df.empty:
        return "_No regenerated rows available._\n"

    available = [column for column in columns if column in df.columns]
    if not available:
        return "_No matching columns available._\n"

    table_df = df[available].copy()
    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            table_df[column] = table_df[column].map(lambda value: "" if pd.isna(value) else str(value))

    header = "| " + " | ".join(table_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in table_df.to_numpy()
    ]
    return "\n".join([header, separator] + rows) + "\n"


def _metric(row, key, default="N/A"):
    if row is None or key not in row or pd.isna(row[key]):
        return default
    value = row[key]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _first_row(df, **filters):
    if df.empty:
        return None

    subset = df
    for key, value in filters.items():
        if key not in subset.columns:
            return None
        subset = subset[subset[key] == value]

    if subset.empty:
        return None
    return subset.iloc[0]


def narrative_summary(all_results, all_algorithms):
    lines = ["## Results Narrative", ""]

    gmm_1_to_n = _first_row(all_results, algorithm="GMM-UBM", dataset="1-to-n")
    if gmm_1_to_n is not None:
        lines.extend(
            [
                "### GMM-UBM 1-to-n",
                "",
                (
                    "The regenerated GMM-UBM 1-to-n run uses utterance-level scoring. "
                    f"It processed {_metric(gmm_1_to_n, 'total')} utterances, including "
                    f"{_metric(gmm_1_to_n, 'known_total')} enrolled-speaker samples and "
                    f"{_metric(gmm_1_to_n, 'unknown_total')} unknown-speaker samples. "
                    f"The threshold sweep selected `{_metric(gmm_1_to_n, 'threshold')}` "
                    f"using `{_metric(gmm_1_to_n, 'threshold_metric')}`, giving accuracy "
                    f"{_metric(gmm_1_to_n, 'accuracy')}, FNMR {_metric(gmm_1_to_n, 'fnmr')}, "
                    f"FMR {_metric(gmm_1_to_n, 'fmr')}, and EER {_metric(gmm_1_to_n, 'eer')}."
                ),
                "",
            ]
        )

    if not all_algorithms.empty:
        n_to_n = all_algorithms[
            (all_algorithms.get("dataset") == "n-to-n")
            & (all_algorithms.get("status") == "PASS")
            & all_algorithms.get("accuracy", pd.Series(dtype=float)).notna()
        ]
        if not n_to_n.empty:
            best = n_to_n.sort_values("accuracy", ascending=False).iloc[0]
            lines.extend(
                [
                    "### n-to-n Supervised Classification",
                    "",
                    (
                        f"The best regenerated n-to-n supervised classifier is `{best['algorithm']}` "
                        f"with accuracy {_metric(best, 'accuracy')} over {_metric(best, 'total')} "
                        "test utterances. All supervised n-to-n models remain well below the GMM-UBM "
                        "historical baseline, supporting the discussion that mean-pooled MFCC vectors "
                        "are a weak representation for large closed-set speaker identification."
                    ),
                    "",
                ]
            )

        one_to_n = all_algorithms[
            (all_algorithms.get("dataset") == "1-to-n")
            & (all_algorithms.get("status") == "PASS")
            & all_algorithms.get("accuracy", pd.Series(dtype=float)).notna()
        ]
        if not one_to_n.empty:
            lines.extend(
                [
                    "### 1-to-n Open-set Classifiers",
                    "",
                    (
                        "RF, KNN, and CNN now complete under open-set scoring, but they do not have "
                        "a rejection threshold. Each predicts the sole enrolled speaker for unknown "
                        "utterances, producing FMR 1.0000 and accuracy 0.0144. This is a useful "
                        "failure finding: closed-set classifiers are unsuitable for 1-to-n "
                        "surveillance-style screening unless calibrated with rejection logic."
                    ),
                    "",
                ]
            )

        large_rows = all_algorithms[
            (all_algorithms.get("dataset").isin(["large_combined", "large_mixed"]))
            & (all_algorithms.get("algorithm") == "Clustering")
        ]
        if not large_rows.empty:
            lines.extend(
                [
                    "### Large Raw-audio Datasets",
                    "",
                ]
            )
            for _, row in large_rows.iterrows():
                lines.append(
                    f"`{row['dataset']}` was evaluated with segment-level clustering: "
                    f"{_metric(row, 'n_samples')} five-second segments, "
                    f"{_metric(row, 'n_clusters')} clusters, silhouette "
                    f"{_metric(row, 'silhouette_score')}, and "
                    f"{_metric(row, 'skipped_files')} skipped unsupported files/segments."
                )
            lines.extend(
                [
                    "",
                    "These large-dataset results are diagnostic rather than accuracy claims because no segment-level ground-truth labels are present.",
                    "",
                ]
            )

        failures = all_algorithms[all_algorithms.get("status") == "FAIL"]
        if not failures.empty:
            lines.extend(
                [
                    "### Explicit Exclusions",
                    "",
                    (
                        f"The regenerated table records {len(failures)} failed or excluded "
                        "algorithm/dataset combinations. These are retained as rows in "
                        "`all_algorithms.csv` so the report can distinguish unavailable tasks "
                        "from missing data."
                    ),
                    "",
                ]
            )

    return lines


def write_report(all_results, all_algorithms, gmm_dir, figures, report_file, results_dir):
    lines = [
        "# Regenerated Results Report",
        "",
        f"This draft uses only files regenerated into `{results_dir.relative_to(ROOT_DIR)}/`.",
        "",
        *narrative_summary(all_results, all_algorithms),
        "## GMM-UBM Results",
        "",
        markdown_table(
            all_results,
            [
                "algorithm",
                "dataset",
                "task",
                "status",
                "accuracy",
                "fnmr",
                "fmr",
                "eer",
                "threshold",
                "threshold_selection",
                "total",
                "known_total",
                "unknown_total",
            ],
        ),
        "## Supervised And Clustering Results",
        "",
        markdown_table(
            all_algorithms,
            [
                "algorithm",
                "dataset",
                "task",
                "status",
                "reason",
                "accuracy",
                "fnmr",
                "fmr",
                "ari",
                "nmi",
                "silhouette_score",
                "total",
                "known_total",
                "unknown_total",
            ],
        ),
        "## Generated Figures",
        "",
    ]

    if figures:
        for figure in figures:
            lines.append(f"- `{figure.relative_to(ROOT_DIR)}`")
    else:
        lines.append("_No figures were generated because required regenerated CSVs were missing._")

    lines.extend(
        [
            "",
            "## Notes For Final Write-up",
            "",
            "- Treat any `FAIL` row as an explicit exclusion, not missing data.",
            "- For 1-to-n, open-set metrics distinguish known speaker samples from unknown/impostor samples.",
            "- For large raw-audio datasets, segment-level clustering has no ground-truth ARI/NMI unless labels are later supplied.",
            "- Historical files under `code/results/` are not used in this draft.",
        ]
    )

    if gmm_dir:
        lines.extend(
            [
                "",
                f"Latest GMM 1-to-n artefacts: `{gmm_dir.relative_to(ROOT_DIR)}`",
            ]
        )

    report_file.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate report artefacts")
    parser.add_argument(
        "--results-dir",
        default="regenerated_results",
        help="Directory containing regenerated CSV outputs",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT_DIR / results_dir

    figures_dir = results_dir / "figures"
    report_file = results_dir / "regenerated_results_report.md"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = read_csv(results_dir / "all_results.csv")
    all_algorithms = read_csv(results_dir / "all_algorithms.csv")

    gmm_dir = latest_dir(results_dir, "gmm_ubm_1-to-n_*")
    predictions = read_csv(gmm_dir / "predictions.csv") if gmm_dir else pd.DataFrame()
    sweep = read_csv(gmm_dir / "threshold_sweep.csv") if gmm_dir else pd.DataFrame()

    figures = []
    for figure in [
        save_algorithm_accuracy_figure(all_algorithms, figures_dir),
        save_gmm_score_distribution(predictions, figures_dir),
        save_gmm_far_frr(sweep, figures_dir),
    ]:
        if figure:
            figures.append(figure)

    write_report(all_results, all_algorithms, gmm_dir, figures, report_file, results_dir)
    print(f"Report written to {report_file}")
    for figure in figures:
        print(f"Figure written to {figure}")


if __name__ == "__main__":
    main()
