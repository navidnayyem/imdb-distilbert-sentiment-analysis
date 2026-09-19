"""Train and evaluate a TF-IDF + Logistic Regression IMDb baseline.

This script reproduces the same duplicate removal and 80/10/10 split used by
the DistilBERT project. It tunes Logistic Regression on the validation set,
evaluates once on the held-out test set, saves the fitted baseline, and creates
a direct performance comparison with DistilBERT.

Run:
    python imdb_tfidf_logistic_baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


def parse_args() -> argparse.Namespace:
    desktop = Path(r"C:\Users\moham\OneDrive\Desktop")
    parser = argparse.ArgumentParser(
        description="IMDb TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument("--csv", type=Path, default=desktop / "IMDB Dataset.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=desktop / "imdb_baseline_output",
    )
    parser.add_argument(
        "--distilbert-metrics",
        type=Path,
        default=(
            desktop
            / "imdb_distilbert_output"
            / "evaluation_report"
            / "metrics_summary.json"
        ),
        help="DistilBERT metrics JSON created by imdb_model_analysis_cuda.py.",
    )
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_review(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_and_split(
    csv_path: Path, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"review", "sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")

    frame = frame[["review", "sentiment"]].dropna().copy()
    frame["sentiment"] = frame["sentiment"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["sentiment"]) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unexpected sentiment labels: {unknown}")

    original_count = len(frame)
    frame = frame.drop_duplicates(subset="review").reset_index(drop=True)
    frame["text"] = frame["review"].map(clean_review)
    frame["label"] = frame["sentiment"].map(LABEL_TO_ID).astype("int64")

    train_frame, temporary = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame["label"],
        random_state=seed,
    )
    validation_frame, test_frame = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label"],
        random_state=seed,
    )

    print(f"Loaded rows       : {original_count:,}")
    print(f"Duplicates removed: {original_count - len(frame):,}")
    print(
        f"Split sizes       : train={len(train_frame):,}, "
        f"validation={len(validation_frame):,}, test={len(test_frame):,}"
    )
    return train_frame, validation_frame, test_frame


def binary_metrics(
    labels: np.ndarray, predictions: np.ndarray, positive_probability: np.ndarray
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, positive_probability)),
        "average_precision": float(
            average_precision_score(labels, positive_probability)
        ),
    }


def tune_and_train(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    max_features: int,
    seed: int,
) -> tuple[Pipeline, pd.DataFrame]:
    print("\nFitting TF-IDF vectorizer on the training set...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(train_frame["text"])
    validation_features = vectorizer.transform(validation_frame["text"])
    print(f"TF-IDF feature matrix: {train_features.shape[0]:,} x {train_features.shape[1]:,}")

    tuning_rows: list[dict[str, float]] = []
    best_classifier: LogisticRegression | None = None
    best_f1 = -1.0
    best_c = None

    for c_value in [0.5, 1.0, 2.0, 4.0]:
        print(f"Training Logistic Regression with C={c_value:g}...")
        classifier = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1_000,
            random_state=seed,
        )
        classifier.fit(train_features, train_frame["label"])
        probabilities = classifier.predict_proba(validation_features)[:, 1]
        predictions = (probabilities >= 0.5).astype("int64")
        metrics = binary_metrics(
            validation_frame["label"].to_numpy(), predictions, probabilities
        )
        tuning_rows.append({"C": c_value, **metrics})
        print(
            f"  validation accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_c = c_value
            best_classifier = classifier

    if best_classifier is None:
        raise RuntimeError("No baseline classifier was trained.")
    print(f"\nSelected C={best_c:g} using validation F1={best_f1:.4f}")

    pipeline = Pipeline(
        [("tfidf", vectorizer), ("classifier", best_classifier)]
    )
    return pipeline, pd.DataFrame(tuning_rows)


def evaluate_test(
    pipeline: Pipeline, test_frame: pd.DataFrame, output_dir: Path
) -> dict[str, float]:
    probabilities = pipeline.predict_proba(test_frame["text"])[:, 1]
    predictions = (probabilities >= 0.5).astype("int64")
    labels = test_frame["label"].to_numpy()
    metrics = binary_metrics(labels, predictions, probabilities)
    metrics["test_samples"] = int(len(test_frame))

    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        labels,
        predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "baseline_classification_report.csv", index_label="class"
    )

    results = test_frame[["review", "sentiment", "label"]].copy()
    results["predicted_label"] = predictions
    results["predicted_sentiment"] = [ID_TO_LABEL[int(x)] for x in predictions]
    results["positive_probability"] = probabilities
    results["confidence"] = np.maximum(probabilities, 1.0 - probabilities)
    results["correct"] = results["label"] == results["predicted_label"]
    results.to_csv(output_dir / "baseline_test_predictions.csv", index=False)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    annotations = np.empty_like(matrix, dtype=object)
    row_percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    for row in range(2):
        for column in range(2):
            annotations[row, column] = (
                f"{matrix[row, column]:,}\n{row_percentages[row, column]:.1f}%"
            )
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Greens",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axis,
    )
    axis.set(
        title="TF-IDF + Logistic Regression confusion matrix",
        xlabel="Predicted",
        ylabel="Actual",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_confusion_matrix.png", dpi=300)
    plt.close(figure)
    return metrics


def plot_comparison(
    baseline_metrics: dict[str, float],
    distilbert_metrics_path: Path,
    output_dir: Path,
) -> None:
    if not distilbert_metrics_path.is_file():
        print(
            "Comparison plot skipped because DistilBERT metrics were not found at: "
            f"{distilbert_metrics_path}"
        )
        return

    with distilbert_metrics_path.open("r", encoding="utf-8") as file:
        distilbert_metrics = json.load(file)

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"]
    rows = []
    for key, label in zip(metric_keys, metric_labels):
        rows.append(
            {"Metric": label, "Model": "TF-IDF + Logistic Regression", "Score": baseline_metrics[key]}
        )
        rows.append(
            {"Metric": label, "Model": "DistilBERT", "Score": distilbert_metrics[key]}
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=comparison, x="Metric", y="Score", hue="Model", ax=axis)
    axis.set(
        title="IMDb sentiment model comparison",
        xlabel="",
        ylabel="Test score",
        ylim=(0, 1),
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    axis.legend(title="Model", loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    train_frame, validation_frame, test_frame = load_and_split(args.csv, args.seed)
    pipeline, tuning_results = tune_and_train(
        train_frame, validation_frame, args.max_features, args.seed
    )
    tuning_results.to_csv(args.output_dir / "validation_tuning_results.csv", index=False)
    joblib.dump(pipeline, args.output_dir / "tfidf_logistic_pipeline.joblib")

    metrics = evaluate_test(pipeline, test_frame, args.output_dir)
    plot_comparison(metrics, args.distilbert_metrics, args.output_dir)

    print("\nBaseline test results")
    for name, value in metrics.items():
        if name == "test_samples":
            print(f"{name:20s}: {int(value):,}")
        else:
            print(f"{name:20s}: {value:.4f}")
    print(f"\nBaseline outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
