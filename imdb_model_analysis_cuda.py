"""Evaluate the trained IMDb DistilBERT model and generate project plots.

Run after imdb_distilbert_cuda.py has finished:

    python imdb_model_analysis_cuda.py

The default paths match the project created on the user's Windows Desktop.
This script requires CUDA and does not retrain the model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


def parse_args() -> argparse.Namespace:
    desktop = Path(r"C:\Users\moham\OneDrive\Desktop")
    parser = argparse.ArgumentParser(
        description="Generate plots and detailed test-set analysis for IMDb DistilBERT."
    )
    parser.add_argument("--csv", type=Path, default=desktop / "IMDB Dataset.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=desktop / "imdb_distilbert_output" / "best_model",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=desktop / "imdb_distilbert_output",
        help="Folder containing checkpoints; used only for the training-history plot.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=desktop / "imdb_distilbert_output" / "evaluation_report",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_cuda() -> torch.device:
    print("=" * 72)
    print("CUDA / GPU CHECK")
    print("=" * 72)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Analysis stopped instead of using CPU.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Selected GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 72)
    return device


def clean_review(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def reproduce_test_split(csv_path: Path, seed: int) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"review", "sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")

    frame = frame[["review", "sentiment"]].dropna().copy()
    frame["sentiment"] = frame["sentiment"].astype(str).str.strip().str.lower()
    frame = frame.drop_duplicates(subset="review").reset_index(drop=True)
    frame["text"] = frame["review"].map(clean_review)
    frame["label"] = frame["sentiment"].map(LABEL_TO_ID)
    if frame["label"].isna().any():
        raise ValueError("The sentiment column contains labels other than positive/negative.")
    frame["label"] = frame["label"].astype("int64")

    train_frame, temporary = train_test_split(
        frame, test_size=0.20, stratify=frame["label"], random_state=seed
    )
    del train_frame
    _, test_frame = train_test_split(
        temporary, test_size=0.50, stratify=temporary["label"], random_state=seed
    )
    test_frame = test_frame.reset_index(drop=True)
    print(f"Reproduced test set: {len(test_frame):,} reviews")
    return test_frame


def predict(
    frame: pd.DataFrame,
    model_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Saved model folder not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    model.eval()

    probabilities: list[np.ndarray] = []
    texts = frame["text"].tolist()
    for start in tqdm(range(0, len(texts), batch_size), desc="Running CUDA inference"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            logits = model(**encoded).logits
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    probability_matrix = np.concatenate(probabilities, axis=0)
    predictions = probability_matrix.argmax(axis=1)
    return predictions, probability_matrix


def save_metric_files(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    report_dir: Path,
) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    positive_probability = probabilities[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "test_samples": int(len(frame)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, positive_probability)),
        "average_precision": float(
            average_precision_score(labels, positive_probability)
        ),
    }
    with (report_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        labels,
        predictions,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        report_dir / "classification_report.csv", index_label="class"
    )

    results = frame[["review", "sentiment", "label"]].copy()
    results["predicted_label"] = predictions
    results["predicted_sentiment"] = [ID_TO_LABEL[int(x)] for x in predictions]
    results["negative_probability"] = probabilities[:, 0]
    results["positive_probability"] = probabilities[:, 1]
    results["confidence"] = probabilities.max(axis=1)
    results["correct"] = results["label"] == results["predicted_label"]
    results.to_csv(report_dir / "test_predictions.csv", index=False)
    (
        results.loc[~results["correct"]]
        .sort_values("confidence", ascending=False)
        .to_csv(report_dir / "misclassified_reviews.csv", index=False)
    )
    return metrics


def plot_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, report_dir: Path
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(2):
        for column in range(2):
            annotations[row, column] = (
                f"{matrix[row, column]:,}\n{row_percent[row, column]:.1f}%"
            )

    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axis,
    )
    axis.set(title="Test-set confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(report_dir / "01_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_roc_pr(
    labels: np.ndarray, positive_probability: np.ndarray, report_dir: Path
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels, positive_probability
    )
    roc_auc = roc_auc_score(labels, positive_probability)
    precision, recall, _ = precision_recall_curve(labels, positive_probability)
    average_precision = average_precision_score(labels, positive_probability)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2.3,
        label=f"DistilBERT (AUC = {roc_auc:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[0].set(
        title="ROC curve",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall,
        precision,
        linewidth=2.3,
        label=f"DistilBERT (AP = {average_precision:.4f})",
    )
    axes[1].axhline(labels.mean(), linestyle="--", color="gray", label="Prevalence")
    axes[1].set(
        title="Precision–recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(report_dir / "02_roc_and_precision_recall.png", dpi=300)
    plt.close(fig)


def plot_class_metrics(
    labels: np.ndarray, predictions: np.ndarray, report_dir: Path
) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    plot_frame = pd.DataFrame(
        {
            "Class": ["Negative", "Positive"] * 3,
            "Metric": ["Precision"] * 2 + ["Recall"] * 2 + ["F1-score"] * 2,
            "Score": np.concatenate([precision, recall, f1]),
        }
    )
    fig, axis = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=plot_frame, x="Metric", y="Score", hue="Class", ax=axis)
    axis.set(title="Class-wise test performance", xlabel="", ylabel="Score", ylim=(0, 1))
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3)
    axis.legend(title="Class", loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "03_class_metrics.png", dpi=300)
    plt.close(fig)


def plot_confidence(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    report_dir: Path,
) -> None:
    confidence = probabilities.max(axis=1)
    correct = labels == predictions
    fig, axis = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0.5, 1.0, 21)
    axis.hist(
        confidence[correct], bins=bins, alpha=0.75, label="Correct", color="#2878B5"
    )
    axis.hist(
        confidence[~correct], bins=bins, alpha=0.75, label="Incorrect", color="#D9534F"
    )
    axis.set(
        title="Prediction confidence",
        xlabel="Maximum predicted probability",
        ylabel="Number of reviews",
        xlim=(0.5, 1.0),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "04_confidence_distribution.png", dpi=300)
    plt.close(fig)


def find_latest_trainer_state(training_output_dir: Path) -> Path | None:
    candidates = list(training_output_dir.glob("checkpoints/checkpoint-*/trainer_state.json"))
    if not candidates:
        candidates = list(training_output_dir.rglob("trainer_state.json"))

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", str(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=checkpoint_step) if candidates else None


def plot_training_history(training_output_dir: Path, report_dir: Path) -> None:
    state_path = find_latest_trainer_state(training_output_dir)
    if state_path is None:
        print("Training-history plot skipped: trainer_state.json was not found.")
        return

    with state_path.open("r", encoding="utf-8") as file:
        history = json.load(file).get("log_history", [])
    train_rows = [row for row in history if "loss" in row and "eval_loss" not in row]
    eval_rows = [row for row in history if "eval_loss" in row]
    if not train_rows or not eval_rows:
        print("Training-history plot skipped: log history is incomplete.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(
        [row["step"] for row in train_rows],
        [row["loss"] for row in train_rows],
        label="Training loss",
        linewidth=1.8,
    )
    axes[0].plot(
        [row["step"] for row in eval_rows],
        [row["eval_loss"] for row in eval_rows],
        marker="o",
        label="Validation loss",
        linewidth=2,
    )
    axes[0].set(title="Loss during training", xlabel="Training step", ylabel="Loss")
    axes[0].legend()

    for key, label in [
        ("eval_accuracy", "Accuracy"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("eval_f1", "F1-score"),
    ]:
        axes[1].plot(
            [row["step"] for row in eval_rows if key in row],
            [row[key] for row in eval_rows if key in row],
            marker="o",
            label=label,
        )
    axes[1].set(
        title="Validation metrics during training",
        xlabel="Training step",
        ylabel="Score",
        ylim=(0.8, 1.0),
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(report_dir / "05_training_history.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = require_cuda()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    test_frame = reproduce_test_split(args.csv, args.seed)
    predictions, probabilities = predict(
        test_frame,
        args.model_dir,
        device,
        args.batch_size,
        args.max_length,
    )
    labels = test_frame["label"].to_numpy()
    metrics = save_metric_files(
        test_frame, predictions, probabilities, args.report_dir
    )
    plot_confusion_matrix(labels, predictions, args.report_dir)
    plot_roc_pr(labels, probabilities[:, 1], args.report_dir)
    plot_class_metrics(labels, predictions, args.report_dir)
    plot_confidence(labels, predictions, probabilities, args.report_dir)
    plot_training_history(args.training_output_dir, args.report_dir)

    print("\nFinal test metrics")
    for name, value in metrics.items():
        if name == "test_samples":
            print(f"{name:20s}: {int(value):,}")
        else:
            print(f"{name:20s}: {value:.4f}")
    print(f"\nEvaluation report saved to: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
