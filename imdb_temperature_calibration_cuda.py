"""Calibrate the trained IMDb DistilBERT model using temperature scaling.

The temperature is fitted only on the validation set. The untouched test set
is then used once to compare probability calibration before and after scaling.
Model weights and class predictions are not changed.

Run:
    python imdb_temperature_calibration_cuda.py
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
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_TO_ID = {"negative": 0, "positive": 1}


def parse_args() -> argparse.Namespace:
    desktop = Path(r"C:\Users\moham\OneDrive\Desktop")
    parser = argparse.ArgumentParser(
        description="Temperature-scale IMDb DistilBERT probabilities."
    )
    parser.add_argument("--csv", type=Path, default=desktop / "IMDB Dataset.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=desktop / "imdb_distilbert_output" / "best_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=desktop / "imdb_distilbert_output" / "calibration_report",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_cuda() -> torch.device:
    print("=" * 72)
    print("CUDA / GPU CHECK")
    print("=" * 72)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Calibration stopped instead of using CPU.")
    torch.cuda.set_device(0)
    print(f"Selected GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 72)
    return torch.device("cuda:0")


def clean_review(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def reproduce_validation_and_test(
    csv_path: Path, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    frame = frame.drop_duplicates(subset="review").reset_index(drop=True)
    frame["text"] = frame["review"].map(clean_review)
    frame["label"] = frame["sentiment"].map(LABEL_TO_ID).astype("int64")

    train_frame, temporary = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame["label"],
        random_state=seed,
    )
    del train_frame
    validation_frame, test_frame = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label"],
        random_state=seed,
    )
    validation_frame = validation_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)
    print(
        f"Reproduced splits: validation={len(validation_frame):,}, "
        f"test={len(test_frame):,}"
    )
    return validation_frame, test_frame


def load_model(
    model_dir: Path, device: torch.device
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Saved model folder not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    model.eval()
    return tokenizer, model


def collect_logits(
    frame: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    description: str,
) -> np.ndarray:
    all_logits: list[np.ndarray] = []
    texts = frame["text"].tolist()
    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            logits = model(**encoded).logits
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def numpy_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    log_probabilities = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return float(-log_probabilities[np.arange(len(labels)), labels].mean())


def fit_temperature(validation_logits: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(
        lambda temperature: numpy_nll(validation_logits, labels, temperature),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def calibration_bins(
    probabilities: np.ndarray, labels: np.ndarray, number_of_bins: int
) -> pd.DataFrame:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == number_of_bins - 1:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if not selected.any():
            continue
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(selected.sum()),
                "mean_confidence": float(confidence[selected].mean()),
                "accuracy": float(correct[selected].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(bin_frame: pd.DataFrame, total: int) -> float:
    return float(
        (
            bin_frame["count"]
            / total
            * (bin_frame["accuracy"] - bin_frame["mean_confidence"]).abs()
        ).sum()
    )


def binary_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities[:, 1] - labels) ** 2))


def compute_metrics(
    probabilities: np.ndarray, labels: np.ndarray, number_of_bins: int
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = probabilities.argmax(axis=1)
    bins = calibration_bins(probabilities, labels, number_of_bins)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "average_precision": float(
            average_precision_score(labels, probabilities[:, 1])
        ),
        "negative_log_likelihood": float(log_loss(labels, probabilities)),
        "brier_score": binary_brier(probabilities, labels),
        "expected_calibration_error": expected_calibration_error(
            bins, len(labels)
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    return metrics, bins


def plot_reliability(
    before_bins: pd.DataFrame, after_bins: pd.DataFrame, output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(
        before_bins["mean_confidence"],
        before_bins["accuracy"],
        marker="o",
        linewidth=2,
        label="Before scaling",
    )
    axis.plot(
        after_bins["mean_confidence"],
        after_bins["accuracy"],
        marker="o",
        linewidth=2,
        label="After scaling",
    )
    axis.set(
        title="DistilBERT reliability diagram",
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "01_reliability_diagram.png", dpi=300)
    plt.close(figure)


def plot_confidence_histograms(
    before: np.ndarray, after: np.ndarray, output_dir: Path
) -> None:
    bins = np.linspace(0.5, 1.0, 21)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    axes[0].hist(before.max(axis=1), bins=bins, density=True, alpha=0.8)
    axes[0].set(
        title="Before temperature scaling",
        xlabel="Prediction confidence",
        ylabel="Density",
        xlim=(0.5, 1.0),
    )
    axes[1].hist(
        after.max(axis=1), bins=bins, density=True, alpha=0.8, color="#55A868"
    )
    axes[1].set(
        title="After temperature scaling",
        xlabel="Prediction confidence",
        xlim=(0.5, 1.0),
    )
    figure.suptitle("Test-set confidence distribution")
    figure.tight_layout()
    figure.savefig(output_dir / "02_confidence_before_after.png", dpi=300)
    plt.close(figure)


def plot_calibration_metrics(
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    metric_map = {
        "negative_log_likelihood": "NLL",
        "brier_score": "Brier score",
        "expected_calibration_error": "ECE",
    }
    rows = []
    for key, label in metric_map.items():
        rows.append({"Metric": label, "Stage": "Before", "Value": before_metrics[key]})
        rows.append({"Metric": label, "Stage": "After", "Value": after_metrics[key]})
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=frame, x="Metric", y="Value", hue="Stage", ax=axis)
    axis.set(title="Probability calibration improvement", xlabel="", ylabel="Lower is better")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.4f", padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "03_calibration_metrics.png", dpi=300)
    plt.close(figure)


def save_results(
    test_frame: pd.DataFrame,
    before: np.ndarray,
    after: np.ndarray,
    temperature: float,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    before_bins: pd.DataFrame,
    after_bins: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary = {
        "temperature": temperature,
        "fitted_on": "validation set only",
        "test_samples": int(len(test_frame)),
        "before_temperature_scaling": before_metrics,
        "after_temperature_scaling": after_metrics,
    }
    with (output_dir / "calibration_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "temperature.json").open("w", encoding="utf-8") as file:
        json.dump({"temperature": temperature}, file, indent=2)

    before_bins.assign(stage="before").to_csv(
        output_dir / "reliability_bins_before.csv", index=False
    )
    after_bins.assign(stage="after").to_csv(
        output_dir / "reliability_bins_after.csv", index=False
    )

    results = test_frame[["review", "sentiment", "label"]].copy()
    results["predicted_label"] = after.argmax(axis=1)
    results["uncalibrated_positive_probability"] = before[:, 1]
    results["calibrated_positive_probability"] = after[:, 1]
    results["uncalibrated_confidence"] = before.max(axis=1)
    results["calibrated_confidence"] = after.max(axis=1)
    results["correct"] = results["label"] == results["predicted_label"]
    results.to_csv(output_dir / "calibrated_test_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    device = require_cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    validation_frame, test_frame = reproduce_validation_and_test(args.csv, args.seed)
    tokenizer, model = load_model(args.model_dir, device)
    validation_logits = collect_logits(
        validation_frame,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        "Validation logits",
    )
    test_logits = collect_logits(
        test_frame,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        "Test logits",
    )

    validation_labels = validation_frame["label"].to_numpy()
    test_labels = test_frame["label"].to_numpy()
    temperature = fit_temperature(validation_logits, validation_labels)
    before_probabilities = softmax(test_logits, axis=1)
    after_probabilities = softmax(test_logits / temperature, axis=1)

    before_metrics, before_bins = compute_metrics(
        before_probabilities, test_labels, args.bins
    )
    after_metrics, after_bins = compute_metrics(
        after_probabilities, test_labels, args.bins
    )
    save_results(
        test_frame,
        before_probabilities,
        after_probabilities,
        temperature,
        before_metrics,
        after_metrics,
        before_bins,
        after_bins,
        args.output_dir,
    )
    plot_reliability(before_bins, after_bins, args.output_dir)
    plot_confidence_histograms(before_probabilities, after_probabilities, args.output_dir)
    plot_calibration_metrics(before_metrics, after_metrics, args.output_dir)

    print(f"\nOptimal temperature: {temperature:.6f}")
    print("\nCalibration results (lower is better)")
    for key in [
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
    ]:
        print(
            f"{key:30s}: before={before_metrics[key]:.6f}  "
            f"after={after_metrics[key]:.6f}"
        )
    print("\nClassification accuracy and F1 should remain unchanged:")
    print(
        f"accuracy: before={before_metrics['accuracy']:.4f}, "
        f"after={after_metrics['accuracy']:.4f}"
    )
    print(
        f"F1:       before={before_metrics['f1']:.4f}, "
        f"after={after_metrics['f1']:.4f}"
    )
    print(f"\nCalibration outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
