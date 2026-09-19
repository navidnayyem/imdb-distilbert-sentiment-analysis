"""Fine-tune DistilBERT for IMDb sentiment classification using NVIDIA CUDA.

Recommended environment (Python 3.10 or 3.11):

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install "transformers>=4.46" datasets accelerate scikit-learn pandas matplotlib seaborn

Example:

    python imdb_distilbert_cuda.py

The program intentionally stops if CUDA is unavailable; it will not silently
train on the CPU. Outputs are written under --output-dir.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on the IMDb 50K review dataset with CUDA."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(r"C:\Users\moham\OneDrive\Desktop\IMDB Dataset.csv"),
        help="Path to the IMDb CSV dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("imdb_distilbert_output"))
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for a quick test; default uses all rows.",
    )
    return parser.parse_args()


def print_and_require_cuda() -> torch.device:
    print("=" * 72)
    print("CUDA / GPU CHECK")
    print("=" * 72)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA    : {torch.version.cuda}")
    print(f"GPU count       : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training was stopped to prevent CPU use. "
            "Install an NVIDIA driver and a CUDA-enabled PyTorch build, then rerun."
        )

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}          : {torch.cuda.get_device_name(index)}")
        print(f"GPU {index} VRAM     : {properties.total_memory / 1024**3:.2f} GB")

    torch.cuda.set_device(0)
    print(f"Selected device : cuda:0 ({torch.cuda.get_device_name(0)})")
    print("=" * 72)
    return torch.device("cuda:0")


def clean_review(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_split(csv_path: Path, seed: int, sample_size: int | None) -> DatasetDict:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path.resolve()}\n"
            "Pass its location with --csv, for example: "
            'python imdb_distilbert_cuda.py --csv "C:/data/IMDB Dataset.csv"'
        )

    frame = pd.read_csv(csv_path)
    required = {"review", "sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}; found {frame.columns.tolist()}")

    frame = frame[["review", "sentiment"]].dropna().copy()
    frame["sentiment"] = frame["sentiment"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["sentiment"]) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unexpected sentiment labels: {unknown}")

    initial_rows = len(frame)
    frame = frame.drop_duplicates(subset="review").reset_index(drop=True)
    duplicates_removed = initial_rows - len(frame)
    frame["text"] = frame["review"].map(clean_review)
    frame["label"] = frame["sentiment"].map(LABEL_TO_ID).astype("int64")
    frame = frame[["text", "label"]]

    if sample_size is not None:
        if not 100 <= sample_size <= len(frame):
            raise ValueError(f"--sample-size must be from 100 to {len(frame)}")
        frame, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame["label"],
            random_state=seed,
        )
        frame = frame.reset_index(drop=True)

    train_frame, temporary = train_test_split(
        frame, test_size=0.20, stratify=frame["label"], random_state=seed
    )
    validation_frame, test_frame = train_test_split(
        temporary, test_size=0.50, stratify=temporary["label"], random_state=seed
    )

    print(f"Loaded rows       : {initial_rows:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Rows used         : {len(frame):,}")
    print(
        f"Split sizes       : train={len(train_frame):,}, "
        f"validation={len(validation_frame):,}, test={len(test_frame):,}"
    )

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_frame, preserve_index=False),
            "validation": Dataset.from_pandas(validation_frame, preserve_index=False),
            "test": Dataset.from_pandas(test_frame, preserve_index=False),
        }
    )


def compute_metrics(evaluation_prediction) -> dict[str, float]:
    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    output_dir = str(args.output_dir / "checkpoints")
    values = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "fp16": True,
        "bf16": False,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "adamw_torch",
    }

    # Transformers has renamed or removed some TrainingArguments across versions.
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        values["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        values["evaluation_strategy"] = "steps"
    if "save_strategy" in parameters:
        values["save_strategy"] = "steps"

    unsupported = sorted(name for name in values if name not in parameters)
    if unsupported:
        print(
            "Transformers compatibility: ignoring unsupported TrainingArguments: "
            + ", ".join(unsupported)
        )
    compatible_values = {
        name: value for name, value in values.items() if name in parameters
    }
    return TrainingArguments(**compatible_values)


def save_evaluation(
    trainer: Trainer, tokenized: DatasetDict, output_dir: Path
) -> dict[str, float]:
    prediction_output = trainer.predict(tokenized["test"])
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    labels = prediction_output.label_ids
    metrics = {
        key.replace("test_", ""): float(value)
        for key, value in prediction_output.metrics.items()
        if isinstance(value, (float, int))
    }

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("IMDb Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("\nTest results")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")
    return metrics


def main() -> None:
    args = parse_args()
    device = print_and_require_cuda()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_and_split(args.csv, args.seed, args.sample_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing reviews",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Model is not on CUDA; training stopped.")
    print(f"Model device      : {next(model.parameters()).device}")

    training_args = make_training_arguments(args)
    trainer_values = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    # Compatibility with Transformers versions before/after tokenizer rename.
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_values["processing_class"] = tokenizer
    else:
        trainer_values["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_values)
    trainer.train()

    final_model_dir = args.output_dir / "best_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    save_evaluation(trainer, tokenized, args.output_dir)

    print(f"\nBest model saved to: {final_model_dir.resolve()}")
    print(f"All outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
