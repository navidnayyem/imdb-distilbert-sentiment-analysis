"""Local Streamlit interface for the calibrated IMDb DistilBERT model.

Install Streamlit once:
    python -m pip install streamlit

Launch the app:
    streamlit run imdb_sentiment_streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DESKTOP = Path(r"C:\Users\moham\OneDrive\Desktop")
MODEL_DIR = DESKTOP / "imdb_distilbert_output" / "best_model"
TEMPERATURE_FILE = (
    DESKTOP
    / "imdb_distilbert_output"
    / "calibration_report"
    / "temperature.json"
)
MAX_LENGTH = 256


st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the calibrated DistilBERT model...")
def load_resources(model_directory: str, temperature_file: str):
    model_path = Path(model_directory)
    calibration_path = Path(temperature_file)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Temperature file not found: {calibration_path}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Start the app from the Conda environment where "
            "the RTX 4060 was detected."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()

    with calibration_path.open("r", encoding="utf-8") as file:
        temperature = float(json.load(file)["temperature"])
    if temperature <= 0:
        raise ValueError("The calibration temperature must be positive.")

    return tokenizer, model, device, temperature


def predict_sentiment(
    review: str,
    tokenizer,
    model,
    device: torch.device,
    temperature: float,
) -> dict[str, float | str]:
    encoded = tokenizer(
        review,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        logits = model(**encoded).logits.float()

    calibrated_probabilities = torch.softmax(logits / temperature, dim=-1)[0]
    negative_probability = float(calibrated_probabilities[0].cpu())
    positive_probability = float(calibrated_probabilities[1].cpu())
    label = "Positive" if positive_probability >= negative_probability else "Negative"
    confidence = max(negative_probability, positive_probability)
    token_count = int(encoded["input_ids"].shape[1])
    return {
        "label": label,
        "confidence": confidence,
        "negative_probability": negative_probability,
        "positive_probability": positive_probability,
        "token_count": token_count,
    }


st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.caption(
    "Fine-tuned DistilBERT with validation-fitted temperature scaling. "
    "The model classifies English movie reviews as positive or negative."
)

try:
    tokenizer, model, device, temperature = load_resources(
        str(MODEL_DIR), str(TEMPERATURE_FILE)
    )
except Exception as error:
    st.error(str(error))
    st.info(
        "Confirm that the trained model and calibration_report folders are inside "
        r"C:\Users\moham\OneDrive\Desktop\imdb_distilbert_output."
    )
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write("**Model:** DistilBERT")
    st.write("**Task:** Binary sentiment classification")
    st.write(f"**Device:** {torch.cuda.get_device_name(0)}")
    st.write(f"**Temperature:** {temperature:.6f}")
    st.write("**Test accuracy:** 91.51%")
    st.write("**Test F1-score:** 91.55%")
    st.write("**Test ROC-AUC:** 0.9724")
    st.divider()
    st.caption(
        "Probabilities are calibrated estimates, not guarantees. Sarcasm, mixed "
        "sentiment, and reviews longer than 256 tokens can still cause errors."
    )

with st.form("sentiment_form"):
    review = st.text_area(
        "Enter a movie review",
        height=210,
        placeholder=(
            "Example: The performances were excellent, and the story kept me "
            "engaged until the final scene."
        ),
        max_chars=15_000,
    )
    submitted = st.form_submit_button(
        "Analyze sentiment", type="primary", use_container_width=True
    )

if submitted:
    normalized_review = " ".join(review.split())
    if len(normalized_review) < 10:
        st.warning("Enter a meaningful movie review containing at least 10 characters.")
    else:
        with st.spinner("Analyzing the review on CUDA..."):
            result = predict_sentiment(
                normalized_review, tokenizer, model, device, temperature
            )

        if result["label"] == "Positive":
            st.success(f"Prediction: **Positive** — {result['confidence']:.2%} confidence")
        else:
            st.error(f"Prediction: **Negative** — {result['confidence']:.2%} confidence")

        left, right = st.columns(2)
        left.metric("Positive probability", f"{result['positive_probability']:.2%}")
        right.metric("Negative probability", f"{result['negative_probability']:.2%}")

        probability_frame = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Negative"],
                "Probability": [
                    result["positive_probability"],
                    result["negative_probability"],
                ],
            }
        ).set_index("Sentiment")
        st.bar_chart(probability_frame, y="Probability", height=280)

        if result["token_count"] >= MAX_LENGTH:
            st.warning(
                "This review reached the 256-token limit, so text after that point "
                "was not analyzed."
            )
        st.caption(
            f"Tokens analyzed: {result['token_count']} · "
            f"Calibration temperature: {temperature:.6f}"
        )

st.divider()
st.caption(
    "Project result: DistilBERT test accuracy 91.51%; TF-IDF + Logistic "
    "Regression baseline accuracy 91.45%."
)
