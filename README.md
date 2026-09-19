# IMDb Sentiment Classification with DistilBERT and CUDA

This project fine-tunes `distilbert-base-uncased` for binary IMDb movie-review sentiment classification. It includes CUDA-only training and inference, held-out evaluation, a TF-IDF logistic-regression baseline, probability calibration, plots, and a Streamlit demo.

## Project results

The original CSV contained 50,000 reviews. After removing 418 duplicate reviews, 49,582 unique reviews remained. The project used a reproducible stratified 80/10/10 split with seed 42.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT | 0.9151 | 0.9143 | 0.9168 | 0.9155 | 0.9724 | 0.9705 |
| TF-IDF + Logistic Regression | 0.9145 | 0.9083 | 0.9229 | 0.9155 | 0.9715 | 0.9705 |

The DistilBERT test confusion matrix was:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 2,256 | 214 |
| Actual positive | 207 | 2,282 |

The transformer only marginally outperformed the baseline: accuracy improved by 0.0006, F1 was identical, and ROC-AUC improved by 0.0009. This is an important result—the simpler baseline is nearly as effective on this dataset.

## Probability calibration

Temperature scaling was fitted only on the validation set. The optimal temperature was `1.857998`.

| Calibration metric | Before | After | Change |
|---|---:|---:|---:|
| Negative log-likelihood | 0.301062 | 0.227487 | 24.4% lower |
| Brier score | 0.071906 | 0.065014 | 9.6% lower |
| Expected calibration error | 0.061494 | 0.032140 | 47.7% lower |

Temperature scaling does not change the predicted class, accuracy, or F1. It makes the displayed probabilities less overconfident and more meaningful.

## Files

| File | Purpose |
|---|---|
| `imdb_distilbert_cuda.py` | Fine-tunes and evaluates DistilBERT on CUDA |
| `imdb_model_analysis_cuda.py` | Generates detailed metrics, predictions, and evaluation plots |
| `imdb_tfidf_logistic_baseline.py` | Trains and evaluates the classical baseline |
| `imdb_temperature_calibration_cuda.py` | Fits validation-set temperature scaling and evaluates calibration |
| `imdb_sentiment_streamlit_app.py` | Runs the interactive calibrated sentiment analyzer |
| `requirements-imdb.txt` | Installs non-PyTorch project dependencies |

## Expected Windows paths

The scripts are configured for:

```text
C:\Users\moham\OneDrive\Desktop\IMDB Dataset.csv
C:\Users\moham\OneDrive\Desktop\imdb_distilbert_output\best_model
C:\Users\moham\OneDrive\Desktop\imdb_distilbert_output\calibration_report\temperature.json
```

Run the commands from:

```text
C:\Users\moham\OneDrive\Desktop
```

## Environment setup

Create and activate an isolated Python 3.11 environment:

```bat
conda create -n llm python=3.11 -y
conda activate llm
set PYTHONNOUSERSITE=1
```

Install the CUDA 12.8 PyTorch build, followed by the project packages:

```bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-imdb.txt
```

Verify CUDA before training:

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The training, analysis, calibration, and Streamlit scripts intentionally stop if CUDA is unavailable.

## Run the complete workflow

Run each command in this order:

```bat
python imdb_distilbert_cuda.py
python imdb_model_analysis_cuda.py
python imdb_tfidf_logistic_baseline.py
python imdb_temperature_calibration_cuda.py
streamlit run imdb_sentiment_streamlit_app.py
```

The Streamlit application opens at `http://localhost:8501` and displays calibrated positive and negative probabilities.

## Main outputs

```text
imdb_distilbert_output/
├── best_model/
├── evaluation_report/
│   ├── 01_confusion_matrix.png
│   ├── 02_roc_and_precision_recall.png
│   ├── 03_class_metrics.png
│   ├── 04_confidence_distribution.png
│   ├── 05_training_history.png
│   ├── classification_report.csv
│   ├── metrics_summary.json
│   ├── misclassified_reviews.csv
│   └── test_predictions.csv
└── calibration_report/
    ├── 01_reliability_diagram.png
    ├── 02_confidence_before_after.png
    ├── 03_calibration_metrics.png
    ├── calibrated_test_predictions.csv
    ├── calibration_metrics.json
    ├── reliability_bins_after.csv
    ├── reliability_bins_before.csv
    └── temperature.json
```

The baseline model and its reports are stored in `imdb_baseline_output`.

## Limitations

- The model was evaluated only on the IMDb dataset; performance may decline on reviews from other domains.
- Input is truncated to 256 tokens, so content after that limit is ignored.
- Sarcasm, mixed sentiment, negation, and unusual phrasing remain difficult.
- Confidence is an estimated probability, not a guarantee.
- The model is intended for demonstration and research, not high-stakes decision-making.

## Dataset citation

This project uses the Large Movie Review Dataset introduced by Maas et al. If you use this project or dataset in academic work, cite:

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

- Dataset used: [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Paper: [ACL Anthology P11-1015](https://aclanthology.org/P11-1015/)
