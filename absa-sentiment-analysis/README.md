# Aspect-Based Sentiment Analysis (ABSA)

Production-ready NLP project for **Aspect-Based Sentiment Analysis** using Python, PyTorch, HuggingFace Transformers, and Scikit-learn. The system is modular, reproducible, and runs end-to-end.

## Project Overview

Given a sentence and an aspect term (e.g. "food", "service"), the system predicts the **polarity** of the aspect: **positive**, **negative**, **neutral**, or **conflict**. Three model families are implemented:

| Model | Description |
|-------|-------------|
| **Baseline** | TF-IDF features on (sentence + aspect) with Logistic Regression or SVM |
| **Bi-LSTM** | Bidirectional LSTM with GloVe embeddings (100d) |
| **Transformer** | BERT fine-tuning (`bert-base-uncased`) for sequence classification |

## Dataset

**Recommended:** put SemEval **tabular train** CSVs in `data/raw/` (columns such as `id`, `Sentence`, `Aspect Term`, `polarity`, `from`, `to`). In `configs/*.yaml`, list them under `csv_train_files` (defaults include `restaurants_train.csv` and `laptops_train.csv`). The pipeline merges them, stratified-splits into train/val/test, and writes `data/processed/*.csv`.

**SemEval XML** is also supported: set `raw_xml` to the filename in `data/raw/` and leave `csv_train_files` empty or unused.

**Unlabeled test files** (e.g. `restaurants_test.csv` with only `id`, `Sentence`) are **not** used for training or evaluation in this project; the test split is held out from the labeled train CSVs.

Use `rebuild_processed: true` in config or **`python src/train.py --model ... --rebuild-data`** to ignore cached `data/processed/*.csv` and rebuild from CSV/XML.

If no CSV/XML/cache is available, the trainer falls back to **minimal dummy data**.

Processed splits are cached under `data/processed/` for reproducibility.

## Model Details

- **Baseline**: TF-IDF (max 10k features, unigrams + bigrams) → Logistic Regression or linear SVM with class weighting.
- **LSTM**: Embedding → Bi-LSTM (2 layers, hidden 128) → Dropout 0.5 → Linear → 4 classes. GloVe 100d loaded from file if `glove_path` is set; otherwise random init.
- **BERT**: `bert-base-uncased` with a classification head; input format `sentence [SEP] aspect`. Fine-tuned with HuggingFace `Trainer` (AdamW, warmup, weight decay).

## Results

After training, metrics are written to `results/metrics/all_results.csv` (accuracy, precision, recall, F1, MCC). Confusion matrices and training curves are saved under `results/figures/`. Example table (fill after running):

| Model    | Accuracy | F1 (weighted) | MCC   |
|----------|----------|---------------|-------|
| Baseline | -        | -             | -     |
| LSTM     | -        | -             | -     |
| BERT     | -        | -             | -     |

## Installation

```bash
cd absa-sentiment-analysis
pip install -r requirements.txt
```

Recommended: use Python 3.10 and a virtual environment.

## Project Structure

```
absa-sentiment-analysis/
├── data/
│   ├── raw/           # SemEval XML (and optional GloVe .txt)
│   └── processed/     # train.csv, val.csv, test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_bert_finetune.ipynb
├── src/
│   ├── data_loader.py   # parse_semeval_xml, load_and_split
│   ├── preprocess.py    # clean_text, format_bert_input, encode_labels
│   ├── train.py         # Unified CLI training
│   ├── evaluate.py      # compute_all_metrics, plots, save_metrics_table
│   ├── utils.py         # set_seed, logging, device
│   └── models/
│       ├── baseline.py  # TF-IDF + LR/SVC
│       ├── lstm_model.py# ABSADataset, BiLSTMClassifier
│       └── bert_model.py# BERT + Trainer support
├── configs/
│   ├── baseline_config.yaml
│   ├── lstm_config.yaml
│   └── bert_config.yaml
├── results/
│   ├── figures/        # Confusion matrices, training curves
│   ├── metrics/        # all_results.csv
│   └── checkpoints/    # Saved models
├── tests/
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

## Running the Pipeline

From the **project root** `absa-sentiment-analysis/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Train each model (uses dummy data if no XML/CSV present)
python src/train.py --model baseline
python src/train.py --model lstm
python src/train.py --model bert
```

Optional:

- `--config path/to/config.yaml` to override default config path.
- `--data_dir path` and `--seed 42` for reproducibility.

## Features

- **Stratified** train/val/test splits and **dataset caching** in `data/processed/`.
- **Early stopping** (LSTM) and **best-model checkpointing** for all models.
- **Class weighting** for imbalanced labels (baseline and LSTM).
- **Structured logging** and **automatic experiment naming** for checkpoints and metrics.
- **Reproducibility**: `set_seed()` and config-driven hyperparameters.

## Tests

```bash
pytest tests/ -v
```

## License

MIT.
