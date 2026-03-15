"""
Unified training pipeline for baseline, LSTM, and BERT ABSA models.

Usage:
  python train.py --model baseline [--config configs/baseline_config.yaml]
  python train.py --model lstm [--config configs/lstm_config.yaml]
  python train.py --model bert [--config configs/bert_config.yaml]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import (
    load_processed_splits,
    load_csv_train_files,
    parse_semeval_xml,
    load_and_split,
)
from src.preprocess import clean_text, encode_labels, format_bert_input, LABEL2ID
from src.utils import set_seed, setup_logging, get_device, get_experiment_name
from src.evaluate import compute_all_metrics, plot_confusion_matrix, save_metrics_table, plot_training_curves
from src.models.baseline import BaselineModel
from src.models.lstm_model import ABSADataset, BiLSTMClassifier, build_vocab_and_weights
from src.models.bert_model import (
    get_bert_model_and_tokenizer,
    prepare_bert_dataset,
    create_bert_trainer,
)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_data(config: dict, logger, rebuild_data: bool = False):
    """Load or create train/val/test splits; return DataFrames and encoded labels."""
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    data_dir = Path(config.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    raw_xml = config.get("raw_xml")
    test_size = config.get("test_size", 0.1)
    val_size = config.get("val_size", 0.1)
    seed = config.get("seed", 42)
    rebuild = rebuild_data or bool(config.get("rebuild_processed", False))
    csv_train_files = config.get("csv_train_files") or []
    raw_dir = data_dir / "raw"

    def _dummy_splits() -> tuple:
        logger.warning("Creating minimal dummy data (no CSV/XML/cache).")
        tr = pd.DataFrame({
            "sentence_id": ["1", "2", "3"] * 20,
            "sentence": ["Great food and service."] * 60,
            "aspect_term": ["food", "service", "ambiance"] * 20,
            "polarity": ["positive", "negative", "neutral"] * 20,
        })
        va = tr.iloc[:10].copy()
        te = tr.iloc[10:20].copy()
        processed_dir.mkdir(parents=True, exist_ok=True)
        tr.to_csv(processed_dir / "train.csv", index=False)
        va.to_csv(processed_dir / "val.csv", index=False)
        te.to_csv(processed_dir / "test.csv", index=False)
        return tr, va, te

    cache_ok = (processed_dir / "train.csv").exists() and not rebuild
    if cache_ok:
        logger.info("Loading cached splits from %s", processed_dir)
        train_df, val_df, test_df = load_processed_splits(str(processed_dir))
    else:
        train_df, val_df, test_df = None, None, None
        if csv_train_files:
            df_csv = load_csv_train_files(raw_dir, list(csv_train_files))
            if not df_csv.empty:
                logger.info(
                    "Built dataset from csv_train_files (%d rows) -> %s",
                    len(df_csv),
                    processed_dir,
                )
                train_df, val_df, test_df = load_and_split(
                    df_csv,
                    test_size=test_size,
                    val_size=val_size,
                    random_state=seed,
                    output_dir=str(processed_dir),
                )
        if train_df is None and raw_xml:
            xml_path = raw_dir / raw_xml
            if xml_path.exists():
                df = parse_semeval_xml(str(xml_path))
                train_df, val_df, test_df = load_and_split(
                    df,
                    test_size=test_size,
                    val_size=val_size,
                    random_state=seed,
                    output_dir=str(processed_dir),
                )
            else:
                logger.warning("XML not found at %s", xml_path)
        if train_df is None:
            train_df, val_df, test_df = _dummy_splits()

    train_df["label"] = encode_labels(train_df["polarity"])
    val_df["label"] = encode_labels(val_df["polarity"])
    test_df["label"] = encode_labels(test_df["polarity"])
    return train_df, val_df, test_df


def train_baseline(config: dict, train_df, val_df, test_df, logger):
    """Train baseline model; save checkpoint and metrics."""
    seed = config.get("seed", 42)
    set_seed(seed)
    exp_name = get_experiment_name("baseline", seed)
    checkpoint_dir = Path(config.get("checkpoint_dir", "results/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config.get("metrics_dir", "results/metrics"))
    figures_dir = Path(config.get("figures_dir", "results/figures"))

    model = BaselineModel(
        classifier=config.get("classifier", "logistic"),
        max_features=config.get("max_features", 10000),
        ngram_range=tuple(config.get("ngram_range", [1, 2])),
        C=config.get("C", 1.0),
        class_weight=config.get("class_weight", "balanced"),
        random_state=seed,
    )
    logger.info("Training baseline (%s)...", model.classifier_type)
    model.fit(
        train_df["sentence"].tolist(),
        train_df["aspect_term"].tolist(),
        train_df["label"].tolist(),
    )
    ckpt_path = checkpoint_dir / f"baseline_{exp_name}.joblib"
    model.save(ckpt_path)
    logger.info("Saved baseline to %s", ckpt_path)

    preds = model.predict(test_df["sentence"].tolist(), test_df["aspect_term"].tolist())
    metrics = compute_all_metrics(test_df["label"].tolist(), preds)
    metrics["model"] = "baseline"
    metrics["experiment"] = exp_name
    plot_confusion_matrix(
        test_df["label"].tolist(), preds,
        save_path=figures_dir / f"cm_baseline_{exp_name}.png",
        title="Baseline Confusion Matrix",
    )
    save_metrics_table([metrics], metrics_dir / "all_results.csv", append=True)
    logger.info("Test metrics: %s", metrics)
    return metrics


def train_lstm(config: dict, train_df, val_df, test_df, logger):
    """Train Bi-LSTM; early stopping, checkpointing, class weights."""
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("use_cuda", True))
    exp_name = get_experiment_name("lstm", seed)
    checkpoint_dir = Path(config.get("checkpoint_dir", "results/checkpoints"))
    metrics_dir = Path(config.get("metrics_dir", "results/metrics"))
    figures_dir = Path(config.get("figures_dir", "results/figures"))
    data_dir = Path(config.get("data_dir", "data"))

    # Vocab and GloVe
    glove_path = config.get("glove_path") or (data_dir / "raw" / "glove.6B.100d.txt")
    if isinstance(glove_path, Path):
        glove_path = str(glove_path)
    word2idx, emb_weights = build_vocab_and_weights(
        train_df["sentence"].tolist(),
        train_df["aspect_term"].tolist(),
        glove_path=glove_path if Path(glove_path).exists() else None,
        embedding_dim=config.get("embedding_dim", 100),
    )
    vocab_size = len(word2idx)
    max_len = config.get("max_len", 128)
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 20)
    early_stopping_patience = config.get("early_stopping_patience", 5)
    lr = config.get("learning_rate", 1e-3)

    train_ds = ABSADataset(
        train_df["sentence"].tolist(),
        train_df["aspect_term"].tolist(),
        train_df["label"].tolist(),
        word2idx,
        max_len=max_len,
    )
    val_ds = ABSADataset(
        val_df["sentence"].tolist(),
        val_df["aspect_term"].tolist(),
        val_df["label"].tolist(),
        word2idx,
        max_len=max_len,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights
    from torch.nn import CrossEntropyLoss
    labels = train_df["label"].tolist()
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([
        1.0 / (counts.get(i, 1) / total) for i in range(4)
    ], dtype=torch.float32, device=device)
    weights = weights / weights.sum() * 4
    criterion = CrossEntropyLoss(weight=weights)

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.get("embedding_dim", 100),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        num_classes=4,
        dropout=config.get("dropout", 0.5),
        padding_idx=0,
        embedding_weights=emb_weights,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    val_f1s = []
    best_val_f1 = -1.0
    best_ckpt = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
        val_losses.append(val_loss / len(val_loader))
        val_metrics = compute_all_metrics(all_labels, all_preds)
        val_f1s.append(val_metrics["f1"])
        logger.info("Epoch %d train_loss=%.4f val_loss=%.4f val_f1=%.4f",
                    epoch + 1, train_losses[-1], val_losses[-1], val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_ckpt = checkpoint_dir / f"lstm_{exp_name}_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "word2idx": word2idx,
                "config": config,
            }, best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Load best for final eval
    if best_ckpt and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_metrics={"f1": val_f1s},
        save_path=figures_dir / f"curves_lstm_{exp_name}.png",
    )

    # Test
    test_ds = ABSADataset(
        test_df["sentence"].tolist(),
        test_df["aspect_term"].tolist(),
        test_df["label"].tolist(),
        word2idx,
        max_len=max_len,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Test"):
            x = x.to(device)
            logits = model(x)
            test_preds.extend(logits.argmax(1).cpu().numpy().tolist())
    metrics = compute_all_metrics(test_df["label"].tolist(), test_preds)
    metrics["model"] = "lstm"
    metrics["experiment"] = exp_name
    plot_confusion_matrix(
        test_df["label"].tolist(), test_preds,
        save_path=figures_dir / f"cm_lstm_{exp_name}.png",
        title="LSTM Confusion Matrix",
    )
    save_metrics_table([metrics], metrics_dir / "all_results.csv", append=True)
    logger.info("Test metrics: %s", metrics)
    return metrics


def train_bert(config: dict, train_df, val_df, test_df, logger):
    """Train BERT with Trainer; checkpointing and metrics."""
    seed = config.get("seed", 42)
    set_seed(seed)
    exp_name = get_experiment_name("bert", seed)
    checkpoint_dir = Path(config.get("checkpoint_dir", "results/checkpoints"))
    metrics_dir = Path(config.get("metrics_dir", "results/metrics"))
    figures_dir = Path(config.get("figures_dir", "results/figures"))
    output_dir = str(checkpoint_dir / f"bert_{exp_name}")

    model_name = config.get("model_name", "bert-base-uncased")
    num_labels = config.get("num_labels", 4)
    max_length = config.get("max_length", 128)
    model, tokenizer = get_bert_model_and_tokenizer(model_name, num_labels=num_labels)

    train_ds = prepare_bert_dataset(
        train_df["sentence"].tolist(),
        train_df["aspect_term"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_ds = prepare_bert_dataset(
        val_df["sentence"].tolist(),
        val_df["aspect_term"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    def compute_metrics_fn(eval_pred):
        from transformers import EvalPrediction
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        return compute_all_metrics(labels, preds)

    trainer = create_bert_trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 4),
        learning_rate=config.get("learning_rate", 2e-5),
        batch_size=config.get("batch_size", 16),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )
    trainer.compute_metrics = compute_metrics_fn

    logger.info("Training BERT...")
    trainer.train()
    trainer.save_model(str(Path(output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(output_dir) / "final"))

    # Test eval
    test_ds = prepare_bert_dataset(
        test_df["sentence"].tolist(),
        test_df["aspect_term"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    out = trainer.predict(test_ds)
    preds = np.argmax(out.predictions, axis=-1)
    metrics = compute_all_metrics(out.label_ids, preds)
    metrics["model"] = "bert"
    metrics["experiment"] = exp_name
    plot_confusion_matrix(
        out.label_ids, preds,
        save_path=figures_dir / f"cm_bert_{exp_name}.png",
        title="BERT Confusion Matrix",
    )
    save_metrics_table([metrics], metrics_dir / "all_results.csv", append=True)
    logger.info("Test metrics: %s", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ABSA model")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "lstm", "bert"],
                        help="Model type")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: configs/<model>_config.yaml)")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Ignore cached data/processed/*.csv and rebuild from csv_train_files or raw_xml",
    )
    args = parser.parse_args()

    config_path = args.config or str(PROJECT_ROOT / "configs" / f"{args.model}_config.yaml")
    if not Path(config_path).exists():
        config = {"seed": 42, "data_dir": "data", "processed_dir": "data/processed",
                  "checkpoint_dir": "results/checkpoints", "metrics_dir": "results/metrics",
                  "figures_dir": "results/figures"}
    else:
        config = load_config(config_path)
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.seed is not None:
        config["seed"] = args.seed
    config.setdefault("data_dir", str(PROJECT_ROOT / "data"))
    config.setdefault("processed_dir", str(PROJECT_ROOT / "data" / "processed"))
    config.setdefault("checkpoint_dir", str(PROJECT_ROOT / "results" / "checkpoints"))
    config.setdefault("metrics_dir", str(PROJECT_ROOT / "results" / "metrics"))
    config.setdefault("figures_dir", str(PROJECT_ROOT / "results" / "figures"))
    # Resolve relative paths in config (cwd-independent)
    for key in (
        "checkpoint_dir",
        "metrics_dir",
        "figures_dir",
        "data_dir",
        "processed_dir",
        "glove_path",
    ):
        if key in config and config[key] and not Path(str(config[key])).is_absolute():
            config[key] = str(PROJECT_ROOT / config[key])

    log_dir = Path(config["checkpoint_dir"]).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=str(log_dir / "train.log"), name="absa")

    train_df, val_df, test_df = get_data(config, logger, rebuild_data=args.rebuild_data)
    logger.info("Train %d val %d test %d", len(train_df), len(val_df), len(test_df))

    if args.model == "baseline":
        train_baseline(config, train_df, val_df, test_df, logger)
    elif args.model == "lstm":
        train_lstm(config, train_df, val_df, test_df, logger)
    elif args.model == "bert":
        train_bert(config, train_df, val_df, test_df, logger)
    else:
        raise ValueError("Unknown model: " + args.model)


if __name__ == "__main__":
    main()
