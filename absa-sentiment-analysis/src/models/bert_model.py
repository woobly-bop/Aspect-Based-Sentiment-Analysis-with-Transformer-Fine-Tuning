"""
BERT-based ABSA model: AutoModelForSequenceClassification + Trainer support.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.preprocess import format_bert_input


def get_bert_model_and_tokenizer(
    model_name: str = "bert-base-uncased",
    num_labels: int = 4,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load BERT tokenizer and sequence classification model.

    Args:
        model_name: HuggingFace model id.
        num_labels: Number of classes (4 for ABSA: pos/neg/neu/conflict).

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model, tokenizer


def prepare_bert_dataset(
    sentences: List[str],
    aspects: List[str],
    labels: List[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> HFDataset:
    """
    Build HuggingFace Dataset with tokenized inputs for BERT.

    Input format: sentence [SEP] aspect. Tokenization with truncation/padding.

    Args:
        sentences: List of sentence strings.
        aspects: List of aspect terms.
        labels: Integer labels.
        tokenizer: BERT tokenizer.
        max_length: Max sequence length.

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, labels.
    """
    texts = [
        format_bert_input(s, a)
        for s, a in zip(sentences, aspects)
    ]
    enc = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    return HFDataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": list(labels),
    })


def create_bert_trainer(
    model: PreTrainedModel,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
    output_dir: str = "results/checkpoints/bert",
    num_train_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_f1",
    greater_is_better: bool = True,
    early_stopping_patience: Optional[int] = None,
) -> Trainer:
    """
    Create HuggingFace Trainer for BERT fine-tuning.

    Args:
        model: PreTrainedModel (e.g. BERT for sequence classification).
        train_dataset: Training HF Dataset.
        eval_dataset: Validation HF Dataset (optional).
        output_dir: Checkpoint and log directory.
        num_train_epochs: Epochs.
        learning_rate: Peak LR.
        batch_size: Per-device batch size.
        warmup_ratio: Warmup fraction.
        weight_decay: Weight decay.
        seed: Random seed.
        logging_steps: Log every N steps.
        save_strategy: "epoch" or "steps".
        load_best_model_at_end: Load best checkpoint at end.
        metric_for_best_model: Metric to select best model.
        greater_is_better: Whether higher metric is better.
        early_stopping_patience: Optional; number of evals without improvement to stop.

    Returns:
        Trainer instance.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    eval_strat = "epoch" if eval_dataset else "no"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "logging_dir": str(Path(output_dir) / "logs"),
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end and eval_dataset is not None,
        "metric_for_best_model": metric_for_best_model if eval_dataset else "loss",
        "greater_is_better": greater_is_better,
        "seed": seed,
        "report_to": "none",
        "save_total_limit": 2,
    }
    # Transformers >= 5 uses eval_strategy; older uses evaluation_strategy
    if "eval_strategy" in ta_params:
        args_kwargs["eval_strategy"] = eval_strat
    else:
        args_kwargs["evaluation_strategy"] = eval_strat
    training_args = TrainingArguments(**args_kwargs)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,  # can be set externally for eval
    )
