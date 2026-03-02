"""
Preprocessing utilities for ABSA: text cleaning, BERT formatting, label encoding.
"""

import re
from typing import Dict, List, Union

import pandas as pd

# Label mapping: polarity string -> integer
LABEL2ID: Dict[str, int] = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "conflict": 3,
}

# Reverse mapping: integer -> polarity string
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def clean_text(text: str) -> str:
    """
    Clean raw text: lowercase, remove HTML, collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower().strip()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_bert_input(sentence: str, aspect: str, sep_token: str = " [SEP] ") -> str:
    """
    Format (sentence, aspect) for BERT-style input: sentence + [SEP] + aspect.

    Args:
        sentence: Full sentence text.
        aspect: Aspect term.
        sep_token: Token to separate sentence and aspect.

    Returns:
        Single string: "sentence [SEP] aspect"
    """
    sentence = (sentence or "").strip()
    aspect = (aspect or "").strip()
    if not aspect or aspect.lower() == "[null]":
        return sentence
    return sentence + sep_token + aspect


def encode_labels(labels: Union[List[str], pd.Series]) -> List[int]:
    """
    Map polarity strings to integer IDs.

    Mapping: positive->0, negative->1, neutral->2, conflict->3.
    Unknown labels are mapped to 2 (neutral).

    Args:
        labels: Iterable of polarity strings.

    Returns:
        List of integer labels.
    """
    if isinstance(labels, pd.Series):
        labels = labels.tolist()
    out = []
    for lab in labels:
        lab = (lab or "neutral").strip().lower()
        out.append(LABEL2ID.get(lab, 2))
    return out


def decode_labels(ids: Union[List[int], int]) -> Union[List[str], str]:
    """
    Map integer IDs back to polarity strings.

    Args:
        ids: Integer label(s).

    Returns:
        Polarity string(s).
    """
    if isinstance(ids, int):
        return ID2LABEL.get(ids, "neutral")
    return [ID2LABEL.get(i, "neutral") for i in ids]
