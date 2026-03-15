"""Tests for SemEval CSV loading and splitting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import parse_semeval_csv, load_and_split


def test_parse_semeval_csv_header(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "id,Sentence,Aspect Term,polarity,from,to\n"
        "1,Great food.,food,positive,0,4\n",
        encoding="utf-8",
    )
    df = parse_semeval_csv(str(p))
    assert list(df.columns) == ["sentence_id", "sentence", "aspect_term", "polarity"]
    assert df.iloc[0]["sentence"] == "Great food."
    assert df.iloc[0]["aspect_term"] == "food"
    assert df.iloc[0]["polarity"] == "positive"


def test_load_and_split_stratify_or_shuffle():
    df = pd.DataFrame({
        "sentence_id": [str(i) for i in range(100)],
        "sentence": ["x"] * 100,
        "aspect_term": ["a"] * 100,
        "polarity": ["positive"] * 50 + ["negative"] * 50,
    })
    tr, va, te = load_and_split(df, test_size=0.2, val_size=0.1, random_state=0)
    assert len(tr) + len(va) + len(te) == 100
    assert set(tr.columns) == set(df.columns)
