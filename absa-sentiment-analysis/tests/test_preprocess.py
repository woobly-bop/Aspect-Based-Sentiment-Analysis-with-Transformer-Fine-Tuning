"""
Tests for src.preprocess: clean_text, encode_labels, format_bert_input.
"""

import sys
from pathlib import Path

import pytest

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text, encode_labels, format_bert_input, decode_labels, LABEL2ID, ID2LABEL


class TestCleanText:
    """Tests for clean_text()."""

    def test_lowercase(self):
        assert clean_text("HELLO World") == "hello world"

    def test_strip(self):
        assert clean_text("  foo  ") == "foo"

    def test_remove_html(self):
        assert "<script>alert(1)</script>hi" not in clean_text("<script>alert(1)</script>hi")
        assert "hi" in clean_text("<b>hi</b>")

    def test_collapse_whitespace(self):
        assert clean_text("a   b\n\tc") == "a b c"

    def test_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_url_removal(self):
        out = clean_text("see https://example.com and www.foo.org")
        assert "https://" not in out and "www." not in out


class TestEncodeLabels:
    """Tests for encode_labels()."""

    def test_positive(self):
        assert encode_labels(["positive"]) == [0]
        assert encode_labels(["Positive"]) == [0]

    def test_negative(self):
        assert encode_labels(["negative"]) == [1]

    def test_neutral(self):
        assert encode_labels(["neutral"]) == [2]

    def test_conflict(self):
        assert encode_labels(["conflict"]) == [3]

    def test_multiple(self):
        assert encode_labels(["positive", "negative", "neutral", "conflict"]) == [0, 1, 2, 3]

    def test_unknown_maps_to_neutral(self):
        assert encode_labels(["unknown_label"]) == [2]

    def test_label2id_consistency(self):
        for name, idx in LABEL2ID.items():
            assert encode_labels([name]) == [idx]


class TestDecodeLabels:
    """Tests for decode_labels()."""

    def test_single(self):
        assert decode_labels(0) == "positive"
        assert decode_labels(1) == "negative"
        assert decode_labels(2) == "neutral"
        assert decode_labels(3) == "conflict"

    def test_list(self):
        assert decode_labels([0, 1, 2]) == ["positive", "negative", "neutral"]


class TestFormatBertInput:
    """Tests for format_bert_input()."""

    def test_basic(self):
        assert "[SEP]" in format_bert_input("The food was great.", "food")
        assert "food" in format_bert_input("The food was great.", "food")
        assert "The food was great." in format_bert_input("The food was great.", "food")

    def test_null_aspect(self):
        out = format_bert_input("Hello world.", "[NULL]")
        assert out == "Hello world."
        out2 = format_bert_input("Hello world.", "")
        assert out2 == "Hello world."

    def test_custom_sep(self):
        out = format_bert_input("s", "a", sep_token=" | ")
        assert " | " in out
        assert out.endswith("a") or "a" in out
