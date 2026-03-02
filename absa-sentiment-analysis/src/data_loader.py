"""
Data loader for Aspect-Based Sentiment Analysis.

Parses SemEval-2014/2015/2016 style XML and provides train/val/test splits.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_semeval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common SemEval CSV headers to sentence_id, sentence, aspect_term, polarity."""
    col_map = {c.strip().lower().replace(" ", "_"): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    sid = pick("id", "sentence_id")
    sent = pick("sentence", "text")
    asp = pick("aspect_term", "aspect")
    pol = pick("polarity", "label")
    if sid is None or sent is None or asp is None or pol is None:
        raise ValueError(
            "CSV must contain id/sentence_id, Sentence/text, Aspect Term/aspect_term, polarity columns. "
            f"Found columns: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "sentence_id": df[sid].astype(str),
        "sentence": df[sent].astype(str).fillna("").str.strip(),
        "aspect_term": df[asp].astype(str).fillna("").str.strip(),
        "polarity": df[pol].astype(str).fillna("neutral").str.strip().str.lower(),
    })
    return out


def parse_semeval_csv(filepath: str) -> pd.DataFrame:
    """
    Load SemEval-2014 style aspect CSV (tabular release).

    Expected columns include: id, Sentence, Aspect Term, polarity (and optional from, to).

    Returns:
        DataFrame with columns: sentence_id, sentence, aspect_term, polarity
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8-sig", on_bad_lines="skip")
    if df.empty:
        return pd.DataFrame(columns=["sentence_id", "sentence", "aspect_term", "polarity"])
    return _normalize_semeval_columns(df)


def load_csv_train_files(raw_dir: Union[str, Path], filenames: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple SemEval train CSVs from data/raw.

    Args:
        raw_dir: Directory containing the CSV files (e.g. data/raw).
        filenames: Basenames like restaurants_train.csv, laptops_train.csv.

    Returns:
        Combined DataFrame in canonical column format.
    """
    raw_dir = Path(raw_dir)
    frames: List[pd.DataFrame] = []
    for name in filenames:
        path = raw_dir / name
        if not path.exists():
            continue
        frames.append(parse_semeval_csv(str(path)))
    if not frames:
        return pd.DataFrame(columns=["sentence_id", "sentence", "aspect_term", "polarity"])
    return pd.concat(frames, ignore_index=True)


def parse_semeval_xml(filepath: str) -> pd.DataFrame:
    """
    Parse SemEval ABSA XML format into a pandas DataFrame.

    Expected XML structure (SemEval-2014 Restaurant/Laptop style):
    <Reviews>
      <Review>
        <sentences>
          <sentence id="...">
            <text>sentence text</text>
            <Opinions>
              <Opinion target="aspect_term" polarity="positive|negative|neutral|conflict"/>
            </Opinions>
          </sentence>
        </sentences>
      </Review>
    </Reviews>

    Returns:
        DataFrame with columns: sentence_id, sentence, aspect_term, polarity
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XML file not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    rows: list[dict] = []
    ns = {}  # handle optional namespace
    for rev in root.findall(".//Review", ns) or root.findall("Review", ns):
        for sent in rev.findall(".//sentence", ns) or rev.findall("sentences/sentence", ns):
            sent_id = sent.get("id", "")
            text_elem = sent.find("text", ns)
            text = (text_elem.text or "").strip() if text_elem is not None else ""
            opinions = sent.find("Opinions", ns) or sent.find(".//Opinions", ns)
            if opinions is None:
                continue
            for op in opinions.findall("Opinion", ns):
                target = (op.get("target") or op.get("aspect_term") or "").strip()
                polarity = (op.get("polarity") or "neutral").strip().lower()
                if not target:
                    target = "[NULL]"
                rows.append({
                    "sentence_id": sent_id,
                    "sentence": text,
                    "aspect_term": target,
                    "polarity": polarity,
                })

    # Alternative: flat structure <sentence id=""><text></text><Opinions>...
    if not rows:
        for sent in root.findall(".//sentence", ns) or root.findall("sentence", ns):
            sent_id = sent.get("id", "")
            text_elem = sent.find("text", ns)
            text = (text_elem.text or "").strip() if text_elem is not None else ""
            opinions = sent.find("Opinions", ns)
            if opinions is None:
                continue
            for op in opinions.findall("Opinion", ns):
                target = (op.get("target") or op.get("aspect_term") or "").strip()
                polarity = (op.get("polarity") or "neutral").strip().lower()
                if not target:
                    target = "[NULL]"
                rows.append({
                    "sentence_id": sent_id,
                    "sentence": text,
                    "aspect_term": target,
                    "polarity": polarity,
                })

    if not rows:
        return pd.DataFrame(columns=["sentence_id", "sentence", "aspect_term", "polarity"])

    return pd.DataFrame(rows)


def load_and_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets using stratified split.

    Args:
        df: DataFrame with at least 'polarity' column for stratification.
        test_size: Fraction for test set (0-1).
        val_size: Fraction for validation set (0-1), from the remainder after test.
        random_state: Random seed for reproducibility.
        output_dir: If set, save train/val/test CSVs to data/processed/.

    Returns:
        (train_df, val_df, test_df)
    """
    if df.empty or "polarity" not in df.columns:
        raise ValueError("DataFrame must be non-empty and contain 'polarity' column")

    def _split_stratify_safe(
        x: pd.DataFrame,
        y: pd.Series,
        test_sz: float,
        rs: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratify when each class has enough samples; otherwise shuffle split."""
        counts = y.value_counts()
        min_class = counts.min()
        use_strat = min_class >= 2 and (counts >= 2).all()
        try:
            if use_strat:
                return train_test_split(
                    x, test_size=test_sz, stratify=y, random_state=rs)
        except ValueError:
            pass
        return train_test_split(x, test_size=test_sz, shuffle=True, random_state=rs)

    # First split: train+val vs test
    train_val, test_df = _split_stratify_safe(df, df["polarity"], test_size, random_state)
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = _split_stratify_safe(
        train_val, train_val["polarity"], val_ratio, random_state,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, val_df, test_df


def load_processed_splits(
    processed_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-saved train/val/test CSV splits from data/processed/.

    Args:
        processed_dir: Path to directory containing train.csv, val.csv, test.csv.

    Returns:
        (train_df, val_df, test_df)
    """
    processed_dir = Path(processed_dir)
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    return train_df, val_df, test_df
