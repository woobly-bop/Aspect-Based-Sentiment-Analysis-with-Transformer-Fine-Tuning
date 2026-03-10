"""
Baseline ABSA model: TF-IDF features + Logistic Regression or SVM.
"""

import joblib
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text


class BaselineModel:
    """
    Baseline classifier: TF-IDF(sentence + aspect) -> LogisticRegression or SVC.
    """

    def __init__(
        self,
        classifier: str = "logistic",
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        """
        Args:
            classifier: "logistic" or "svm".
            max_features: Max vocabulary size for TF-IDF.
            ngram_range: (min_n, max_n) for character/word n-grams (we use word).
            C: Regularization strength (LR/SVM).
            class_weight: "balanced" for imbalanced classes, or None.
            random_state: Random seed.
        """
        self.classifier_type = classifier.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        if self.classifier_type == "svm":
            clf = SVC(
                C=self.C,
                kernel="linear",
                class_weight=self.class_weight,
                random_state=self.random_state,
                probability=False,
            )
        else:
            clf = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
            )
        self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])

    def _prepare_texts(self, sentences: List[str], aspects: Optional[List[str]] = None) -> List[str]:
        """Concatenate sentence + aspect and clean."""
        if aspects is None:
            aspects = [""] * len(sentences)
        texts = [
            clean_text(str(s) + " " + str(a))
            for s, a in zip(sentences, aspects)
        ]
        return texts

    def fit(
        self,
        sentences: List[str],
        aspects: List[str],
        labels: Union[List[int], np.ndarray],
    ) -> "BaselineModel":
        """
        Train the baseline model.

        Args:
            sentences: List of sentence strings.
            aspects: List of aspect terms.
            labels: Integer labels (0=positive, 1=negative, 2=neutral, 3=conflict).

        Returns:
            self
        """
        X = self._prepare_texts(sentences, aspects)
        y = np.asarray(labels, dtype=np.int64)
        self.pipeline.fit(X, y)
        return self

    def predict(
        self,
        sentences: List[str],
        aspects: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Predict polarity labels.

        Args:
            sentences: List of sentence strings.
            aspects: List of aspect terms (optional).

        Returns:
            Array of integer predictions.
        """
        X = self._prepare_texts(sentences, aspects)
        return self.pipeline.predict(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save pipeline (TF-IDF + classifier) with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineModel":
        """
        Load a saved BaselineModel (pipeline only; config inferred).

        Returns an instance with the loaded pipeline. Config args are defaults.
        """
        path = Path(path)
        pipeline = joblib.load(path)
        obj = cls()
        obj.pipeline = pipeline
        return obj
