"""
Bi-LSTM ABSA model with GloVe embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.preprocess import clean_text


class ABSADataset(Dataset):
    """PyTorch Dataset for (sentence, aspect) -> label with token indices."""

    def __init__(
        self,
        sentences: List[str],
        aspects: List[str],
        labels: Union[List[int], np.ndarray],
        word2idx: Dict[str, int],
        max_len: int = 128,
    ):
        self.sentences = sentences
        self.aspects = aspects
        self.labels = np.asarray(labels, dtype=np.int64)
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_idx = word2idx.get("<pad>", 0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sent = self.sentences[idx]
        asp = self.aspects[idx]
        # Input: sentence + " " + aspect (cleaned)
        text = clean_text(str(sent) + " " + str(asp))
        tokens = text.split()[: self.max_len]
        ids = [self.word2idx.get(t, self.word2idx.get("<unk>", 1)) for t in tokens]
        if len(ids) < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class BiLSTMClassifier(nn.Module):
    """
    Bi-LSTM classifier: Embedding -> Bi-LSTM -> Dropout -> Linear -> logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.5,
        padding_idx: int = 0,
        embedding_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = True  # fine-tune
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices.

        Returns:
            logits: (batch, num_classes)
        """
        emb = self.embedding(x)
        out, (h_n, _) = self.lstm(emb)
        # Use last time step from both directions
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        hidden = torch.cat([forward_last, backward_last], dim=1)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


def build_vocab_and_weights(
    sentences: List[str],
    aspects: List[str],
    glove_path: Optional[str] = None,
    embedding_dim: int = 100,
) -> Tuple[Dict[str, int], Optional[torch.Tensor]]:
    """
    Build word2idx from corpus and optionally load GloVe weights.

    vocab: <pad>=0, <unk>=1, then all words from corpus (and GloVe if provided).
    """
    from collections import Counter
    words: set = set()
    for s, a in zip(sentences, aspects):
        text = clean_text(str(s) + " " + str(a))
        words.update(text.split())
    word2idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for w in sorted(words):
        if w not in word2idx:
            word2idx[w] = len(word2idx)

    if glove_path and Path(glove_path).exists():
        # Load GloVe (space-separated: word dim1 dim2 ...)
        embedding_weights = np.random.randn(len(word2idx), embedding_dim).astype(np.float32) * 0.01
        with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word, vec = parts[0], np.array(parts[1:], dtype=np.float32)
                if word in word2idx:
                    embedding_weights[word2idx[word]] = vec
        embedding_weights[0] = 0.0  # pad
        return word2idx, torch.tensor(embedding_weights, dtype=torch.float32)
    return word2idx, None
