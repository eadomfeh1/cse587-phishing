"""BERT classifier wrapper + tokenized dataset."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, num_labels: int = 2):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tok


class EmailDataset(Dataset):
    """In-memory tokenized email dataset.

    Expects a pandas DataFrame with columns 'text' and 'label'.
    """

    def __init__(self, df, tokenizer, max_length: int = 256):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
