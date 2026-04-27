"""Centralized config: paths, hyperparameters, model defaults."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
for _d in (RAW_DIR, PROC_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
# Champa et al. (2024) curated phishing email dataset on figshare.
FIGSHARE_RECORD = "24899952"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_RECORD}"

# Possible column names across the curated CSVs (we normalize on load).
TEXT_COLUMNS = ("body", "Email Text", "text", "message", "content", "Body")
LABEL_COLUMNS = ("label", "Email Type", "class", "is_phishing", "Label")
SUBJECT_COLUMNS = ("subject", "Subject", "subject_line")


@dataclass
class TrainConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    # data
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    # augmentation
    augment_train: bool = False
    augment_factor: int = 1     # how many augmented copies per phishing email
    aug_ops: tuple = field(
        default_factory=lambda: (
            "mask_urls",
            "remove_urgency",
            "rewrite_subject",
            "synonym_swap",
        )
    )
    # quick smoke-test mode
    quick: bool = False
    quick_n: int = 400  # total samples used in quick mode
    # output
    output_dir: str = "results"
    fp16: bool = True


# Phishing-cue lexicons used by augmentation + diagnostics.
URGENCY_KEYWORDS = (
    "urgent", "immediately", "asap", "right away", "act now", "verify now",
    "limited time", "expires today", "final notice", "last chance",
    "important", "alert", "warning", "attention", "critical",
    "verify your account", "confirm your identity", "click here",
    "suspended", "deactivated", "compromised", "unusual activity",
)

# Regex for URL detection used in augmentation + URL-strip baseline.
URL_REGEX = (
    r"(https?://\S+|www\.\S+|[a-z0-9-]+\.[a-z]{2,}/\S*|"
    r"[A-Za-z0-9-]+\.(?:com|net|org|io|co|info|biz|me)\S*)"
)
