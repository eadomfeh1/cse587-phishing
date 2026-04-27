"""Load the Champa et al. curated phishing dataset, normalize columns, split."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    LABEL_COLUMNS,
    PROC_DIR,
    RAW_DIR,
    SUBJECT_COLUMNS,
    TEXT_COLUMNS,
    TrainConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------
def _first_match(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _normalize_label(series: pd.Series) -> pd.Series:
    """Map heterogeneous label spellings to {0,1}."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(int) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    phishing_tokens = {
        "1", "phishing", "phish", "spam", "malicious", "fraud",
        "fraudulent", "phishing email",
    }
    benign_tokens = {
        "0", "benign", "ham", "legit", "legitimate", "safe", "non-phishing",
        "not phishing", "safe email",
    }
    out = pd.Series(np.full(len(s), -1, dtype=int), index=s.index)
    out[s.isin(phishing_tokens)] = 1
    out[s.isin(benign_tokens)] = 0
    if (out == -1).any():
        # fallback: anything containing "phish" or "spam" -> 1
        out[(out == -1) & s.str.contains("phish|spam|malicious", na=False)] = 1
        out[out == -1] = 0
    return out


def load_raw_csvs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate every CSV under data/raw into one normalized dataframe."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs found under {raw_dir}. "
            "Run: python scripts/download_data.py"
        )
    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path, on_bad_lines="skip", encoding_errors="replace")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to read %s: %s", path, e)
            continue
        text_col = _first_match(df.columns, TEXT_COLUMNS)
        label_col = _first_match(df.columns, LABEL_COLUMNS)
        if text_col is None or label_col is None:
            logger.warning(
                "Skipping %s — could not find text/label columns. Cols=%s",
                path.name, list(df.columns),
            )
            continue
        subj_col = _first_match(df.columns, SUBJECT_COLUMNS)
        out = pd.DataFrame({
            "subject": df[subj_col].fillna("") if subj_col else "",
            "body": df[text_col].fillna(""),
            "label": _normalize_label(df[label_col]),
            "source": path.stem,
        })
        frames.append(out)

    if not frames:
        raise RuntimeError("No usable CSVs after column normalization.")
    full = pd.concat(frames, ignore_index=True)
    # Basic cleaning
    full["subject"] = full["subject"].astype(str).str.strip()
    full["body"] = full["body"].astype(str).str.strip()
    full = full[(full["body"].str.len() > 0)]
    full = full.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    logger.info(
        "Loaded %d rows from %d CSVs (phishing=%d, benign=%d)",
        len(full), len(frames), int((full.label == 1).sum()), int((full.label == 0).sum()),
    )
    return full


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def make_text_field(subject: str, body: str) -> str:
    """Combine subject + body for the transformer input."""
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject:
        return f"Subject: {subject}\n{body}"
    return body


def stratified_split(df: pd.DataFrame, cfg: TrainConfig) -> dict:
    """Stratified train/val/test split."""
    df = df.copy()
    df["text"] = [make_text_field(s, b) for s, b in zip(df.subject, df.body)]
    train_df, temp_df = train_test_split(
        df, test_size=(1 - cfg.train_frac), stratify=df.label,
        random_state=cfg.seed,
    )
    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df.label,
        random_state=cfg.seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: dict, out_dir: Path = PROC_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)


def load_splits(out_dir: Path = PROC_DIR) -> dict:
    return {n: pd.read_parquet(out_dir / f"{n}.parquet") for n in ("train", "val", "test")}


def quick_subset(df: pd.DataFrame, n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Smoke-test subset: balanced, capped at n total rows."""
    pos = df[df.label == 1].sample(min(n // 2, (df.label == 1).sum()), random_state=seed)
    neg = df[df.label == 0].sample(min(n // 2, (df.label == 0).sum()), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
