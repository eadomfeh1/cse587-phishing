"""Tiny end-to-end training sanity test.

Uses synthetic data so it runs without network/large dataset and verifies that
the full pipeline (split -> tokenize -> forward -> backward -> save -> reload
-> eval) works on CPU with distilbert. Intended as a smoke test, NOT a real
result.

Run:
    python scripts/tiny_train_test.py
"""
from __future__ import annotations
import logging
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation import AugmentationConfig, augment_dataframe, make_smart_attacker_eval  # noqa: E402
from src.config import TrainConfig  # noqa: E402
from src.data_loader import make_text_field, stratified_split  # noqa: E402
from src.evaluate import metrics_from  # noqa: E402
from src.model import EmailDataset, load_model_and_tokenizer  # noqa: E402
from src.utils import set_seed  # noqa: E402
from scripts.sanity_check import make_synthetic_data  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("tiny")


def run():
    set_seed(42)
    # Tiny CI-style model for the smoke test. The real Colab runs use BERT/DistilBERT.
    # See: https://huggingface.co/hf-internal-testing/tiny-random-bert
    cfg = TrainConfig(
        model_name="hf-internal-testing/tiny-random-bert",
        max_length=64,
        batch_size=8,
        eval_batch_size=8,
        epochs=1,
        lr=5e-4,
        fp16=False,
    )
    df = make_synthetic_data(n_per_class=40)
    log.info("Synthetic data: %d rows | balance=%s",
             len(df), df.label.value_counts().to_dict())

    splits = stratified_split(df, cfg)
    aug_cfg = AugmentationConfig(
        ops=("mask_urls", "remove_urgency", "rewrite_subject"), factor=1
    )
    splits["train"] = augment_dataframe(splits["train"], aug_cfg, only_phishing=True)
    log.info("Augmented train size: %d", len(splits["train"]))

    log.info("Loading distilbert (this will download ~250MB on first run) ...")
    model, tok = load_model_and_tokenizer(cfg.model_name)
    device = torch.device("cpu")
    model.to(device)

    train_ds = EmailDataset(splits["train"], tok, cfg.max_length)
    val_ds = EmailDataset(splits["val"], tok, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    log.info("Training: %d steps over %d epochs", len(train_loader), cfg.epochs)
    model.train()
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 2 == 0:
            log.info("  step %d/%d loss=%.4f", step, len(train_loader), out.loss.item())

    # Evaluate on standard test
    test_df = splits["test"].copy()
    test_df["text"] = [make_text_field(s, b) for s, b in zip(test_df.subject, test_df.body)]
    test_ds = EmailDataset(test_df, tok, cfg.max_length)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size)
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            o = model(**batch)
            logits_list.append(o.logits.cpu().numpy())
            y_list.append(labels.numpy())
    import numpy as np
    std_metrics = metrics_from(np.concatenate(logits_list), np.concatenate(y_list))

    # Smart-attacker eval
    smart_df = make_smart_attacker_eval(test_df, aug_cfg)
    smart_ds = EmailDataset(smart_df, tok, cfg.max_length)
    smart_loader = DataLoader(smart_ds, batch_size=cfg.eval_batch_size)
    smart_logits, smart_y = [], []
    with torch.no_grad():
        for batch in smart_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            o = model(**batch)
            smart_logits.append(o.logits.cpu().numpy())
            smart_y.append(labels.numpy())
    sm_metrics = metrics_from(np.concatenate(smart_logits), np.concatenate(smart_y))

    log.info("Standard      : acc=%.4f f1=%.4f",
             std_metrics["accuracy"], std_metrics["f1_phish"])
    log.info("Smart-attacker: acc=%.4f f1=%.4f",
             sm_metrics["accuracy"], sm_metrics["f1_phish"])

    # Save + reload checkpoint
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        tok.save_pretrained(td)
        rel_model, rel_tok = load_model_and_tokenizer(td)
        log.info("Checkpoint round-trip OK (params=%d)",
                 sum(p.numel() for p in rel_model.parameters()))

    log.info("Tiny training experiment finished in %.1fs", time.time() - t0)
    return {"standard": std_metrics, "smart_attacker": sm_metrics}


if __name__ == "__main__":
    print(run())
