"""Training entrypoint.

Usage
-----
    python -m src.train                     # baseline BERT, full data
    python -m src.train --quick             # smoke test (tiny subset, distilbert)
    python -m src.train --augment           # augmented training run
    python -m src.train --model distilbert-base-uncased --epochs 2
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .augmentation import AugmentationConfig, augment_dataframe
from .config import PROC_DIR, RESULTS_DIR, TrainConfig
from .data_loader import (
    load_raw_csvs,
    quick_subset,
    save_splits,
    stratified_split,
)
from .evaluate import evaluate_checkpoint, render_markdown_report
from .model import EmailDataset, load_model_and_tokenizer
from .utils import dump_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true",
                   help="Augment phishing rows in training set")
    p.add_argument("--augment_factor", type=int, default=1)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: tiny subset, distilbert, 1 epoch")
    p.add_argument("--output_dir", default="results")
    args = p.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        augment_train=args.augment,
        augment_factor=args.augment_factor,
        quick=args.quick,
        output_dir=args.output_dir,
    )
    if cfg.quick:
        cfg.model_name = "distilbert-base-uncased"
        cfg.epochs = 1
        cfg.batch_size = 8
        cfg.max_length = 128
    return cfg


def prepare_splits(cfg: TrainConfig) -> dict:
    """Build (or reload) train/val/test splits."""
    tr = PROC_DIR / "train.parquet"
    if tr.exists() and not cfg.quick:
        logger.info("Reusing cached splits in %s", PROC_DIR)
        from .data_loader import load_splits
        return load_splits()
    df = load_raw_csvs()
    if cfg.quick:
        df = quick_subset(df, n=cfg.quick_n, seed=cfg.seed)
    splits = stratified_split(df, cfg)
    if not cfg.quick:
        save_splits(splits)
    return splits


def maybe_augment(splits: dict, cfg: TrainConfig) -> dict:
    if not cfg.augment_train:
        return splits
    aug_cfg = AugmentationConfig(
        ops=tuple(cfg.aug_ops),
        factor=cfg.augment_factor,
        seed=cfg.seed,
    )
    splits = dict(splits)
    n_before = len(splits["train"])
    splits["train"] = augment_dataframe(splits["train"], aug_cfg, only_phishing=True)
    logger.info(
        "Augmented training set: %d -> %d rows", n_before, len(splits["train"])
    )
    return splits


def train_loop(cfg: TrainConfig) -> Path:
    setup_logging()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | model: %s | epochs=%d | bs=%d | lr=%g",
                device, cfg.model_name, cfg.epochs, cfg.batch_size, cfg.lr)

    splits = prepare_splits(cfg)
    splits = maybe_augment(splits, cfg)

    model, tok = load_model_and_tokenizer(cfg.model_name)
    model.to(device)

    train_ds = EmailDataset(splits["train"], tok, cfg.max_length)
    val_ds = EmailDataset(splits["val"], tok, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    # Mixed precision: prefer bfloat16 on A100/H100 (no GradScaler needed),
    # fall back to fp16 + GradScaler on T4/V100/older GPUs.
    use_amp = cfg.fp16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (
        use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if use_amp:
        logger.info("Mixed precision enabled: %s",
                    "bfloat16" if amp_dtype == torch.bfloat16 else "float16")

    best_val = -float("inf")
    out_dir = Path(cfg.output_dir) / (
        "augmented" if cfg.augment_train else "baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                logger.info("epoch %d step %d/%d loss=%.4f",
                            epoch, step, len(train_loader), loss.item())
        epoch_loss = running / max(1, len(train_loader))

        # Quick val
        val_metrics = _eval_dataloader(model, val_loader, device)
        logger.info(
            "epoch %d done: train_loss=%.4f val_acc=%.4f val_f1=%.4f (%.1fs)",
            epoch, epoch_loss, val_metrics["accuracy"], val_metrics["f1_phish"],
            time.time() - t0,
        )
        if val_metrics["f1_phish"] > best_val:
            best_val = val_metrics["f1_phish"]
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            dump_json(val_metrics, out_dir / "val_metrics.json")
            logger.info("Saved best checkpoint to %s (val F1=%.4f)",
                        out_dir, best_val)

    return out_dir


@torch.no_grad()
def _eval_dataloader(model, loader, device) -> dict:
    from .evaluate import metrics_from
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.detach().cpu().numpy())
        all_y.append(labels.numpy())
    return metrics_from(np.concatenate(all_logits), np.concatenate(all_y))


def main():
    cfg = parse_args()
    out_dir = train_loop(cfg)
    # Always evaluate at the end
    rep = evaluate_checkpoint(out_dir, cfg)
    md = render_markdown_report(rep)
    rep_path = Path(RESULTS_DIR) / (
        "report_augmented.md" if cfg.augment_train else "report_baseline.md"
    )
    rep_path.write_text(md)
    print(md)
    print(f"\nReport saved to {rep_path}")


if __name__ == "__main__":
    main()
