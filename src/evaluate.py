"""Evaluation: standard test, smart-attacker test, per-edit ablation."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .augmentation import AugmentationConfig, make_smart_attacker_eval
from .config import RESULTS_DIR, TrainConfig
from .data_loader import load_splits
from .model import EmailDataset, load_model_and_tokenizer
from .utils import dump_json, setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return logits, y


def metrics_from(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1]).tolist()
    out = {
        "accuracy": float(acc),
        "precision_phish": float(p),
        "recall_phish": float(r),
        "f1_phish": float(f1),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }
    if logits.shape[1] == 2:
        try:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            out["roc_auc"] = float(roc_auc_score(y, probs))
        except Exception:
            pass
    return out


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    cfg: TrainConfig,
    aug_cfg: AugmentationConfig | None = None,
) -> Dict[str, dict]:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(str(checkpoint_dir))
    model.to(device)

    splits = load_splits()
    test = splits["test"]

    # 1) Standard
    std_ds = EmailDataset(test, tok, cfg.max_length)
    std_loader = DataLoader(std_ds, batch_size=cfg.eval_batch_size)
    std_logits, std_y = predict(model, std_loader, device)
    std_m = metrics_from(std_logits, std_y)

    # 2) Smart attacker
    if aug_cfg is None:
        aug_cfg = AugmentationConfig(ops=tuple(cfg.aug_ops))
    smart_df = make_smart_attacker_eval(test, aug_cfg)
    smart_ds = EmailDataset(smart_df, tok, cfg.max_length)
    smart_loader = DataLoader(smart_ds, batch_size=cfg.eval_batch_size)
    smart_logits, smart_y = predict(model, smart_loader, device)
    smart_m = metrics_from(smart_logits, smart_y)

    # 3) Per-edit ablation (apply each op alone to phishing rows)
    ablation = {}
    for op in aug_cfg.ops:
        single_cfg = AugmentationConfig(ops=(op,))
        df_op = make_smart_attacker_eval(test, single_cfg)
        ds_op = EmailDataset(df_op, tok, cfg.max_length)
        loader_op = DataLoader(ds_op, batch_size=cfg.eval_batch_size)
        l_op, y_op = predict(model, loader_op, device)
        ablation[op] = metrics_from(l_op, y_op)

    report = {
        "standard": std_m,
        "smart_attacker": smart_m,
        "ablation": ablation,
        "config": {
            "model_name": cfg.model_name,
            "seed": cfg.seed,
            "max_length": cfg.max_length,
            "augment_train": cfg.augment_train,
            "aug_ops": list(aug_cfg.ops),
        },
    }
    # Write the per-run report inside the checkpoint directory so each
    # (config, seed) pair has its own JSON and aggregate_seeds.py can find
    # all of them via a glob.
    out_path = Path(checkpoint_dir) / "eval_report.json"
    dump_json(report, out_path)
    # Also keep a "latest" copy at the results root for backwards-compat with
    # older notebooks/cells that read results/eval_report.json directly.
    dump_json(report, Path(RESULTS_DIR) / "eval_report.json")
    logger.info("Eval report written to %s", out_path)
    return report


def render_markdown_report(report: dict) -> str:
    lines = ["# Evaluation report\n"]
    lines.append("## Standard test\n")
    lines.append(_metrics_table(report["standard"]))
    lines.append("\n## Smart-attacker test\n")
    lines.append(_metrics_table(report["smart_attacker"]))
    delta_acc = report["smart_attacker"]["accuracy"] - report["standard"]["accuracy"]
    delta_f1 = report["smart_attacker"]["f1_phish"] - report["standard"]["f1_phish"]
    lines.append(f"\n**Δ accuracy (smart − standard):** {delta_acc:+.4f}\n")
    lines.append(f"**Δ F1(phish):** {delta_f1:+.4f}\n")
    lines.append("\n## Per-edit ablation\n")
    for op, m in report["ablation"].items():
        lines.append(f"### {op}\n")
        lines.append(_metrics_table(m))
    return "\n".join(lines)


def _metrics_table(m: dict) -> str:
    rows = [
        ("accuracy", m["accuracy"]),
        ("precision (phishing)", m["precision_phish"]),
        ("recall (phishing)", m["recall_phish"]),
        ("F1 (phishing)", m["f1_phish"]),
        ("macro F1", m["macro_f1"]),
    ]
    if "roc_auc" in m:
        rows.append(("ROC-AUC", m["roc_auc"]))
    out = ["| metric | value |", "|---|---|"]
    out += [f"| {k} | {v:.4f} |" for k, v in rows]
    cm = m.get("confusion_matrix")
    if cm:
        out.append("\nConfusion matrix (rows=true, cols=pred, label order [benign, phish]):\n")
        out.append("```")
        out.append(f"        pred=0  pred=1")
        out.append(f"true=0  {cm[0][0]:>6}  {cm[0][1]:>6}")
        out.append(f"true=1  {cm[1][0]:>6}  {cm[1][1]:>6}")
        out.append("```")
    return "\n".join(out)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--report", choices=("json", "markdown"), default="markdown")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=32)
    args = p.parse_args()

    cfg = TrainConfig(
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
    )
    rep = evaluate_checkpoint(args.checkpoint, cfg)
    if args.report == "json":
        import json
        print(json.dumps(rep, indent=2))
    else:
        print(render_markdown_report(rep))


if __name__ == "__main__":
    cli()
