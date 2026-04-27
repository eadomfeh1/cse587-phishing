"""End-to-end sanity check that does NOT require torch/transformers.

Validates:
* Synthetic dataset loading & label normalization
* Stratified split logic
* Each augmentation op on representative phishing emails
* Smart-attacker eval transformation

Run with:
    python scripts/sanity_check.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation import (  # noqa: E402
    AugmentationConfig,
    apply_pipeline,
    augment_dataframe,
    char_perturb,
    make_smart_attacker_eval,
    mask_urls,
    remove_urgency,
    rewrite_subject,
    synonym_swap,
)
from src.config import TrainConfig  # noqa: E402
from src.data_loader import (  # noqa: E402
    _normalize_label,
    make_text_field,
    stratified_split,
)


def make_synthetic_data(n_per_class: int = 30) -> pd.DataFrame:
    phishing = [
        ("URGENT: Verification required",
         "Your Amazon account is subject to deletion unless you verify "
         "immediately at https://amaz0n-verify.example.com/login"),
        ("Action Required: Account suspended",
         "We noticed unusual activity. Click here right away: "
         "http://secure-paypal-update.tk/auth"),
        ("FINAL NOTICE: Confirm your identity",
         "Your access will be revoked. Verify now via www.bank-verify.co/u"),
    ] * n_per_class
    benign = [
        ("Meeting moved to Tuesday",
         "Hello all, the meeting is now Tuesday at the same time in the same room. Thanks"),
        ("Lunch?",
         "Want to grab lunch at noon? Let me know."),
        ("Project status update",
         "Sharing the weekly summary. Numbers look good. Talk Friday."),
    ] * n_per_class
    rows = []
    for s, b in phishing[:n_per_class]:
        rows.append({"subject": s, "body": b, "label": 1, "source": "synth"})
    for s, b in benign[:n_per_class]:
        rows.append({"subject": s, "body": b, "label": 0, "source": "synth"})
    return pd.DataFrame(rows)


def main():
    print("=" * 68)
    print("Sanity check: data + augmentation pipeline")
    print("=" * 68)

    # 1) Label normalization
    print("\n[1] Label normalization")
    raw_labels = pd.Series(["phishing", "benign", "1", "0", "Phishing Email", "ham"])
    out = _normalize_label(raw_labels)
    print("   raw   :", list(raw_labels))
    print("   normd :", list(out))
    assert list(out) == [1, 0, 1, 0, 1, 0], f"Label normalization failed: {list(out)}"
    print("   ✓ ok")

    # 2) Build synthetic data + split
    print("\n[2] Stratified split on synthetic data")
    df = make_synthetic_data(n_per_class=30)
    cfg = TrainConfig(seed=42)
    splits = stratified_split(df, cfg)
    for k, v in splits.items():
        bal = v.label.value_counts().to_dict()
        print(f"   {k:5s} n={len(v):3d} class_balance={bal}")
    assert len(splits["train"]) > len(splits["val"]) > 0
    print("   ✓ ok")

    # 3) Each augmentation op
    print("\n[3] Individual augmentation ops on a phishing example")
    sample = make_text_field(
        "URGENT: Verification required",
        "Your Amazon account is subject to deletion unless you verify "
        "immediately at https://amaz0n-verify.example.com/login. Act now!",
    )
    print("\n   ORIGINAL:")
    print("   " + sample.replace("\n", "\n   "))

    print("\n   mask_urls:")
    print("   " + mask_urls(sample))

    print("\n   remove_urgency:")
    print("   " + remove_urgency(sample))

    print("\n   rewrite_subject:")
    print("   " + rewrite_subject(sample).replace("\n", "\n   "))

    print("\n   synonym_swap (may need WordNet download):")
    try:
        print("   " + synonym_swap(sample, p=0.4, max_swaps=4))
    except Exception as e:
        print(f"   (skipped — wordnet unavailable: {e})")

    print("\n   char_perturb (p=0.05):")
    print("   " + char_perturb(sample, p=0.05))

    # 4) Full pipeline
    print("\n[4] Full augmentation pipeline")
    full = apply_pipeline(sample, ("mask_urls", "remove_urgency", "rewrite_subject"))
    print("   " + full.replace("\n", "\n   "))
    assert "URGENT" not in full and "https://" not in full and "Act now" not in full
    print("   ✓ urgency keywords + URLs removed/replaced")

    # 5) augment_dataframe
    print("\n[5] augment_dataframe (factor=2, phishing only)")
    train = splits["train"].copy()
    train["text"] = [make_text_field(s, b) for s, b in zip(train.subject, train.body)]
    aug = augment_dataframe(
        train,
        AugmentationConfig(
            ops=("mask_urls", "remove_urgency", "rewrite_subject"),
            factor=2,
        ),
    )
    print(f"   before: {len(train)} rows | after: {len(aug)} rows")
    assert len(aug) > len(train)
    aug_only = aug[aug.source.str.endswith("__aug")]
    assert (aug_only.label == 1).all(), "Augmentation should only touch phishing rows"
    print("   ✓ ok (augmented rows are all phishing)")

    # 6) make_smart_attacker_eval
    print("\n[6] make_smart_attacker_eval")
    test = splits["test"].copy()
    test["text"] = [make_text_field(s, b) for s, b in zip(test.subject, test.body)]
    smart = make_smart_attacker_eval(test, AugmentationConfig())
    # phishing rows should differ; benign rows should be identical
    diffs = (smart.text.values != test.text.values)
    is_phish = (test.label.values == 1)
    is_benign = (test.label.values == 0)
    print(f"   phishing rows changed : {int(diffs[is_phish].sum())}/{int(is_phish.sum())}")
    print(f"   benign   rows changed : {int(diffs[is_benign].sum())}/{int(is_benign.sum())}")
    assert diffs[is_phish].any()
    assert not diffs[is_benign].any()
    print("   ✓ ok (phishing perturbed, benign untouched)")

    print("\nAll sanity checks passed ✓")


if __name__ == "__main__":
    main()
