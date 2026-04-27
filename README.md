# Reeling in Smart Attackers' Phishing Emails with Neural Networks

CSE 587 — Group 5
Parker Davis · Dillon VanGilder · Emmanuel Adjei Domfeh

## Overview

Binary phishing email classification (benign vs. phishing) using a fine-tuned
Transformer (BERT/DistilBERT). The contribution is **robustness to "smart
attacker" edits** — emails where obvious phishing cues (urgency keywords,
explicit URLs) have been removed or rewritten — addressed via targeted data
augmentation and adversarial-style retraining.

We follow the workflow:

```
raw dataset → train baseline → evaluate on standard + smart-attacker test
            → augment training data → retrain → re-evaluate
            → compare across 3 stages
```

## Project layout

```
cse587/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # downloaded Champa et al. CSV(s)
│   └── processed/      # train/val/test/smart-attacker splits
├── scripts/
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── config.py       # paths, hyperparameters
│   ├── data_loader.py  # load + split + tokenize
│   ├── augmentation.py # smart-attacker edits (URL mask, urgency removal, etc.)
│   ├── model.py        # PhishingClassifier wrapper
│   ├── train.py        # training loop
│   ├── evaluate.py     # metrics + confusion matrices
│   └── utils.py
├── notebooks/
│   └── phishing_pipeline.ipynb   # Colab end-to-end notebook
├── results/            # saved metrics, plots
└── manuscript/
    ├── main.tex        # IEEEtran two-column manuscript
    ├── references.bib
    └── figures/
```

## Quick start (local — VS Code)

```bash
# from /Users/.../Downloads/cse587/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Download the Champa et al. curated phishing dataset
python scripts/download_data.py

# 2) Smoke-test pipeline (tiny subset, distilbert, 1 epoch)
python -m src.train --quick

# 3) Full baseline
python -m src.train --model bert-base-uncased --epochs 3

# 4) Train with augmentation
python -m src.train --augment --epochs 3

# 5) Evaluate
python -m src.evaluate --checkpoint results/best.pt
```

## Quick start (Colab)

Open `notebooks/phishing_pipeline.ipynb` in Google Colab. The notebook clones
this repo (or just uploads the `src/` folder), installs deps, downloads the
dataset, and runs all three stages with GPU-accelerated training.

## Dataset

Curated phishing email dataset from
[Champa, Rabbi & Zibran (ICMI 2024)](https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952).
Contains both benign and phishing emails standardized across multiple public
corpora.

## Smart-attacker augmentation

Implemented in `src/augmentation.py`. Each augmentation is **label-preserving
for phishing emails**:

| Edit | Description |
|------|-------------|
| `mask_urls` | Replace URLs with neutral placeholders (`[LINK]`) or domain-only forms |
| `remove_urgency` | Remove or paraphrase urgency keywords (URGENT, immediately, ...) |
| `rewrite_subject` | Drop suspicious subject prefixes; rewrite in neutral tone |
| `synonym_swap` | WordNet/embedding-based synonym replacement for cue words |
| `char_perturb` | Light typo-style noise to test character robustness |
| `paraphrase` (optional) | T5-paraphrase via HuggingFace; off by default for speed |

The augmentation pipeline supports both:
- **Train-time augmentation** — produce harder phishing examples to improve robustness
- **Test-time stress test** — a held-out "smart-attacker" eval split

## Reproducing manuscript figures

After running the full pipeline:

```bash
python -m src.evaluate --report markdown > results/report.md
```

The TeX manuscript references the same metrics dump in `results/`.
