"""Smart-attacker data augmentation.

Each operation simulates an edit a sophisticated attacker (or an LLM-generated
phishing email) would apply to evade detection while preserving the malicious
intent of the message.

Operations
----------
* mask_urls         — replace URLs with neutral placeholders or domain-only forms
* remove_urgency    — drop urgency cues (URGENT, ASAP, immediately, ...)
* rewrite_subject   — strip suspicious prefixes / rephrase subject neutrally
* synonym_swap      — WordNet synonym swap on cue / sentiment words
* char_perturb      — light typo-style noise (homoglyph + whitespace + swap)
* paraphrase        — (optional) T5 paraphrase, requires an extra model load

All operations preserve the *label* of phishing emails — they only remove or
disguise surface cues, not the underlying content.
"""
from __future__ import annotations
import logging
import random
import re
import string
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .config import URGENCY_KEYWORDS, URL_REGEX

logger = logging.getLogger(__name__)

# WordNet is loaded lazily on first synonym swap.
_WN_INITIALIZED = False


def _ensure_wordnet():
    global _WN_INITIALIZED
    if _WN_INITIALIZED:
        return
    try:
        import nltk
        from nltk.corpus import wordnet  # noqa: F401
        try:
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        _WN_INITIALIZED = True
    except Exception as e:  # pragma: no cover
        logger.warning("WordNet unavailable; synonym_swap disabled: %s", e)


# ---------------------------------------------------------------------------
# Individual ops
# ---------------------------------------------------------------------------
def mask_urls(text: str, mode: str = "placeholder") -> str:
    """Replace URLs.

    mode = 'placeholder' -> '[LINK]'
    mode = 'domain'      -> keep only the domain (looks innocent: 'example.com')
    mode = 'remove'      -> delete entirely
    """
    pattern = re.compile(URL_REGEX, flags=re.IGNORECASE)

    def _sub(m: re.Match) -> str:
        url = m.group(0)
        if mode == "placeholder":
            return "[LINK]"
        if mode == "remove":
            return ""
        if mode == "domain":
            dom = re.search(r"([a-z0-9-]+\.[a-z]{2,})", url, flags=re.IGNORECASE)
            return dom.group(1) if dom else "[LINK]"
        return url

    return re.sub(r"\s+", " ", pattern.sub(_sub, text)).strip()


_URGENCY_PATTERNS = [
    re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE) for kw in URGENCY_KEYWORDS
]
_URGENCY_REPLACEMENTS = {
    "urgent": "", "immediately": "soon", "asap": "soon",
    "right away": "soon", "act now": "please respond",
    "verify now": "please verify when convenient",
    "important": "", "alert": "notice", "warning": "notice",
    "attention": "", "critical": "",
    "click here": "see link", "limited time": "available",
    "final notice": "notice", "last chance": "opportunity",
    "suspended": "needs review", "deactivated": "needs review",
    "compromised": "needs review", "unusual activity": "recent activity",
}


def remove_urgency(text: str, keep_prob: float = 0.0) -> str:
    """Strip / soften urgency keywords."""
    out = text
    for kw, pat in zip(URGENCY_KEYWORDS, _URGENCY_PATTERNS):
        if random.random() < keep_prob:
            continue
        repl = _URGENCY_REPLACEMENTS.get(kw.lower(), "")
        out = pat.sub(repl, out)
    # Cleanup: collapse whitespace, drop dangling/double punctuation that
    # the keyword removals can leave behind (e.g. "Subject: : Foo").
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s*([,;:!?])\s*\1+", r"\1", out)         # ": :"  -> ":"
    out = re.sub(r"([:;,])\s*([:;,!?])", r"\1", out)        # ": ,"  -> ":"
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)              # " ,"   -> ","
    return out.strip(" ,.;:")


_SUBJ_BAD_PREFIXES = re.compile(
    r"^\s*(re:\s*|fw:\s*|urgent[:\-]?\s*|important[:\-]?\s*|"
    r"alert[:\-]?\s*|warning[:\-]?\s*|attention[:\-]?\s*|action required[:\-]?\s*)+",
    flags=re.IGNORECASE,
)


def rewrite_subject(text: str) -> str:
    """If the input contains a 'Subject: ...' line, strip suspicious prefixes."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            head, _, rest = line.partition(":")
            cleaned = _SUBJ_BAD_PREFIXES.sub("", rest).strip()
            cleaned = cleaned.replace("!!", "").replace("!!!", "").strip()
            if not cleaned:
                cleaned = "Update"
            lines[i] = f"{head}: {cleaned}"
            break
    return "\n".join(lines)


def synonym_swap(text: str, p: float = 0.1, max_swaps: int = 6) -> str:
    """Swap a small fraction of content words with WordNet synonyms."""
    _ensure_wordnet()
    try:
        from nltk.corpus import wordnet
    except Exception:
        return text

    tokens = text.split()
    swaps = 0
    for i, tok in enumerate(tokens):
        if swaps >= max_swaps:
            break
        clean = tok.strip(string.punctuation)
        if len(clean) < 4 or not clean.isalpha():
            continue
        if random.random() > p:
            continue
        syns = wordnet.synsets(clean.lower())
        cands = []
        for s in syns:
            for lemma in s.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != clean.lower() and " " not in name and name.isalpha():
                    cands.append(name)
        if not cands:
            continue
        repl = random.choice(cands)
        if clean[0].isupper():
            repl = repl.capitalize()
        tokens[i] = tok.replace(clean, repl)
        swaps += 1
    return " ".join(tokens)


_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}  # cyrillic look-alikes


def char_perturb(text: str, p: float = 0.02) -> str:
    """Light character-level noise: occasional homoglyph or adjacent swap."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isalpha() and random.random() < p:
            roll = random.random()
            if roll < 0.5 and ch.lower() in _HOMOGLYPHS:
                out.append(_HOMOGLYPHS[ch.lower()])
            elif roll < 0.8 and i + 1 < len(text) and text[i + 1].isalpha():
                out.append(text[i + 1])
                out.append(ch)
                i += 2
                continue
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
OP_REGISTRY: dict[str, Callable[[str], str]] = {
    "mask_urls": mask_urls,
    "remove_urgency": remove_urgency,
    "rewrite_subject": rewrite_subject,
    "synonym_swap": synonym_swap,
    "char_perturb": char_perturb,
}


@dataclass
class AugmentationConfig:
    ops: tuple = ("mask_urls", "remove_urgency", "rewrite_subject", "synonym_swap")
    factor: int = 1     # how many augmented copies per original phishing email
    seed: int = 42


def apply_pipeline(text: str, ops: Iterable[str]) -> str:
    out = text
    for op_name in ops:
        op = OP_REGISTRY.get(op_name)
        if op is None:
            logger.warning("Unknown augmentation op: %s", op_name)
            continue
        try:
            out = op(out)
        except Exception as e:  # pragma: no cover
            logger.warning("Op %s failed on a sample: %s", op_name, e)
    return out


def augment_dataframe(
    df: pd.DataFrame,
    cfg: AugmentationConfig,
    only_phishing: bool = True,
) -> pd.DataFrame:
    """Return df + N augmented copies appended.

    Augmentation is restricted to the phishing class by default — that's where
    "smart-attacker" simulation matters. Augmenting benign emails would just
    add noise to a class that doesn't have these cues to begin with.
    """
    random.seed(cfg.seed)
    new_rows = []
    target = df[df.label == 1] if only_phishing else df
    for _ in range(cfg.factor):
        for _, row in target.iterrows():
            new_rows.append({
                "subject": row.get("subject", ""),
                "body": row["body"],
                "label": row["label"],
                "source": f"{row.get('source', 'orig')}__aug",
                "text": apply_pipeline(
                    row.get("text", row["body"]), cfg.ops
                ),
            })
    if not new_rows:
        return df.copy()
    aug_df = pd.DataFrame(new_rows)
    if "text" not in df.columns:
        df = df.assign(text=df["body"])
    return pd.concat([df, aug_df], ignore_index=True)


def make_smart_attacker_eval(df: pd.DataFrame, cfg: AugmentationConfig) -> pd.DataFrame:
    """Held-out 'smart attacker' eval split: phishing rows have the full pipeline
    applied; benign rows are unchanged. Same labels as input."""
    out = df.copy()
    if "text" not in out.columns:
        out["text"] = out["body"]
    mask = out.label == 1
    out.loc[mask, "text"] = out.loc[mask, "text"].apply(
        lambda t: apply_pipeline(t, cfg.ops)
    )
    out["source"] = out["source"].astype(str) + "__smart"
    return out
