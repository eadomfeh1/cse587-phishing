"""Download the Champa et al. (2024) curated phishing email dataset.

figshare record: https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952
"""
from __future__ import annotations
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FIGSHARE_API, RAW_DIR  # noqa: E402


def main() -> None:
    print(f"Querying figshare metadata: {FIGSHARE_API}")
    meta = requests.get(FIGSHARE_API, timeout=30).json()
    files = meta.get("files", [])
    if not files:
        raise SystemExit(
            "No files found in figshare metadata. "
            "Manually download from figshare and place CSVs under data/raw/."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f["name"]
        url = f["download_url"]
        target = RAW_DIR / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  ✓ {name} (cached)")
            continue
        print(f"  ↓ {name} from {url}")
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        target.write_bytes(r.content)
        # auto-unzip to raw/
        if name.lower().endswith(".zip"):
            print(f"    extracting {name}")
            with zipfile.ZipFile(io.BytesIO(target.read_bytes())) as z:
                z.extractall(RAW_DIR)
    print(f"Done. Files under {RAW_DIR}")


if __name__ == "__main__":
    main()
