"""Misc utilities: seeding, logging, JSON IO."""
from __future__ import annotations
import json
import logging
import os
import random
from pathlib import Path

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dump_json(obj, path: Path | str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path | str):
    return json.loads(Path(path).read_text())
