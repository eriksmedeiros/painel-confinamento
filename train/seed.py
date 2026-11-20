"""Utilitário de reprodutibilidade para experimentos (RNF01)."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixa seeds de Python, NumPy e PyTorch para garantir reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
