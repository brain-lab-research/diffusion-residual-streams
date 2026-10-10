"""Shared artifact readers and writers for causal experiments."""
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pack_path(output, model, split, condition):
    path = Path(output) / 'effects' / model / split / (condition + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pack(path, pack, intact=None):
    if intact is not None:
        pack = dict(pack)
        for index, name in [(0, 'ce'), (1, 'target_logit'), (2, 'entropy'),
                            (5, 'correct'), (6, 'target_margin')]:
            pack['token_delta_' + name] = pack['token_stats'][:, index] - intact['token_stats'][:, index]
    np.savez_compressed(path, **pack)


def load_pack(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}
