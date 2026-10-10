"""Freeze and verify the model weights from the retained step-124000 checkpoint."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("ONE_B_CHECKPOINT", ROOT.parent / "runs_pt/diff1b_mhc_n2_10b_s1337/ckpt_124000.pt"))
TARGET = ROOT / "mhc_step124000_model_only.pt"
MANIFEST = ROOT / "snapshot_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if TARGET.exists() or MANIFEST.exists():
        raise FileExistsError("Frozen checkpoint or manifest already exists")
    source_stat = SOURCE.stat()
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    assert state["step"] == 124000 and state["arm"] == "mhc"
    frozen = {key: state[key] for key in ("model", "step", "arm")}
    temporary = TARGET.with_suffix(".pt.tmp")
    torch.save(frozen, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=False)
    assert restored["step"] == 124000 and restored["arm"] == "mhc"
    assert restored["model"].keys() == state["model"].keys()
    for name, tensor in state["model"].items():
        if not torch.equal(tensor, restored["model"][name]):
            raise AssertionError(f"Changed tensor: {name}")
    digest = sha256(temporary)
    temporary.replace(TARGET)
    result = {
        "checkpoint_step": 124000,
        "total_input_tokens": (124000 + 1) * 65536,
        "frozen_model": str(TARGET.relative_to(ROOT.parent)),
        "frozen_bytes": TARGET.stat().st_size,
        "frozen_sha256": digest,
        "source_training_checkpoint": os.path.relpath(SOURCE, ROOT.parent),
        "source_bytes_at_capture": source_stat.st_size,
        "source_mtime_at_capture_utc": datetime.fromtimestamp(
            source_stat.st_mtime, timezone.utc).isoformat(),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contents": list(frozen),
        "weights_modified": False,
        "bitwise_verified_against_source_model_tensors": len(state["model"]),
    }
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
