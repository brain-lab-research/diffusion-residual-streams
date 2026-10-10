"""Capture n=2 router outputs on disjoint calibration blocks for all noise levels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_audit_n2.engine import batches, load_model, make_split
from mechanism_audit_n2.factors import ROOT
from mechanism_audit_n2.safety import GPUReservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    config = json.loads((ROOT / "mechanism_audit_n2/config.json").read_text())
    reservation = GPUReservation(args.gpu, config["gpu_policy"])
    engine = load_model(config, "full_n2")
    output = ROOT / "residual_circuits_n2/captures"
    output.mkdir(parents=True, exist_ok=True)
    for ratio in (0.1, 0.3, 0.5, 0.7, 0.9):
        split = make_split(config, "calibration", ratio)
        path = output / f"r{round(ratio * 100):02d}_all_positions.npz"
        pieces = []
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            for _, _, _, _, corrupted, _, mask in batches(split, config["batch_size"]):
                reservation.check()
                local = []
                def capture(index, before, after, routes, routed, delta):
                    local.append(routes.reshape(-1, 6).float().cpu().numpy())
                engine.run(corrupted, mask, callback=capture)
                pieces.append(np.stack(local))
        router = np.concatenate(pieces, axis=1)
        np.savez_compressed(path, router=router, mask=split["mask"],
                            block_ids=split["block_ids"], ratio=ratio,
                            mask_sha256=split["mask_sha256"], data_sha256=split["data_sha256"])
        print(f"{path}: {router.shape}", flush=True)


if __name__ == "__main__":
    main()
