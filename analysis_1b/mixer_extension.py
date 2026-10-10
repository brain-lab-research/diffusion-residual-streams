"""Complete the 1B mixer controls on the frozen 128-sequence split."""

import json

import torch

from audit import CONFIG, ROOT, calibrate, evaluate, load_model, make_split, paired_delta, save_json
from mechanism_audit_n2.role_interventions import RoleFactorHooks


CASES = (
    ("mixer", "role"),
    ("mixer", "role_swap"),
)


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, "full_n2")
    assert engine.checkpoint_step == 124000 and len(engine.interfaces) == 64
    means, _, _ = calibrate(engine)
    split = make_split(CONFIG, "test", 0.5)
    intact = evaluate(engine, split, "intact")
    hooks = RoleFactorHooks(engine, means)
    results = {"checkpoint_step": 124000, "ratio": 0.5,
               "test_sequences": len(split["ids"]),
               "calibration_sequences": 64, "conditions": {}}
    try:
        for factor, policy in CASES:
            name = f"{factor}_{policy}"
            pack = evaluate(engine, split, name, hooks, (factor,), policy)
            results["conditions"][name] = paired_delta(pack, intact)
            print("ROLE EXTENSION", name, results["conditions"][name], flush=True)
            save_json(ROOT / "roles_extended.json", results)
    finally:
        hooks.close()
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
