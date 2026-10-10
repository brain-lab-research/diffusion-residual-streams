"""Sealed splits and condition names for physical destination experiments."""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def settings(model):
    if model == '337m':
        from mechanism_audit_n2.engine import batches, load_model, make_split
        config = json.loads((BASE / 'mechanism_audit_n2/config.json').read_text())
        calibration = BASE / 'volume_analysis/calibration_means.json'
        reference = BASE / 'volume_analysis/references'
        return config, calibration, reference, 'confirmation', 53399, 32, batches, load_model, make_split
    if model == '1b':
        sys.path.insert(0, str(BASE / 'analysis_1b'))
        from audit import CONFIG, ROOT, batches, load_model, make_split
        return CONFIG, ROOT / 'calibration.json', ROOT, 'test', 124000, 64, batches, load_model, make_split
    raise ValueError(model)


def cases(model):
    if model == '337m':
        for factor in ('read', 'mixer'):
            for policy in ('global', 'role', 'role_swap'):
                yield f'{factor}_{policy}', (factor,), policy
    for policy in ('global', 'role', 'role_swap'):
        if model == '337m' or policy != 'role_swap':
            yield f'destination_{policy}', ('destination',), policy
    if model == '337m':
        for policy in ('global', 'role', 'role_swap'):
            yield f'joint_{policy}', ('read', 'volume', 'destination', 'mixer'), policy
