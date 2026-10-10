"""Configuration provenance and diagnostics for FRR runs."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from frr import FRRInterface, route_tables

ROOT = Path(__file__).resolve().parent


def git_output(*args):
    if not (ROOT / '.git').exists():
        return None
    result = subprocess.run(['git', *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def provenance():
    paths = [ROOT / name for name in ('model_and_train.py', 'frr.py', 'frr_experiment.py')]
    paths.append(ROOT / 'textdiffusion/models/mhc_lite.py')
    return dict(commit=git_output('rev-parse', 'HEAD'),
                git_status=git_output('status', '--short'),
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


# Options added to the CLI after these runs started.  A resume is allowed only while each
# sits at the value the original code behaved with, so the numerics are unchanged:
# val_micro 16 and warmup_steps 300 were hardcoded before, target_tokens leaves steps alone.
RESUME_NEUTRAL_ARGS = {'val_micro': 16, 'warmup_steps': 300, 'target_tokens': None}


def _source_drift(old_hashes, new_hashes):
    return {path: {'recorded': digest, 'now': new_hashes.get(path)}
            for path, digest in old_hashes.items() if new_hashes.get(path) != digest}


def record_config(args, world):
    if world != 1 or args.micro != 16 or args.accum != 4 or args.steps != 53400 or args.objective != 'mdlm':
        raise ValueError('Production FRR must match diff36_lit_s2: world=1, micro=16, accum=4, steps=53400, MDLM')
    env = {k: os.environ.get(k) for k in ('MHC_STREAMS', 'MHC_PAPER', 'MHC_PROPER_INIT', 'MHC_RESMIX', 'PRETRAIN_DATA')}
    fixed = {k: v for k, v in vars(args).items() if k not in ('stop_after',)}
    prov = provenance()
    payload = dict(arguments=fixed, environment=env, world=world, seed=1337, tokens_per_update=65536,
                   volume_init=1.0, backbone_init='Construct full paper n=2 first, then replace routers',
                   data_meta=json.loads((Path(env['PRETRAIN_DATA'])/'meta.json').read_text()), **prov)
    path = Path(args.out) / 'config.json'
    if not path.exists():
        atomic_json(path, payload)
        return
    old = json.loads(path.read_text())
    drift = {}
    for key in ('arguments', 'environment', 'world', 'source_sha256', 'data_meta'):
        if old[key] == payload[key]:
            continue
        if key == 'arguments':
            changed = {k: [old[key][k], payload[key].get(k)] for k in old[key]
                       if old[key][k] != payload[key].get(k)}
            added = {k: v for k, v in payload[key].items() if k not in old[key]}
            unsafe = {k: v for k, v in added.items()
                      if k not in RESUME_NEUTRAL_ARGS or v != RESUME_NEUTRAL_ARGS[k]}
            if changed or unsafe:
                raise ValueError(f'Refusing incompatible resume: arguments {changed or unsafe}')
            drift['arguments_added_at_neutral_defaults'] = added
            continue
        if key == 'source_sha256':
            # The training script changed after this run started.  Resuming is allowed, but
            # the run records exactly which sources moved and how, so the provenance survives.
            drift['modified_sources'] = _source_drift(old[key], payload[key])
            try:
                names = [Path(name).name for name in drift['modified_sources']]
                drift['modified_sources_diff'] = subprocess.check_output(
                    ['git', 'diff', '--', *names], cwd=ROOT, text=True)[:200000]
            except Exception as error:
                drift['modified_sources_diff'] = f'unavailable: {error}'
            continue
        raise ValueError(f'Refusing incompatible resume: {key}')
    if drift:
        drift['recorded_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        drift['commit_now'] = payload['commit']
        old.setdefault('resumes_under_changed_provenance', []).append(drift)
        atomic_json(path, old)


@torch.no_grad()
def diagnostics(model, blocks, device, out, step):
    from model_and_train import MASK_ID
    interfaces = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, FRRInterface)]
    training = model.training
    model.eval()
    for _, mod in interfaces:
        mod.collect_stats = True
    ids = blocks[:4].to(device)
    g = torch.Generator().manual_seed(779)
    mask = (torch.rand(ids.shape, generator=g) < .5).to(device)
    inp = ids.clone()
    inp[mask] = MASK_ID
    roles = inp.eq(MASK_ID)
    try:
        with torch.autocast('cuda', torch.bfloat16, enabled=str(device).startswith('cuda')):
            model(inp)
        records = route_tables(model)
        for record, (_, mod) in zip(records, interfaces):
            stats = mod.last_stats
            record['roles'] = {}
            for role, label in [(False, 'visible'), (True, 'masked')]:
                selected = roles == role
                volume = stats['volume'][selected]
                record['roles'][label] = dict(volume_mean=volume.mean().item(), volume_std=volume.std(unbiased=False).item(),
                    volume_min=volume.min().item(), volume_max=volume.max().item(),
                    volume_quantiles=torch.quantile(volume, torch.tensor([0., .1, .5, .9, 1.], device=volume.device)).tolist(),
                    saturation_fraction=((volume < .01) | (volume > 1.99)).float().mean().item(),
                    stream_norm=stats['stream_norm'][selected].mean(0).tolist(),
                    stream_cosine=stats['stream_cosine'][selected].mean().item(),
                    stream_difference=stats['stream_difference'][selected].mean().item())
        atomic_json(Path(out)/f'diagnostics_{step}.json', dict(step=step, interfaces=records))
    finally:
        for _, mod in interfaces:
            mod.collect_stats = False
            mod.last_stats = None
        model.train(training)
