"""Pure-volume controls for the depth endpoints and corruption sweep."""
import json

import torch

from audit import CONFIG, ROOT, load_model, make_split, paired_delta, save_json
from volume_calibration import intact_for_ratio, mean_volume_for_ratio
from volume_physical_control import PhysicalShareHooks, evaluate


def main():
    torch.set_num_threads(4)
    engine = load_model(CONFIG, 'full_n2')
    assert engine.checkpoint_step == 124000
    summary_path = ROOT/'volume_physical_followup.json'
    summary = {'checkpoint_step': 124000, 'conditions': {}}
    for ratio in (.5, .1, .3, .7, .9):
        split = make_split(CONFIG, 'test', ratio)
        means = mean_volume_for_ratio(engine, ratio)
        intact = intact_for_ratio(engine, split, ratio)
        hooks = PhysicalShareHooks(engine, means)
        try:
            cases = ([(f'mean_{role}', (2*block, 2*block+1))
                      for block in range(32) for role in ('masked', 'visible')]
                     if ratio == .5 else [('mean_both', None)])
            for mode, layers in cases:
                key = f'r{round(100*ratio):02d}_{mode}_' + (
                    'all' if layers is None else '_'.join(map(str, layers)))
                pack, diag = evaluate(engine, split, hooks, mode, 'proportional',
                                      ratio=ratio, layers=layers)
                summary['conditions'][key] = {
                    'effect_vs_intact': paired_delta(pack, intact), 'diagnostics': diag}
                save_json(summary_path, summary)
        finally:
            hooks.close()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
