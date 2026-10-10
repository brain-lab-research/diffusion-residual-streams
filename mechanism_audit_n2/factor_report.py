"""Exactly one result table, one causal heatmap and seven claim sentences."""
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .engine import aggregate
from .factors import FACTORS, QUARTILES, bootstrap_metrics
from .primary import load_pack, write_json

LABELS = {'read': 'read', 'volume': 'write volume V', 'destination': 'write destination q', 'mixer': 'residual mixer'}


def row(name, group, value):
    return {'group': group, 'condition': name, 'ratio': value['ratio'], 'fixed_factors': '+'.join(value['fixed']),
            'interfaces': ','.join(map(str, value['layers'])), 'ce': value['ce'],
            'intact_ce_at_ratio': value['ce'] - value['delta_ce_intact'],
            'baseline_ce_at_ratio': value['ce'] - value['delta_ce_baseline'],
            'delta_ce_intact': value['delta_ce_intact'], 'delta_ce_intact_ci_low': value['delta_intact_ci95'][0],
            'delta_ce_intact_ci_high': value['delta_intact_ci95'][1], 'fraction_gain_lost': value['fraction_gain_lost'],
            'fraction_gain_lost_ci_low': value['gain_lost_ci95'][0], 'fraction_gain_lost_ci_high': value['gain_lost_ci95'][1],
            'delta_ce_baseline': value['delta_ce_baseline'], 'delta_ce_baseline_ci_low': value['delta_baseline_ci95'][0],
            'delta_ce_baseline_ci_high': value['delta_baseline_ci95'][1], 'accuracy': value['accuracy'],
            'target_logit': value['target_logit'], 'entropy': value['entropy'],
            'masked_tokens': value['masked_tokens'], 'sequences': value['sequences'], 'intact_gain_ce': value['intact_gain_ce']}


def claims(result, intact, baseline):
    primary = result['primary']; selection = result['selection']; selected = selection['selected_condition']
    sentences = [f"На финальных checkpoint’ах mHC-lite n=2 и residual baseline, при r=0.5 на тех же 1024 held-out последовательностях, n=2 имеет actual-mask CE {intact['ce']:.4f} против {baseline['ce']:.4f}, то есть выигрыш {baseline['ce'] - intact['ce']:.4f} CE."]
    if selected:
        value = selection['selection_metrics']; fixed = selection['selected_factors']
        factor_names = ' + '.join(LABELS[f] for f in fixed)
        lo, hi = value['gain_lost_ci95']
        sentences.append(f"Замена только {factor_names} на per-interface calibration mean при сохранении остальных текущих router outputs динамическими даёт ΔCE {value['delta_ce_intact']:+.4f} и убирает {value['fraction_gain_lost']:.1%} наблюдаемого выигрыша n=2, с paired 95% CI [{lo:.1%}; {hi:.1%}].")
        if len(fixed) == 1:
            leave = primary['only_' + fixed[0] + '_dynamic']
            remaining = 1 - leave['fraction_gain_lost']
            sentences.append(f"Когда только {factor_names} остаётся динамическим, а остальные три фактора заменены средними, сохраняется {remaining:.1%} исходного выигрыша, и ΔCE относительно baseline равно {leave['delta_ce_baseline']:+.4f}; это проверка достаточности в рамках этих вмешательств.")
        else:
            sentences.append('Ни один одиночный фактор не убирает половины выигрыша; порог проходит выбранная минимальная комбинация, поэтому результат относится к совместной зависимости факторов, а не к единственному маршрутизирующему сигналу.')
        extensions = result['extensions']
        ratio_summary = '; '.join('r=' + ratio + ': ' + f"{value['all_depths']['fraction_gain_lost']:.0%}" for ratio, value in extensions.items())
        sentences.append('На пяти заранее заданных mask ratios потери выигрыша при том же выбранном вмешательстве составляют ' + ratio_summary + ', с отдельными calibration means для каждого r.')
        quartiles = extensions['0.5']
        largest = max(QUARTILES, key=lambda key: quartiles[key]['delta_ce_intact'])
        localization = '; '.join(key + ': ' + f"{quartiles[key]['fraction_gain_lost']:.1%}" for key in QUARTILES)
        sentences.append(f"Локализация при r=0.5 по четырём непересекающимся квартилям interfaces даёт {localization}, с наибольшим ущербом в {largest}; эти отдельные эффекты не считаются аддитивным разложением полного вмешательства.")
    else:
        sentences.append('Ни одиночные mean interventions, ни минимальная проверенная комбинация не достигают50% потери исходного выигрыша, поэтому расширение по ratios и depth не выполнено.')
        sentences.append('Leave-one-dynamic условия показывают контекстные эффекты факторов, но не устанавливают одного доминирующего носителя выигрыша.')
        sentences.append('Широкие проверки дополнительных механизмов, новые модели и readout probes в этот confirmatory протокол не входят.')
        sentences.append('Наблюдаемое влияние mean interventions ограничено проверенными frozen checkpoint’ами и заданными coefficient replacements.')
    interaction = selection['interaction_contrasts']['all_factor_mean']
    term = interaction['joint_minus_sum_of_single_delta_ce']; lo, hi = interaction['paired_ci95']
    if selection['singletons_below50pct'] and primary['all_factor_mean']['fraction_gain_lost'] >= .5:
        description = 'над-аддитивный interaction подтверждён' if lo > 0 else 'над-аддитивный interaction не установлен'
        sentences.append(f"All-factor mean убирает {primary['all_factor_mean']['fraction_gain_lost']:.1%} gain при слабых одиночных вмешательствах, а joint−sum(single) контраст равен {term:+.4f} CE с CI [{lo:+.4f}; {hi:+.4f}], поэтому {description} и совместный эффект нужно отличать от суммы малых потерь.")
    else:
        sentences.append(f"All-factor mean убирает {primary['all_factor_mean']['fraction_gain_lost']:.1%} gain, а его joint−sum(single) контраст равен {term:+.4f} CE с CI [{lo:+.4f}; {hi:+.4f}]; этот контраст не превращает gain-loss ratios в доли независимых механизмов.")
    sentences.append('Результат поддерживает причинную роль проверенной адаптивной компоненты residual routing в преимуществе этого обученного n=2 на masked positions; доверительные интервалы условны на checkpoint’ах, calibration means и выборе фактора, а переносимость между training seeds и происхождение управляющего сигнала остаются за пределами этой проверки.')
    assert len(sentences) == 7
    return sentences


def heatmap(result, output):
    primary = result['primary']; selected = result['selection']['selected_factors']
    has_extension = bool(result['extensions'])
    fig, axes = plt.subplots(1, 3 if has_extension else 1, figsize=(16 if has_extension else 8, 4.5),
                             gridspec_kw={'width_ratios': [1.6, 1.4, 1.2]} if has_extension else None)
    axes = np.atleast_1d(axes)
    main = np.array([[primary['mean_' + f]['fraction_gain_lost'] for f in FACTORS],
                     [primary['only_' + f + '_dynamic']['fraction_gain_lost'] for f in FACTORS]])
    mats = [main]
    if has_extension:
        mats.append(np.array([[result['extensions'][str(r)]['all_depths']['fraction_gain_lost'] for r in [.1, .3, .5, .7, .9]]]))
        mats.append(np.array([[result['extensions']['0.5'][q]['fraction_gain_lost'] for q in QUARTILES]]))
    extreme = max(1., max(float(np.abs(matrix).max()) for matrix in mats))
    norm = TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
    def draw(ax, matrix, xs, ys, title):
        im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_xticks(np.arange(len(xs)), xs, rotation=20, ha='right'); ax.set_yticks(np.arange(len(ys)), ys)
        ax.set_title(title, fontsize=11)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                color = 'white' if abs(value) > .65 * extreme else '#17272e'
                ax.text(x, y, f'{value:.1%}', ha='center', va='center', fontsize=10, color=color)
        return im
    image = draw(axes[0], main, ['Read', 'Volume V', 'Destination q', 'Mixer'], ['Mean only factor', 'Only factor dynamic'], 'Factor interventions at r=0.5')
    selected_label = ' + '.join(LABELS[f] for f in selected)
    if has_extension:
        draw(axes[1], mats[1], ['0.1', '0.3', '0.5', '0.7', '0.9'], ['All depths'], 'Selected: ' + selected_label + '\nMask ratio specificity')
        draw(axes[2], mats[2], list(QUARTILES), ['r=0.5'], 'Selected: ' + selected_label + '\nDepth quartiles')
    fig.suptitle('Fraction of observed n=2 gain lost (sequence-paired interventions)', fontsize=13)
    fig.text(.03, .03, 'All-factor mean: ' + f"{primary['all_factor_mean']['fraction_gain_lost']:.1%}" +
             '   |   Q1: blocks0–3, Q2:4–7, Q3:8–11, Q4:12–15   |   >100% means worse than baseline', fontsize=9)
    fig.subplots_adjust(left=.1, right=.92, bottom=.23, top=.74, wspace=.5)
    color_axis = fig.add_axes([.94, .24, .012, .49]); bar = fig.colorbar(image, cax=color_axis); bar.set_label('Gain lost', fontsize=9)
    fig.savefig(output / 'causal_heatmap.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / 'causal_heatmap.svg', bbox_inches='tight')
    plt.close(fig)


def render(config, output):
    result = json.loads((output / 'results.json').read_text())
    intact_pack = load_pack(output / 'effects/r50/intact.npz'); base_pack = load_pack(output / 'effects/r50/baseline.npz')
    intact, baseline = aggregate(intact_pack), aggregate(base_pack)
    rows = []
    for name, pack in [('Intact n=2', intact_pack), ('Residual baseline', base_pack)]:
        value = bootstrap_metrics(pack, intact_pack, base_pack, config)
        value.update(ratio=.5, fixed=[], layers=list(range(32)))
        rows.append(row(name, 'primary', value))
    for name, value in result['primary'].items():
        rows.append(row(name, 'primary', value))
    for ratio, values in result['extensions'].items():
        for scope, value in values.items():
            rows.append(row('selected_' + scope, 'selected_extension', value))
    with (output / 'final_table.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sentences = claims(result, intact, baseline)
    heatmap(result, output)
    (output / 'claim.md').write_text(' '.join(sentences) + '\n')
    result['table'] = rows; result['claim_sentences'] = sentences
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    result['definition'] = {'volume': 'V = g0 + g1', 'destination': 'q_i=(g_i-L)/(U-L), L=max(0,V-2), U=min(2,V)',
                            'reconstruction': 'g_i=L(V)+(U(V)-L(V))*q_i',
                            'means': 'Float64 weighted means over actual masked positions of64 separate intact calibration sequences per ratio; simplex normalization only.',
                            'dynamic': 'Unselected factors freshly computed by the current router on the intervened states; visible coefficients unchanged.',
                            'degenerate': 'q=(0.5,0.5) at V=0 or4, where no discretionary destination exists; no gain clipping.',
                            'prior_joint_mean': 'Prior reference averages raw gains jointly; all-factor mean instead averages V andq separately, so these are distinct interventions.'}
    scripts = ['factors.py', 'factor_protocol.py', 'factor_report.py', 'factor_verify.py', 'engine.py', 'primary.py', 'safety.py']
    result['provenance'] = {'analysis_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in scripts},
        'frozen_sources_unchanged': {file: hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest for file, digest in config['source_files'].items()},
        'model_paths': {name: config['models'][name]['checkpoint'] for name in ['full_n2', 'baseline']}}
    write_json(output / 'final_results.json', result)
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>n=2 factor-wise causal confirmation</title><style>body{font:15px/1.5 system-ui,sans-serif;color:#203039;max-width:1400px;margin:auto;padding:28px}h1{line-height:1.2}p{max-width:1100px}.claim{background:#edf6f8;border-left:4px solid #267c90;padding:14px 22px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px 7px;border-bottom:1px solid #dae4e7;text-align:left}th{background:#eaf1f4;position:sticky;top:0}.wrap{overflow-x:auto}pre{white-space:pre-wrap;background:#f0f5f7;padding:15px}.small{font-size:13px;color:#52646e}</style><body><h1>mHC-lite n=2: factor-wise causal confirmation</h1>']
    parts.append('<div class="claim"><p>' + ' '.join(esc(s) for s in sentences) + '</p></div>')
    parts.append('<p class="small">Frozen final checkpoint53399; actual-mask CE; paired sequence bootstrap5000 draws; 64 calibration и1024 test sequences, disjoint; только выбранное вмешательство проверено по ratios и depth, без новых моделей/probes/lenses/regressions.</p>')
    parts.append('<p>Физический объём записи: V=g₀+g₁. Допустимый destination: qᵢ=(gᵢ−L)/(U−L), где L=max(0,V−2), U=min(2,V); после вмешательства gᵢ=L+(U−L)qᵢ. Means вычислены в float64 по masked positions отдельно для каждого interface и r; q и mixer сохраняют simplex, read∈[0,1], gᵢ∈[0,2], без clipping gains.</p>')
    parts.append('<img alt="Causal gain loss heatmap" src="data:image/png;base64,' + base64.b64encode((output / 'causal_heatmap.png').read_bytes()).decode() + '">')
    parts.append('<h2>Итоговая таблица</h2><p>Доля потери = (CE_condition−CE_n2)/(CE_baseline−CE_n2); значения вне0–100% допустимы. Все CI парные по sequences. Quartile means меняют только interfaces указанного диапазона.</p><div class="wrap"><table><thead><tr><th>Условие</th><th>r</th><th>CE</th><th>ΔCE n2 [95% CI]</th><th>Gain lost [95% CI]</th><th>ΔCE baseline [95% CI]</th><th>Accuracy</th><th>Target logit</th><th>Entropy</th></tr></thead><tbody>')
    for v in rows:
        cells = [v['condition'], str(v['ratio']), f"{v['ce']:.5f}",
                 f"{v['delta_ce_intact']:+.5f} [{v['delta_ce_intact_ci_low']:+.5f}; {v['delta_ce_intact_ci_high']:+.5f}]",
                 f"{v['fraction_gain_lost']:.1%} [{v['fraction_gain_lost_ci_low']:.1%}; {v['fraction_gain_lost_ci_high']:.1%}]",
                 f"{v['delta_ce_baseline']:+.5f} [{v['delta_ce_baseline_ci_low']:+.5f}; {v['delta_ce_baseline_ci_high']:+.5f}]",
                 f"{v['accuracy']:.2%}", f"{v['target_logit']:.3f}", f"{v['entropy']:.3f}"]
        parts.append('<tr>' + ''.join('<td>' + esc(cell) + '</td>' for cell in cells) + '</tr>')
    parts.append('</tbody></table></div><details><summary>Определения, selection rule и проверки invariants</summary><pre>' + esc(json.dumps({'definition': result['definition'], 'protocol': result['protocol'], 'selection': result['selection'], 'unit': result['unit']}, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<p class="small">CLI: python -m mechanism_audit_n2.factors --stage full --coordinate bounded --gpu 0 (на свободной GPU; guard чужих процессов). Артефакты: final_table.csv, causal_heatmap.png/svg, claim.md, final_results.json, config_full.json, calibration/*.json и effects/*/*.npz.</p></body></html>')
    (output / 'report.html').write_text('\n'.join(parts))
    (output / 'README.md').write_text('Factor-specific causal confirmation, final mHC-lite n=2.\n\n' +
        'Run from analysis directory:\n\n```bash\npython -m mechanism_audit_n2.factors --stage full --coordinate bounded --gpu 0\npython -m mechanism_audit_n2.factors --stage render\n```\n\n' +
        'Same 1024 held-out block IDs, per-ratio fixed masks, separate 64 calibration blocks; means in float64. No models trained.\n' +
        'V=g0+g1, q_i=(g_i-L)/(U-L), g_i=L+(U-L)q_i, L=max(0,V-2), U=min(2,V).\n' +
        'Original per-stream gain constraints maintained, no clipping. Current unselected factors and all visible coefficients stay dynamic.\n' +
        'Only minimal selected >=50% point-estimate gain-loss set extends to five ratios; quartiles localized at r=.5 only.\n' +
        'final_table.csv: all conditions andmetrics with three pairedCE/gain intervals; causal_heatmap.png/svg: single causal figure; claim.md: seven sentences.\n' +
        'effects/r*/condition.npz: aligned token_stats (CE,target logit,entropy,target rank,top1,correct,target margin,prediction margin), seq_ce_sum,actual seq_mask_count,block_ids,positions,token_sequence.\n' +
        'CI are nominal conditional sequence bootstrap, not a training seed or independent post-selection replication interval.\n\n' +
        'Verify completed masks, means, raw sums and scope: `python -m mechanism_audit_n2.factor_verify`; result in verification.json.\n')
    print('DELIVERABLE COMPLETE', str(output / 'report.html'), len(rows), 'rows', flush=True)
