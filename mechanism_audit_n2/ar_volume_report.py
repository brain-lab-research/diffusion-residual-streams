"""Paired sequence bootstrap and standalone AR volume report."""
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
import numpy as np

from .ar_volume import CONDITIONS, OUTPUT, POSITION_BINS, ROOT, sources
from .primary import load_pack, write_json

LABELS = {'dynamic': 'AR n=2 · dynamic', 'baseline': 'AR baseline',
    'volume_global': 'V · interface mean', 'volume_context_quartile': 'V · context quartile mean',
    'volume_position': 'V · exact position mean',
    'volume_Q1': 'V mean · depth Q1', 'volume_Q2': 'V mean · depth Q2',
    'volume_Q3': 'V mean · depth Q3', 'volume_Q4': 'V mean · depth Q4',
    'volume_attention': 'V mean · attention only', 'volume_mlp': 'V mean · MLP only'}
METRICS = {'ce': 0, 'target_logit': 1, 'entropy': 2, 'accuracy': 5, 'target_margin': 6}


def ci(samples):
    return np.quantile(samples, [.025, .975]).tolist()


def interval_text(point, bounds, digits=5):
    return f'{point:+.{digits}f} [{bounds[0]:+.{digits}f}, {bounds[1]:+.{digits}f}]'


def summarize(config):
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    assert protocol['computation_sha256'] == sources()
    repeats, seed = config['bootstrap_repeats'], config['bootstrap_seed']
    draws = np.random.default_rng(seed).integers(1024, size=(repeats, 1024))
    arrays, means, by_position = {}, {}, {}
    fingerprints = {}
    for condition in CONDITIONS:
        path = OUTPUT / 'effects' / (condition + '.npz')
        pack = load_pack(path)
        assert np.array_equal(pack['block_ids'], np.asarray(protocol['test_blocks']))
        assert np.all(pack['seq_mask_count'] == 1023)
        assert pack['token_stats'].shape == (1024 * 1023, 8)
        assert np.isfinite(pack['token_stats']).all()
        tokens = pack['token_stats'].reshape(1024, 1023, 8)
        arrays[condition] = tokens.mean(1, dtype=np.float64)
        # Recompute CE from float64 sums, also verifying shifted scoring provenance.
        np.testing.assert_allclose(arrays[condition][:, 0], pack['seq_ce_sum'] / 1023, atol=1e-12, rtol=0)
        means[condition] = arrays[condition].mean(0)
        by_position[condition] = np.stack([tokens[:, lo:hi, 0].mean(1, dtype=np.float64)
                                          for lo, hi in POSITION_BINS], axis=1)
        fingerprints[condition] = hashlib.sha256(path.read_bytes()).hexdigest()
    gain_seq = arrays['baseline'][:, 0] - arrays['dynamic'][:, 0]
    gain = float(gain_seq.mean()); gain_draws = gain_seq[draws].mean(1)
    stable_gain = gain > 0 and np.all(gain_draws > 1e-12)
    rows = []
    for condition in CONDITIONS:
        left = arrays[condition]; dynamic = arrays['dynamic']; baseline = arrays['baseline']
        row = {'condition': condition, 'label': LABELS[condition]}
        for name, index in METRICS.items():
            row[name] = float(means[condition][index])
            diff = left[:, index] - dynamic[:, index]
            row['delta_' + name] = float(diff.mean())
            row['delta_' + name + '_ci95'] = ci(diff[draws].mean(1))
        delta_baseline = left[:, 0] - baseline[:, 0]
        row['delta_ce_baseline'] = float(delta_baseline.mean())
        row['delta_ce_baseline_ci95'] = ci(delta_baseline[draws].mean(1))
        delta = left[:, 0] - dynamic[:, 0]
        row['gain_lost_pct'] = 100 * float(delta.mean()) / gain if stable_gain else None
        row['gain_lost_pct_ci95'] = ci(100 * delta[draws].mean(1) / gain_draws) if stable_gain else None
        row['position_bins'] = []
        for k, (lo, hi) in enumerate(POSITION_BINS):
            difference = by_position[condition][:, k] - by_position['dynamic'][:, k]
            row['position_bins'].append({'source_positions': [lo, hi - 1],
                'context_lengths': [lo + 1, hi], 'tokens': 1024 * (hi - lo),
                'ce': float(by_position[condition][:, k].mean()),
                'delta_ce': float(difference.mean()), 'delta_ce_ci95': ci(difference[draws].mean(1))})
        rows.append(row)
    comparisons = {}
    for condition in ('volume_context_quartile', 'volume_position'):
        difference = arrays['volume_global'][:, 0] - arrays[condition][:, 0]
        numerator = difference[draws].mean(1)
        denominator = (arrays['volume_global'][:, 0] - arrays['dynamic'][:, 0])[draws].mean(1)
        stable = np.all(denominator > 1e-12) and float(denominator.mean()) > 0
        comparisons[condition] = {'global_minus_condition_ce': float(difference.mean()),
            'ci95': ci(numerator), 'rescue_of_global_volume_damage_pct':
                100 * float(difference.mean()) / (means['volume_global'][0] - means['dynamic'][0]) if stable else None,
            'rescue_ci95': ci(100 * numerator / denominator) if stable else None}
    joint_delta = arrays['volume_global'][:, 0] - arrays['dynamic'][:, 0]
    quartile_sum = sum(arrays['volume_' + q][:, 0] - arrays['dynamic'][:, 0]
                       for q in ('Q1', 'Q2', 'Q3', 'Q4'))
    interaction = joint_delta - quartile_sum
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
        'sequences': 1024, 'next_token_targets': 1024 * 1023,
        'baseline_minus_n2_gain': gain, 'gain_ci95': ci(gain_draws), 'gain_ratio_stable': bool(stable_gain),
        'rows': rows, 'position_control_comparisons': comparisons,
        'depth_nonadditivity': {'joint_minus_sum_quartile_delta_ce': float(interaction.mean()),
            'ci95': ci(interaction[draws].mean(1)),
            'interpretation': 'Descriptive interaction contrast of the same volume mean replacements across four depth groups; not an additive attribution of the baseline gain.'},
        'pack_sha256': fingerprints,
        'bootstrap': {'unit': 'Paired held-out sequence; full sequence resampled with all its predictions.',
                      'repeats': repeats, 'seed': seed, 'interval': 'Percentile 95%; conditional on fixed calibration split.'}}
    return result


def figure(result, calibration):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = {r['condition']: r for r in result['rows']}
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1))
    ax = fig.add_subplot(grid[0, 0])
    selected = [n for n in CONDITIONS if n not in ('dynamic', 'baseline')]
    values = np.array([rows[n]['delta_ce'] for n in selected])
    bounds = np.array([rows[n]['delta_ce_ci95'] for n in selected])
    y = np.arange(len(selected))
    colors = ['#c84d2d' if n == 'volume_global' else '#24798e' for n in selected]
    ax.barh(y, values, color=colors, alpha=.85)
    ax.errorbar(values, y, xerr=np.maximum(0, np.stack([values - bounds[:, 0], bounds[:, 1] - values])),
                fmt='none', ecolor='#222', capsize=3)
    ax.set_yticks(y, [LABELS[n].removeprefix('V ') for n in selected]); ax.invert_yaxis()
    ax.axvline(0, color='#444', linewidth=.8)
    ax.axvline(result['baseline_minus_n2_gain'], color='#555', linestyle='--', label='Baseline − intact n=2')
    ax.set_xlabel('Δ next-token CE vs intact AR n=2'); ax.set_title('A. Causal volume interventions · paired 95% CI')
    ax.legend(fontsize=9, loc='lower right')
    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(4)
    for condition, color in [('volume_global', '#c84d2d'), ('volume_context_quartile', '#24798e'),
                             ('volume_position', '#4b9d62')]:
        point = np.array([b['delta_ce'] for b in rows[condition]['position_bins']])
        bounds = np.array([b['delta_ce_ci95'] for b in rows[condition]['position_bins']])
        ax.errorbar(x, point, yerr=np.maximum(0, np.stack([point - bounds[:, 0], bounds[:, 1] - point])),
                    color=color, marker='o', capsize=3, label=LABELS[condition])
    ax.axhline(0, color='#555', linewidth=.8)
    ax.set_xticks(x, ['1–256', '257–512', '513–768', '769–1023'])
    ax.set_xlabel('Available prefix length'); ax.set_ylabel('Δ next-token CE')
    ax.set_title('B. Content adaptation beyond fixed position policies'); ax.legend(fontsize=9)
    ax = fig.add_subplot(grid[1, 0])
    heat_names = ['volume_Q1', 'volume_Q2', 'volume_Q3', 'volume_Q4', 'volume_attention', 'volume_mlp']
    matrix = np.array([[b['delta_ce'] for b in rows[n]['position_bins']] for n in heat_names])
    limit = max(abs(matrix).max(), 1e-6)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(len(heat_names)), ['Depth Q1 · blocks 1–4', 'Depth Q2 · blocks 5–8',
        'Depth Q3 · blocks 9–12', 'Depth Q4 · blocks 13–16', 'Attention interfaces', 'MLP interfaces'])
    ax.set_xticks(range(4), ['1–256', '257–512', '513–768', '769–1023'])
    for i, name in enumerate(heat_names):
        for j in range(4):
            bounds = rows[name]['position_bins'][j]['delta_ce_ci95']
            star = '*' if bounds[0] > 0 or bounds[1] < 0 else ''
            ax.text(j, i, f'{matrix[i, j]:+.4f}{star}', ha='center', va='center',
                    color='white' if abs(matrix[i, j]) > .58 * limit else '#222', fontsize=10)
    ax.set_xlabel('Available prefix length'); ax.set_title('C. Causal localization · ΔCE (* 95% CI excludes 0)')
    fig.colorbar(im, ax=ax, shrink=.8, label='Δ next-token CE')
    ax = fig.add_subplot(grid[1, 1])
    profile = np.asarray(calibration['position_bin_volume_mean'])
    lines = np.arange(1, 17)
    ax.plot(lines, profile[::2].mean(1), 'o-', color='#24798e', label='Attention · calibration mean')
    ax.plot(lines, profile[1::2].mean(1), 'o-', color='#c84d2d', label='MLP · calibration mean')
    # Global means weight the final bin's 255 predictions correctly.
    global_mean = np.asarray(calibration['volume_mean'])
    ax.lines[0].set_ydata(global_mean[::2]); ax.lines[1].set_ydata(global_mean[1::2])
    for offset, color in ((0, '#24798e'), (1, '#c84d2d')):
        ax.fill_between(lines, profile[offset::2].min(1), profile[offset::2].max(1), color=color, alpha=.15)
    ax.set_xticks([1, 4, 8, 12, 16]); ax.set_ylim(0, 4)
    ax.set_xlabel('Transformer block'); ax.set_ylabel('Physical volume V = g₀ + g₁')
    ax.set_title('D. Learned update amplitudes · calibration only')
    ax.legend(fontsize=9)
    fig.suptitle('Final AR mHC-lite n=2 · volume-only interventions · 1024 held-out sequences', fontsize=15)
    path = ROOT / 'ar_volume_n2.png'
    fig.savefig(path, dpi=170, facecolor='white'); fig.savefig(OUTPUT / 'causal_figure.svg'); plt.close(fig)
    return path


def claims(result):
    rows = {r['condition']: r for r in result['rows']}
    intact, baseline, global_mean = (rows[n] for n in ('dynamic', 'baseline', 'volume_global'))
    gain = result['baseline_minus_n2_gain']
    point = interval_text(global_mean['delta_ce'], global_mean['delta_ce_ci95'])
    gain_text = f"; потеря {global_mean['gain_lost_pct']:.1f}% выигрыша n=2 над baseline" if result['gain_ratio_stable'] else '; доля выигрыша не определена, поскольку положительный gain неустойчив'
    top = max(('Q1', 'Q2', 'Q3', 'Q4'), key=lambda q: rows['volume_' + q]['delta_ce'])
    strong = global_mean['delta_ce_ci95'][0] > 0
    adjective = 'причинно важна' if strong else 'не показала устойчивого ухудшения в этом вмешательстве'
    comparison = result['position_control_comparisons']['volume_position']
    rescue = comparison['rescue_of_global_volume_damage_pct']
    rescue_text = f'{rescue:.1f}%' if rescue is not None else 'неустойчивая доля'
    sentences = [
        f"На финальных AR checkpoint’ах (шаг 53 399) next-token CE составляет {intact['ce']:.6f} для n=2 и {baseline['ce']:.6f} для baseline; baseline−n=2 = {interval_text(gain, result['gain_ci95'])}.",
        f"Динамическая вариация физического write volume {adjective}: замена только V на независимый per-interface calibration mean даёт ΔCE {point}{gain_text}.",
        f"Средние для каждой точной позиции дают ΔCE {interval_text(rows['volume_position']['delta_ce'], rows['volume_position']['delta_ce_ci95'])} и снимают {rescue_text} ущерба от глобального среднего; этот контроль сохраняет фиксированный позиционный профиль, но убирает зависимость объёма от конкретного префикса.",
        f"Наибольший отдельный эффект среди depth quartiles наблюдается в {top}: ΔCE {interval_text(rows['volume_' + top]['delta_ce'], rows['volume_' + top]['delta_ce_ci95'])}; эффекты attention и MLP равны {rows['volume_attention']['delta_ce']:+.5f} и {rows['volume_mlp']['delta_ce']:+.5f}, а joint−sum(depth quartiles) = {interval_text(result['depth_nonadditivity']['joint_minus_sum_quartile_delta_ce'], result['depth_nonadditivity']['ci95'])}, поэтому отдельные depth effects не дают аддитивную атрибуцию.",
        f"При глобальном усреднении V accuracy меняется на {100 * global_mean['delta_accuracy']:+.3f} п.п., target logit на {global_mean['delta_target_logit']:+.4f}, entropy на {global_mean['delta_entropy']:+.4f}; эти метрики описывают результат вмешательства, а не устанавливают, что router оценивает неопределённость.",
        'Архитектурно V регулирует силу нового branch update относительно переносимого residual: для doubly stochastic mixer среднее двух потоков получает (V/2)·δ, поэтому эффект не сводится к общему масштабированию финальных logits.',
        'Вывод относится к динамике volume на этих frozen AR моделях и teacher-forced held-out next-token задаче; CI условны на calibration split, а механизм принятия решений router и качество свободной генерации здесь не проверялись.'
    ]
    return sentences


def render(config):
    result = summarize(config)
    calibration = json.loads((OUTPUT / 'calibration_summary.json').read_text())
    result['claims_ru'] = claims(result)
    write_json(OUTPUT / 'final_results.json', result)
    image = figure(result, calibration)
    csv_path = ROOT / 'ar_volume_n2.csv'
    fields = ['condition', 'ce', 'delta_ce', 'delta_ce_ci_low', 'delta_ce_ci_high',
              'gain_lost_pct', 'gain_lost_pct_ci_low', 'gain_lost_pct_ci_high',
              'delta_ce_baseline', 'delta_ce_baseline_ci_low', 'delta_ce_baseline_ci_high',
              'accuracy', 'delta_accuracy', 'delta_accuracy_ci_low', 'delta_accuracy_ci_high',
              'target_logit', 'delta_target_logit', 'delta_target_logit_ci_low', 'delta_target_logit_ci_high',
              'entropy', 'delta_entropy', 'delta_entropy_ci_low', 'delta_entropy_ci_high', 'target_margin']
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fields); writer.writeheader()
        for row in result['rows']:
            compact = {k: row[k] for k in fields if k in row}
            for metric in ('delta_ce', 'gain_lost_pct', 'delta_ce_baseline', 'delta_accuracy', 'delta_target_logit', 'delta_entropy'):
                bounds = row[metric + '_ci95']
                compact[metric + '_ci_low'], compact[metric + '_ci_high'] = bounds if bounds else (None, None)
            writer.writerow(compact)
    rows_html = []
    for row in result['rows']:
        fraction = '—' if row['gain_lost_pct'] is None else interval_text(row['gain_lost_pct'], row['gain_lost_pct_ci95'], 1)
        cells = [html.escape(row['label']), f"{row['ce']:.6f}",
                 interval_text(row['delta_ce'], row['delta_ce_ci95']), fraction,
                 interval_text(row['delta_ce_baseline'], row['delta_ce_baseline_ci95']),
                 f"{100 * row['accuracy']:.3f}%<br><small>Δ {100 * row['delta_accuracy']:+.3f} п.п.</small>",
                 f"{row['target_logit']:.4f}<br><small>Δ {row['delta_target_logit']:+.4f}</small>",
                 f"{row['entropy']:.4f}<br><small>Δ {row['delta_entropy']:+.4f}</small>"]
        rows_html.append('<tr>' + ''.join('<td>' + c + '</td>' for c in cells) + '</tr>')
    claim_html = ''.join('<p>' + html.escape(c) + '</p>' for c in result['claims_ru'])
    units = json.loads((OUTPUT / 'unit_checks.json').read_text())
    v_means = np.asarray(calibration['volume_mean'])
    update_ratios = np.asarray(calibration['relative_mean_write_norm_median'])
    payload = base64.b64encode(image.read_bytes()).decode()
    report = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>AR n=2 — роль write volume</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#243344;background:#fafbfc}}h1,h2{{line-height:1.2}}.lead{{font-size:19px}}img{{width:100%;background:white;border:1px solid #ddd}}table{{border-collapse:collapse;background:white;font-size:13px;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dde3e7;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}small{{color:#62727d}}.scroll{{overflow-x:auto}}code,pre{{background:#edf1f4;padding:3px 6px}}pre{{white-space:pre-wrap}}.card{{padding:18px;background:white;border:1px solid #dde3e7;margin:18px 0}}a{{color:#24798e}}</style>
<h1>AR mHC-lite n=2: насколько важна «громкость»</h1>
<p class="lead">Volume-only causal interventions на финальных checkpoint’ах, шаг 53 399. 1024 held-out sequences × 1023 next-token targets = {result['next_token_targets']:,} предсказаний.</p>
<div class="card">{claim_html}</div>
<h2>Итоговая таблица</h2><p>CE в nats; в квадратных скобках — paired bootstrap 95% CI по 1024 последовательностям (5000 resamples). Доля потерянного gain = 100·(CE_condition−CE_dynamic)/(CE_baseline−CE_dynamic), без обрезания на 0–100%. Accuracy, target logit и entropy рассчитаны на тех же shifted targets; полные CI их изменений есть в CSV/JSON.</p>
<div class="scroll"><table><thead><tr><th>Условие</th><th>CE ↓</th><th>ΔCE vs intact [95% CI]</th><th>Потеря gain, % [CI]</th><th>ΔCE vs baseline [95% CI]</th><th>Accuracy</th><th>Target logit</th><th>Entropy</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>
<h2>Причинные эффекты и локализация</h2><img alt="AR volume intervention effects, context and depth localization" src="data:image/png;base64,{payload}">
<p>A–C: held-out causal effects. D: независимая calibration split; линия — token-weighted mean V, заливка — диапазон четырёх средних по четвертям контекста. В heatmap звёздочка означает marginal 95% CI без нуля, без поправки на множественные сравнения. Длина префикса не равна расстоянию от последнего EOS: оценка использует исходную AR recipe на packed blocks.</p>
<p>Средние V по interfaces лежат в диапазоне {v_means.min():.4f}–{v_means.max():.4f}. Однако V нельзя читать как полную энергию записи или сравнивать напрямую между слоями: важен также norm текущего δ. Calibration medians для <code>||(V/2)δ|| / ||mean(h)||</code> лежат в диапазоне {update_ratios.min():.3f}–{update_ratios.max():.3f}; даже малый ранний V может соответствовать крупному обновлению относительно residual. Эти описательные нормы не объясняют причинный driver router.</p>
<h2>Что именно делает volume</h2><div class="card"><p>Для каждого interface <code>h′ᵢ = Σⱼ Rᵢⱼ hⱼ + gᵢ δ</code>, где δ — output текущего attention/MLP branch. Поскольку R doubly stochastic, <code>mean(h′) = mean(h) + (V/2) δ</code>. Volume меняет вес нового обновления по отношению к накопленному состоянию; последующие attention и routers получают изменённое состояние. Общий scale состояния затем нормируется RMSNorm, но относительная смесь residual и нового δ меняется.</p>
<p>Независимые coordinates при gains в [0,2]: <code>V=g₀+g₁; L=max(0,V−2); U=min(2,V); qᵢ=(gᵢ−L)/(U−L); g′ᵢ=L(V′)+(U(V′)−L(V′))qᵢ</code>. q — допустимый destination, а не обычная доля gᵢ/V. При фиксированном q общий объём также меняет допустимый масштаб контраста записей: <code>g′₀−g′₁=(U−L)(q₀−q₁)</code>. Этот audit оценивает заранее согласованный bounded physical factor.</p></div>
<h2>Протокол и проверки</h2><p>64 calibration blocks 128–191; 1024 test blocks 1024–2047, все в validation partition FineWeb-Edu. Полностью заполненные 1024-token blocks, padding отсутствует. Входы не изменяются и не маскируются; source position t оценивается по target ids[t+1]. Вмешательство выполняется на source positions 0–1022, последний source не оценивается. Все means накоплены в float64 на intact AR n=2; остальные outputs каждого текущего router динамические. Global mean — отдельно для каждого из 32 interfaces. Context mean — четыре prefix-length bins; exact-position mean — отдельно для каждой из 1023 позиций. Depth Q1–Q4 включают по четыре блока / восемь interfaces; attention/MLP условия затрагивают по 16 interfaces.</p>
<p>Intact instrumented forward совпадает с оригиналом в FP32/BF16: ошибки {html.escape(str(units['original_forward_error']))}. Изменение будущих input tokens сохраняет все предыдущие states точно: {html.escape(str(units['future_prefix_error']))}. Shifted CE сверена с исходными logits и targets; untouched read/mixer и unselected gains совпадают точно. Проверены [0,2] gains, simplex q и восстановление V; gains не clipping’уются. Все условия используют одинаковые ids и frozen model head. Новые модели, probes, regressions и lenses не запускались.</p>
<p>Bootstrap оценивает вариативность held-out sequences при фиксированных checkpoint’ах и calibration means, не вариативность обучения. Teacher-forced CE не проверяет качество свободной генерации. Отдельные depth effects не являются аддитивной декомпозицией. Усреднение V проверяет необходимость его динамической вариации, а не необходимость ненулевой записи вообще.</p>
<p>Большая доля потерянного gain характеризует чувствительность уже обученного n=2 к вмешательству: learned branches и другие routers остаются совместно адаптированными к динамическому V. Например, потеря около 1906% означает ущерб примерно в 19 раз больше исходного небольшого преимущества над baseline; это не означает, что volume «объясняет 1906%» межмодельного выигрыша или что модель, обученная с фиксированным V, дала бы такой же результат. В AR вход не имеет MASKED/VISIBLE ролей, поэтому данный эффект требует вариации внутри полностью видимого префикса.</p>
<p>Сравнение с MDLM: в предыдущем masked-only audit усреднение V при r=0.5 давало +0.06836 CE и потерю 104.0% gain. AR использует другую задачу и все next-token source positions; абсолютные CE и размеры ΔCE между этими протоколами не являются сопоставлением качества моделей.</p>
<h2>Артефакты</h2><p><a href="ar_volume_n2.csv">CSV</a> · <a href="ar_volume_n2.png">PNG</a> · <a href="ar_volume_n2/final_results.json">JSON</a> · <a href="ar_volume_n2/protocol.json">Protocol</a> · <a href="ar_volume_n2/verification.json">Verification</a></p>
<pre>python -m mechanism_audit_n2.ar_volume --stage full --gpu 5
python -m mechanism_audit_n2.ar_volume --stage render</pre>
</html>'''
    report_path = ROOT / 'ar_volume_n2.html'; report_path.write_text(report)
    verify(result, config)
    (OUTPUT / 'README.md').write_text('''# AR n=2 physical write volume

Frozen existing AR n=2 and matched baseline, final checkpoint step 53399.
No input masking: source t predicts ids[t+1], matching model_and_train.ar_loss.
64 independent calibration sequences; 1024 held-out sequences; float64 per-interface means.
Volume-only interventions preserve current read, bounded destination q, and mixer.

Report: ../ar_volume_n2.html. Table: ../ar_volume_n2.csv. Figure: ../ar_volume_n2.png.
Protocol, unit checks, pack hashes and paired 5000-resample sequence bootstrap are included.
Raw calibration/effect NPZ files stay local and are excluded from git.

```bash
python -m mechanism_audit_n2.ar_volume --stage full --gpu 5
python -m mechanism_audit_n2.ar_volume --stage render
```
''')
    print('AR REPORT WRITTEN', report_path, flush=True)


def verify(result, config):
    assert len(result['rows']) == len(CONDITIONS)
    assert result['next_token_targets'] == 1047552
    units = json.loads((OUTPUT / 'unit_checks.json').read_text())
    assert all(x == 0 for x in units['original_forward_error'].values())
    assert all(x == 0 for x in units['dynamic_hook_error'].values())
    assert all(x == 0 for x in units['future_prefix_error'].values())
    assert units['unchanged_outputs_exact'] and units['unselected_gains_exact']
    captured = load_pack(OUTPUT / 'calibration.npz')
    volume = captured['volume']
    assert volume.dtype == np.float64 and volume.shape == (32, 64, 1023)
    summary = json.loads((OUTPUT / 'calibration_summary.json').read_text())
    expected_means = {'global': volume.mean((1, 2), dtype=np.float64)[:, None],
        'position': volume.mean(1, dtype=np.float64)[..., None],
        'context_quartile': np.stack([volume[..., lo:hi].mean((1, 2), dtype=np.float64)
                                     for lo, hi in POSITION_BINS], axis=1)[..., None]}
    for policy, mean in expected_means.items():
        np.testing.assert_array_equal(mean, np.asarray(summary['means'][policy]))
        assert np.all((mean >= 0) & (mean <= 4))
    for condition in CONDITIONS:
        execution = json.loads((OUTPUT / (condition + '_execution.json')).read_text())
        assert execution['checkpoint_step'] == 53399
        assert max(execution['invariants'].values()) <= 1e-4
        assert abs(execution['metrics']['ce'] - next(r['ce'] for r in result['rows'] if r['condition'] == condition)) < 1e-10
    csv_rows = list(csv.DictReader((ROOT / 'ar_volume_n2.csv').open()))
    assert [r['condition'] for r in csv_rows] == list(CONDITIONS)
    for left, right in zip(csv_rows, result['rows']):
        for field in ('ce', 'delta_ce', 'delta_ce_baseline', 'accuracy', 'target_logit', 'entropy'):
            assert float(left[field]) == right[field]
    assert sources() == result['protocol']['computation_sha256']
    for path, expected in config['source_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    reports = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (ROOT / 'ar_volume_n2.html', ROOT / 'ar_volume_n2.csv', ROOT / 'ar_volume_n2.png')}
    write_json(OUTPUT / 'verification.json', {'passed': True, 'conditions': len(CONDITIONS),
        'sequences_per_condition': 1024, 'next_token_targets_per_condition': 1047552,
        'shifted_targets': True, 'uncorrupted_input': True, 'all_frozen_sources_unchanged': True,
        'intact_original_exact_fp32_bf16': True, 'future_prefix_causality_exact': True,
        'csv_json_consistent': True, 'float64_calibration_means_recomputed_exactly': True,
        'source_sha256': sources(), 'artifact_sha256': reports,
        'report_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'calibration_capture_sha256': hashlib.sha256((OUTPUT / 'calibration.npz').read_bytes()).hexdigest()})
