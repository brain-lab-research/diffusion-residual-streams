"""AR volume comparison under a common odds-simplex factorization."""
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

from .ar_volume_streams import (FULL_CONDITIONS, MAIN_CONDITIONS, OUTPUT,
                                POSITION_BINS, ROOT, sources)
from .primary import load_pack, write_json

LABELS = {'dynamic': 'Intact', 'volume_global': 'V · interface mean',
    'volume_context_quartile': 'V · context quartile mean', 'volume_position': 'V · exact position mean',
    'volume_Q1': 'V mean · depth Q1', 'volume_Q2': 'V mean · depth Q2',
    'volume_Q3': 'V mean · depth Q3', 'volume_Q4': 'V mean · depth Q4',
    'volume_attention': 'V mean · attention only', 'volume_mlp': 'V mean · MLP only'}
METRICS = {'ce': 0, 'target_logit': 1, 'entropy': 2, 'accuracy': 5, 'target_margin': 6}


def ci(value): return np.quantile(value, [.025, .975]).tolist()


def interval(point, bounds, digits=5):
    return f'{point:+.{digits}f} [{bounds[0]:+.{digits}f}, {bounds[1]:+.{digits}f}]'


def sequence_stats(path, test_blocks):
    pack = load_pack(path)
    np.testing.assert_array_equal(pack['block_ids'], test_blocks)
    assert np.all(pack['seq_mask_count'] == 1023)
    assert pack['token_stats'].shape == (1024 * 1023, 8) and np.isfinite(pack['token_stats']).all()
    tokens = pack['token_stats'].reshape(1024, 1023, 8)
    means = tokens.mean(1, dtype=np.float64)
    np.testing.assert_allclose(means[:, 0], pack['seq_ce_sum'] / 1023, atol=1e-12, rtol=0)
    bins = np.stack([tokens[:, lo:hi, 0].mean(1, dtype=np.float64) for lo, hi in POSITION_BINS], axis=1)
    return means, bins, hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(config):
    protocol = json.loads((OUTPUT / 'protocol.json').read_text()); assert protocol['computation_sha256'] == sources()
    draws = np.random.default_rng(config['bootstrap_seed']).integers(1024, size=(config['bootstrap_repeats'], 1024))
    blocks = np.asarray(protocol['test_blocks'])
    baseline, baseline_bins, baseline_hash = sequence_stats(ROOT / 'ar_volume_n2/effects/baseline.npz', blocks)
    assert baseline_hash == protocol['baseline_pack_sha256']
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
        'baseline': {name: float(baseline[:, index].mean()) for name, index in METRICS.items()},
        'models': {}, 'rows': [], 'cross_model_effect_comparisons': {}, 'pack_sha256': {'baseline': baseline_hash},
        'bootstrap': {'unit': 'Paired sequence, shared resamples across models and conditions',
            'seed': config['bootstrap_seed'], 'repeats': config['bootstrap_repeats'],
            'interval': 'Percentile 95%, conditional on fixed checkpoints and calibration means.'}}
    global_deltas = {}
    for n in (2, 3, 4):
        output = OUTPUT / ('n' + str(n)); conditions = MAIN_CONDITIONS if n == 2 else FULL_CONDITIONS
        seq, bins = {}, {}
        for condition in conditions:
            seq[condition], bins[condition], checksum = sequence_stats(output / 'effects' / (condition + '.npz'), blocks)
            result['pack_sha256'][f'n{n}/{condition}'] = checksum
        dynamic = seq['dynamic']; gain_seq = baseline[:, 0] - dynamic[:, 0]
        gain_draws = gain_seq[draws].mean(1); gain = float(gain_seq.mean())
        stable = gain > 0 and bool(np.all(gain_draws > 1e-12))
        model = {'streams': n, 'gain': gain, 'gain_ci95': ci(gain_draws), 'gain_ratio_stable': stable,
                 'rows': [], 'position_rescue': {}}
        for condition in conditions:
            left = seq[condition]
            row = {'streams': n, 'condition': condition, 'label': LABELS[condition]}
            for name, index in METRICS.items():
                difference = left[:, index] - dynamic[:, index]
                row[name] = float(left[:, index].mean()); row['delta_' + name] = float(difference.mean())
                row['delta_' + name + '_ci95'] = ci(difference[draws].mean(1))
            difference = left[:, 0] - baseline[:, 0]
            row['delta_ce_baseline'] = float(difference.mean()); row['delta_ce_baseline_ci95'] = ci(difference[draws].mean(1))
            delta = left[:, 0] - dynamic[:, 0]
            row['gain_lost_pct'] = 100 * float(delta.mean()) / gain if stable else None
            row['gain_lost_pct_ci95'] = ci(100 * delta[draws].mean(1) / gain_draws) if stable else None
            row['position_bins'] = []
            for k, (lo, hi) in enumerate(POSITION_BINS):
                difference = bins[condition][:, k] - bins['dynamic'][:, k]
                row['position_bins'].append({'prefix_lengths': [lo + 1, hi], 'tokens': 1024 * (hi - lo),
                    'ce': float(bins[condition][:, k].mean()), 'delta_ce': float(difference.mean()),
                    'delta_ce_ci95': ci(difference[draws].mean(1))})
            model['rows'].append(row); result['rows'].append(row)
        global_delta = seq['volume_global'][:, 0] - dynamic[:, 0]; global_deltas[n] = global_delta
        divisor = global_delta[draws].mean(1)
        for condition in ('volume_context_quartile', 'volume_position'):
            numerator = seq['volume_global'][:, 0] - seq[condition][:, 0]
            denominator_stable = float(global_delta.mean()) > 0 and bool(np.all(divisor > 1e-12))
            model['position_rescue'][condition] = {'global_minus_condition_ce': float(numerator.mean()),
                'ci95': ci(numerator[draws].mean(1)),
                'rescue_pct': 100 * float(numerator.mean()) / float(global_delta.mean()) if denominator_stable else None,
                'rescue_pct_ci95': ci(100 * numerator[draws].mean(1) / divisor) if denominator_stable else None}
        if n > 2:
            summed = sum(seq['volume_' + q][:, 0] - dynamic[:, 0] for q in ('Q1', 'Q2', 'Q3', 'Q4'))
            interaction = global_delta - summed
            model['depth_nonadditivity'] = {'joint_minus_sum_delta_ce': float(interaction.mean()), 'ci95': ci(interaction[draws].mean(1))}
            branch_interaction = global_delta - (seq['volume_attention'][:, 0] - dynamic[:, 0]) - (seq['volume_mlp'][:, 0] - dynamic[:, 0])
            model['branch_nonadditivity'] = {'joint_minus_attention_minus_mlp_delta_ce': float(branch_interaction.mean()),
                                            'ci95': ci(branch_interaction[draws].mean(1))}
        result['models'][str(n)] = model
    for larger, smaller in ((3, 2), (4, 2), (4, 3)):
        difference = global_deltas[larger] - global_deltas[smaller]
        result['cross_model_effect_comparisons'][f'n{larger}_minus_n{smaller}'] = {
            'difference_in_global_volume_delta_ce': float(difference.mean()), 'ci95': ci(difference[draws].mean(1)),
            'interpretation': 'Matched held-out intervention sensitivities in separately trained checkpoints; not a pure causal effect of stream count.'}
    old = json.loads((ROOT / 'ar_volume_n2/final_results.json').read_text())
    result['prior_n2_bounded_coordinate'] = {'definition': old['protocol']['factorization'],
        'global_delta_ce': next(r['delta_ce'] for r in old['rows'] if r['condition'] == 'volume_global'),
        'exact_position_delta_ce': next(r['delta_ce'] for r in old['rows'] if r['condition'] == 'volume_position'),
        'note': 'Preserved original evidence; headline cross-stream comparisons here use the common odds coordinate.'}
    return result


def make_figure(result):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = {(r['streams'], r['condition']): r for r in result['rows']}
    colors = {2: '#24798e', 3: '#b25883', 4: '#ce6538'}
    fig = plt.figure(figsize=(16, 13), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.5))
    ax = fig.add_subplot(grid[0, 0])
    policies = MAIN_CONDITIONS[1:]; y = np.arange(3)
    for k, policy in enumerate(policies):
        points = np.asarray([rows[n, policy]['delta_ce'] for n in (2, 3, 4)])
        bounds = np.asarray([rows[n, policy]['delta_ce_ci95'] for n in (2, 3, 4)])
        offset = (k - 1) * .23
        ax.barh(y + offset, points, height=.21, color=['#333', '#88969e', '#52a087'][k], label=LABELS[policy])
        ax.errorbar(points, y + offset, xerr=np.maximum(0, np.stack([points - bounds[:, 0], bounds[:, 1] - points])),
                    fmt='none', color='#222', capsize=3)
    gains = [result['models'][str(n)]['gain'] for n in (2, 3, 4)]
    ax.scatter(gains, y, marker='D', color='#dd9b14', s=36, label='Baseline − intact gain', zorder=4)
    ax.set_yticks(y, ['AR n=2 · bridge', 'AR n=3', 'AR n=4']); ax.invert_yaxis()
    ax.set_xlabel('Δ next-token CE vs respective intact model')
    ax.set_title('A. Volume-only interventions · shared odds destination')
    ax.legend(fontsize=9, loc='lower right')
    ax = fig.add_subplot(grid[0, 1]); x = np.arange(4)
    for n in (2, 3, 4):
        for policy, style in (('volume_global', '-'), ('volume_position', '--')):
            values = np.asarray([b['delta_ce'] for b in rows[n, policy]['position_bins']])
            bounds = np.asarray([b['delta_ce_ci95'] for b in rows[n, policy]['position_bins']])
            ax.errorbar(x, values, yerr=np.maximum(0, np.stack([values - bounds[:, 0], bounds[:, 1] - values])),
                color=colors[n], linestyle=style, marker='o', capsize=3,
                label=f'n={n} · ' + ('global mean' if policy == 'volume_global' else 'exact-position mean'))
    ax.axhline(0, color='#555', linewidth=.8); ax.set_xticks(x, ['1–256', '257–512', '513–768', '769–1023'])
    ax.set_xlabel('Available prefix length'); ax.set_ylabel('Δ next-token CE')
    ax.set_title('B. Global vs position-conditioned means'); ax.legend(fontsize=8, ncol=2)
    ax = fig.add_subplot(grid[1, 0])
    local = ('volume_Q1', 'volume_Q2', 'volume_Q3', 'volume_Q4', 'volume_attention', 'volume_mlp')
    order = [(n, c) for n in (3, 4) for c in local]
    matrix = np.asarray([[b['delta_ce'] for b in rows[n, c]['position_bins']] for n, c in order])
    limit = max(float(abs(matrix).max()), 1e-6)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(12), [f'n={n} · ' + LABELS[c].removeprefix('V mean · ') for n, c in order])
    ax.set_xticks(range(4), ['1–256', '257–512', '513–768', '769–1023'])
    ax.axhline(5.5, color='white', linewidth=3)
    for i, (n, c) in enumerate(order):
        for j in range(4):
            low, high = rows[n, c]['position_bins'][j]['delta_ce_ci95']
            star = '*' if low > 0 or high < 0 else ''
            ax.text(j, i, f'{matrix[i, j]:+.4f}{star}', ha='center', va='center',
                color='white' if abs(matrix[i, j]) > .58 * limit else '#222', fontsize=9)
    ax.set_title('C. Causal localization · * paired 95% CI excludes 0')
    ax.set_xlabel('Available prefix length'); fig.colorbar(im, ax=ax, shrink=.75, label='Δ next-token CE')
    ax = fig.add_subplot(grid[1, 1])
    for n in (2, 3, 4):
        means = json.loads((OUTPUT / f'n{n}/calibration_means.json').read_text())
        volume = np.asarray(means['volume_mean']) / n
        ax.plot(range(1, 17), volume[::2], color=colors[n], marker='o', markersize=4, label=f'n={n} · attention')
        ax.plot(range(1, 17), volume[1::2], color=colors[n], marker='s', markersize=4, linestyle='--', label=f'n={n} · MLP')
    ax.set_ylim(0, 2); ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlabel('Transformer block')
    ax.set_ylabel('Calibration mean volume per stream · V/n')
    ax.set_title('D. Learned amplitude profiles · calibration only'); ax.legend(fontsize=9)
    fig.suptitle('Final AR mHC-lite · physical volume interventions · n=2 / n=3 / n=4', fontsize=16)
    path = OUTPUT / 'causal_heatmap.png'; fig.savefig(path, dpi=170, facecolor='white')
    fig.savefig(OUTPUT / 'causal_heatmap.svg'); plt.close(fig)
    return path


def claim_sentences(result):
    models = result['models']; rows = {(r['streams'], r['condition']): r for r in result['rows']}
    sentences = [
        'На финальных AR mHC-lite checkpoints n=2,3,4 проверены frozen volume-only interventions с общей факторизацией V=Σg и destination q=softmax(write logits), при сохранении текущих read и residual mixer.',
        f"На одних и тех же 1024 held-out последовательностях intact CE равен {rows[2, 'dynamic']['ce']:.6f}, {rows[3, 'dynamic']['ce']:.6f} и {rows[4, 'dynamic']['ce']:.6f}, а CE baseline равен {result['baseline']['ce']:.6f}."
    ]
    for n in (3, 4):
        mean = rows[n, 'volume_global']; position = rows[n, 'volume_position']
        rescue = models[str(n)]['position_rescue']['volume_position']['rescue_pct']
        rescue_text = f'{rescue:.2f}%' if rescue is not None else 'неустойчивая доля'
        fraction = f"{mean['gain_lost_pct']:.1f}%" if mean['gain_lost_pct'] is not None else 'не определена'
        sentences.append(f"Для n={n} замена V на per-interface calibration mean даёт ΔCE {interval(mean['delta_ce'], mean['delta_ce_ci95'])}, ΔCE относительно baseline {interval(mean['delta_ce_baseline'], mean['delta_ce_baseline_ci95'])} и потерю gain {fraction}; exact-position means дают ΔCE {interval(position['delta_ce'], position['delta_ce_ci95'])} и снимают {rescue_text} ущерба.")
    local = []
    for n in (3, 4):
        top = max(('Q1', 'Q2', 'Q3', 'Q4'), key=lambda q: rows[n, 'volume_' + q]['delta_ce'])
        local.append(f"n={n}: {top}, ΔCE {rows[n, 'volume_' + top]['delta_ce']:+.5f}; attention {rows[n, 'volume_attention']['delta_ce']:+.5f}, MLP {rows[n, 'volume_mlp']['delta_ce']:+.5f}")
    sentences.append('Самый чувствительный depth quartile и тип branch: ' + '; '.join(local) + ', при этом отдельные эффекты не являются аддитивной декомпозицией.')
    sentences.append('Для doubly stochastic mixer mean(h′)=mean(h)+(V/n)·δ, поэтому volume регулирует силу branch update относительно переносимого residual; эти вмешательства проверяют необходимость динамической вариации V и не устанавливают, что router оценивает неопределённость.')
    sentences.append('Сравнение n использует отдельно обученные checkpoints и общий odds-контроль, а большие доли потерянного gain описывают чувствительность frozen моделей к вмешательству; CI условны на calibration split, качество свободной генерации и фиксированный-volume training здесь не проверены.')
    assert len(sentences) == 7
    return sentences


def render(config):
    result = summarize(config); result['claims_ru'] = claim_sentences(result)
    write_json(OUTPUT / 'final_results.json', result)
    figure = make_figure(result)
    fields = ['streams', 'condition', 'ce', 'delta_ce', 'delta_ce_ci_low', 'delta_ce_ci_high',
        'gain_lost_pct', 'gain_lost_pct_ci_low', 'gain_lost_pct_ci_high',
        'delta_ce_baseline', 'delta_ce_baseline_ci_low', 'delta_ce_baseline_ci_high',
        'accuracy', 'delta_accuracy', 'delta_accuracy_ci_low', 'delta_accuracy_ci_high',
        'target_logit', 'delta_target_logit', 'delta_target_logit_ci_low', 'delta_target_logit_ci_high',
        'entropy', 'delta_entropy', 'delta_entropy_ci_low', 'delta_entropy_ci_high', 'target_margin']
    csv_path = OUTPUT / 'final_table.csv'
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fields); writer.writeheader()
        for row in result['rows']:
            output = {key: row[key] for key in fields if key in row}
            for name in ('delta_ce', 'gain_lost_pct', 'delta_ce_baseline', 'delta_accuracy', 'delta_target_logit', 'delta_entropy'):
                output[name + '_ci_low'], output[name + '_ci_high'] = row[name + '_ci95'] or (None, None)
            writer.writerow(output)
    table_rows = []
    for row in result['rows']:
        percent = '—' if row['gain_lost_pct'] is None else interval(row['gain_lost_pct'], row['gain_lost_pct_ci95'], 1)
        cells = [str(row['streams']), html.escape(row['label']), f"{row['ce']:.6f}",
            interval(row['delta_ce'], row['delta_ce_ci95']), percent,
            interval(row['delta_ce_baseline'], row['delta_ce_baseline_ci95']),
            f"{100 * row['accuracy']:.3f}%<br><small>Δ {100 * row['delta_accuracy']:+.3f} п.п.</small>",
            f"{row['target_logit']:.4f}<br><small>Δ {row['delta_target_logit']:+.4f}</small>",
            f"{row['entropy']:.4f}<br><small>Δ {row['delta_entropy']:+.4f}</small>"]
        table_rows.append('<tr>' + ''.join('<td>' + c + '</td>' for c in cells) + '</tr>')
    sentences = ''.join('<p>' + html.escape(c) + '</p>' for c in result['claims_ru'])
    image_data = base64.b64encode(figure.read_bytes()).decode()
    comparison_text = ''.join('<p>' + html.escape(name + ': ' + interval(
        entry['difference_in_global_volume_delta_ce'], entry['ci95'])) + '</p>'
        for name, entry in result['cross_model_effect_comparisons'].items())
    prior = result['prior_n2_bounded_coordinate']
    interaction_text = ''.join('<p>' + html.escape(f"n={n}: joint−sum(depth quartiles) = " + interval(
        result['models'][str(n)]['depth_nonadditivity']['joint_minus_sum_delta_ce'],
        result['models'][str(n)]['depth_nonadditivity']['ci95']) + '; joint−attention−MLP = ' + interval(
        result['models'][str(n)]['branch_nonadditivity']['joint_minus_attention_minus_mlp_delta_ce'],
        result['models'][str(n)]['branch_nonadditivity']['ci95'])) + '</p>' for n in (3, 4))
    report = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>AR n=2/3/4 — causal write volume</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1550px;margin:32px auto;padding:0 24px;color:#243344;background:#fafbfc}}h1,h2{{line-height:1.2}}.lead{{font-size:19px}}img{{width:100%;background:white;border:1px solid #ddd}}table{{border-collapse:collapse;background:white;font-size:13px;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dde3e7;text-align:right;white-space:nowrap}}th:nth-child(2),td:nth-child(2){{text-align:left}}small{{color:#62727d}}.scroll{{overflow-x:auto}}code,pre{{background:#edf1f4;padding:3px 6px}}pre{{white-space:pre-wrap}}.card{{padding:18px;background:white;border:1px solid #dde3e7;margin:18px 0}}a{{color:#24798e}}</style>
<h1>AR n=3 и n=4: причинная важность write volume</h1>
<p class="lead">Финальные checkpoints, шаг 53 399. 64 независимые calibration sequences; 1024 общие held-out sequences × 1023 next-token targets. Общий volume/destination контроль для n=2,3,4; новые модели не обучались.</p>
<div class="card">{sentences}</div>
<h2>Итоговая таблица</h2><p>Общий AR baseline: CE <strong>{result['baseline']['ce']:.6f}</strong>, accuracy {100 * result['baseline']['accuracy']:.3f}%, target logit {result['baseline']['target_logit']:.4f}, entropy {result['baseline']['entropy']:.4f}. Каждая ΔCE отсчитывается от собственного intact checkpoint. В квадратных скобках paired bootstrap 95% CI, 5000 общих sequence resamples. Все Δ и gain fractions вычисляются внутри одного resample; ratios подавляются при неустойчивом положительном denominator и не обрезаются на 0–100%. Полные CI accuracy/logit/entropy доступны в CSV/JSON.</p>
<div class="scroll"><table><thead><tr><th>n</th><th>Условие</th><th>CE ↓</th><th>ΔCE vs intact [95% CI]</th><th>Потеря gain, % [CI]</th><th>ΔCE vs baseline [95% CI]</th><th>Accuracy</th><th>Target logit</th><th>Entropy</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Причинные эффекты и локализация</h2><img alt="AR volume interventions and depth-context causal heatmap" src="data:image/png;base64,{image_data}">
<p>A–C — held-out causal effects, D — independent calibration statistics. Depth quartiles включают по четыре блока / восемь interfaces; attention и MLP условия — по 16 interfaces. В heatmap * означает marginal paired 95% CI без нуля, без поправки на множественные сравнения. Отдельные depth effects не складываются в атрибуцию общего gain. Для сопоставления learned profiles на D показан V/n, поскольку mean-stream update равен (V/n)δ; V не является полной нормой записи, которая зависит также от δ и residual. Prefix length отсчитывается от начала packed block, а не от последнего EOS.</p>
<h2>Сопоставление чувствительности checkpoints</h2><p>Парные разности ΔCE для общего per-interface mean volume:</p>{comparison_text}
<p>Это сравнение конкретных отдельно обученных моделей при одинаковых данных и вмешательстве. Оно не изолирует число потоков от различий выученных branches и router weights.</p>
<p>Описательные interaction contrasts из уже выполненных условий:</p>{interaction_text}
<h2>Почему для n&gt;2 нужен общий контроль</h2><div class="card"><p>В прежнем n=2 audit использовались V=g₀+g₁ и bounded destination qᵢ=(gᵢ−L)/(U−L). Эта бинарная формула не даёт независимый simplex для n&gt;2. Здесь используется общий вариант: <code>gᵢ=2 sigmoid(zᵢ); V=Σᵢgᵢ; qᵢ=exp(zᵢ)/Σⱼexp(zⱼ)</code>, то есть simplex нормированных gain odds <code>gᵢ/(2−gᵢ)</code>. Для заданного calibration V′ находится единственный common shift c с <code>Σᵢ2 sigmoid(zᵢ+c)=V′</code>. Все относительные logits и q сохраняются, gains принадлежат [0,2] без clipping. Read и residual mixer берутся из текущего router без изменений; subsequent routers продолжают реагировать на изменённое состояние.</p>
<p>Воспроизведён n=2 bridge под этим же определением. Его intact per-token outputs совпали с предыдущим прогоном точно. Старый bounded n=2 отчёт сохранён: global ΔCE {prior['global_delta_ce']:+.5f}, exact-position ΔCE {prior['exact_position_delta_ce']:+.5f}. Численные межмодельные сравнения выше используют общий odds coordinate; его q отличается от прежнего bounded q и от обычных долей g/V.</p>
<p>Архитектурно <code>h′ᵢ=ΣⱼRᵢⱼhⱼ+gᵢδ</code>; doubly stochastic R сохраняет stream mean, поэтому <code>mean(h′)=mean(h)+(V/n)δ</code>. Volume меняет вес нового attention/MLP обновления относительно накопленного состояния. Причинная важность динамики этого веса не устанавливает признаки, по которым router его выбирает.</p></div>
<h2>Протокол и проверки</h2><p>Final AR n=3: ar36_lit_s3/ckpt_53399.pt; n=4: ar36_lit/ckpt_53399.pt; n=2 bridge: ar36_lit_s2/ckpt_53399.pt. Matched baseline ar36_base/ckpt_53399.pt повторно использует sealed outputs на тех же ids. Calibration blocks 128–191 и test blocks 1024–2047 лежат в validation partition FineWeb-Edu и не пересекаются; dense full 1024-token blocks, padding отсутствует.</p>
<p>Входы не маскируются и не меняются. Source t предсказывает исходный ids[t+1]; вмешательство на sources 0–1022, последний source не оценивается. Float64 token-weighted means рассчитаны отдельно для каждого из 32 interfaces: global; четыре prefix-length quartiles; каждая из 1023 точных позиций. Для n=3/4 выполнены также четыре depth quartiles и attention/MLP interventions.</p>
<p>Solver с аналитической float64 скобкой и 40 bisections проверен против отдельного 90-step PyTorch float64 reference, на extreme volumes 0 и 2n и на stream permutations. Instrumented intact forward совпадает с original в FP32/BF16; causal prefix и shifted CE проверены отдельно. Неизменяемые outputs и unselected gains совпадают точно; каждый batch проверяет gain bounds, V reconstruction, odds-simplex preservation и повторное получение original write logits. Computation и frozen source hashes, effect-pack hashes и проверка CSV/JSON сохранены.</p>
<p>Доля потерянного gain = 100·(CE_condition−CE_intact)/(CE_baseline−CE_intact). Большие значения отражают чувствительность совместно обученных branches/routers к вмешательству и не являются долей причин межмодельного улучшения. Усреднение V проверяет динамическую вариацию объёма, а не необходимость ненулевого write или качество модели, заново обученной с фиксированным V. Bootstrap CI условны на checkpoint и calibration split; teacher-forced оценка не измеряет свободную генерацию. В AR отсутствует MASKED/VISIBLE разделение входа.</p>
<p><a href="final_table.csv">CSV</a> · <a href="causal_heatmap.png">PNG</a> · <a href="causal_heatmap.svg">SVG</a> · <a href="final_results.json">JSON</a> · <a href="protocol.json">Protocol</a> · <a href="verification.json">Verification</a> · <a href="../ar_volume_n2.html">Предыдущий n=2 audit</a></p>
<pre>python -m mechanism_audit_n2.ar_volume_streams --stage full --gpu 5
python -m mechanism_audit_n2.ar_volume_streams --stage render</pre>
</html>'''
    (OUTPUT / 'report.html').write_text(report)
    verify(result, config)
    (OUTPUT / 'README.md').write_text('''# AR physical write volume: n=3 / n=4 + n=2 bridge

Final frozen AR checkpoints, step 53399; 64 calibration and 1024 held-out sequences.
Uncorrupted inputs, shifted next-token targets. Common coordinates V=sum(g),
q=softmax(write logits). A float64 common-logit shift fixes V and preserves q,
current read, and current residual mixer. No gain clipping or new training.

report.html is standalone. final_table.csv / final_results.json contain paired
5000-resample sequence-bootstrap CI; causal_heatmap.png and SVG localize effects.
The previous bounded binary n=2 report is preserved; comparisons here use a
fresh n=2 odds bridge. Raw per-token NPZ files stay local and are git-ignored.

```bash
python -m mechanism_audit_n2.ar_volume_streams --stage full --gpu 5
python -m mechanism_audit_n2.ar_volume_streams --stage render
```
''')
    print('STREAMS REPORT WRITTEN', OUTPUT / 'report.html', flush=True)


def verify(result, config):
    assert len(result['rows']) == 24
    assert sources() == result['protocol']['computation_sha256']
    solver = json.loads((OUTPUT / 'solver_checks.json').read_text())
    assert all(max(item.values()) <= 1e-6 for item in solver.values())
    for n in (2, 3, 4):
        output = OUTPUT / f'n{n}'; units = json.loads((output / 'unit_checks.json').read_text())
        assert all(x == 0 for x in units['original_forward_error'].values())
        assert all(x == 0 for x in units['future_prefix_error'].values())
        assert units['untouched_outputs_exact'] and units['unselected_gains_exact']
        assert max(units['maximum_errors'].values()) <= 1e-6
        captured = load_pack(output / 'calibration.npz'); volume = captured['volume']
        assert volume.dtype == np.float64 and volume.shape == (32, 64, 1023)
        summary = json.loads((output / 'calibration_means.json').read_text())
        expected = {'global': volume.mean((1, 2), dtype=np.float64)[:, None],
            'position': volume.mean(1, dtype=np.float64)[..., None],
            'context_quartile': np.stack([volume[..., lo:hi].mean((1, 2), dtype=np.float64)
                                         for lo, hi in POSITION_BINS], axis=1)[..., None]}
        for name, mean in expected.items():
            np.testing.assert_array_equal(mean, np.asarray(summary['means'][name]))
            assert np.all((mean > 0) & (mean < 2 * n))
        for row in result['models'][str(n)]['rows']:
            execution = json.loads((output / (row['condition'] + '_execution.json')).read_text())
            assert execution['checkpoint_step'] == 53399 and execution['streams'] == n
            assert max(execution['maximum_errors'].values()) <= 1e-6
            assert abs(execution['metrics']['ce'] - row['ce']) < 1e-10
    for path, checksum in config['source_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
    rows = list(csv.DictReader((OUTPUT / 'final_table.csv').open()))
    assert len(rows) == 24
    for left, right in zip(rows, result['rows']):
        assert int(left['streams']) == right['streams'] and left['condition'] == right['condition']
        for field in ('ce', 'delta_ce', 'delta_ce_baseline', 'accuracy', 'target_logit', 'entropy'):
            assert float(left[field]) == right[field]
    previous = load_pack(ROOT / 'ar_volume_n2/effects/dynamic.npz')
    fresh = load_pack(OUTPUT / 'n2/effects/dynamic.npz')
    np.testing.assert_array_equal(previous['token_stats'], fresh['token_stats'])
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (OUTPUT / 'report.html', OUTPUT / 'final_table.csv', OUTPUT / 'causal_heatmap.png')}
    write_json(OUTPUT / 'verification.json', {'passed': True, 'inference_conditions': 24,
        'sequences_per_condition': 1024, 'next_token_targets_per_condition': 1047552,
        'float64_calibration_recomputed_exactly': True, 'n2_intact_matches_previous_exactly': True,
        'same_factorization_for_cross_stream_comparison': True, 'baseline_pack_checksum_sealed': True,
        'original_forward_fp32_bf16_exact': True, 'future_prefix_causality_exact': True,
        'csv_json_consistent': True, 'all_frozen_sources_unchanged': True,
        'computation_sha256': sources(), 'report_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifact_sha256': artifacts})
