"""Layer-by-layer atlas of what the two residual streams do in the frozen n=2 model.

Consolidates three independent measurement families that already exist on disk:

  routing   residual_circuits_n2/captures/r*_all_positions.npz  (read / write / mixer
            coefficients per interface, five corruption levels)
  geometry  side_retrieval_n2/side_drift.json                   (per-step cosine, norms,
            cosine to the early references, t=.5)
  causal    side_retrieval_n2/final_results.json                (intervention CE + CI)
            side_retrieval_n2/side_identity.json                (patch at the parked boundary)

python -m mechanism_audit_n2.stream_atlas
"""
from __future__ import annotations

import csv
from html import escape
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .factors import ROOT

OUTPUT = ROOT / "side_retrieval_n2"
CAPTURES = ROOT / "residual_circuits_n2" / "captures"
CHECKPOINT = json.loads((ROOT / "residual_circuits_n2" / "config.json").read_text())
INTERFACES = tuple(f"{kind}{block}" for block in range(16) for kind in ("a", "m"))
RATIOS = (.1, .3, .5, .7, .9)
ROLES = ("masked", "visible")
# Router coordinate layout, as assembled in engine.py:
#   [read stream0, read stream1, write gain stream0, write gain stream1,
#    mixer P(identity), mixer P(swap)]
COORDS = {"read_s0": 0, "read_s1": 1, "gain_s0": 2, "gain_s1": 3, "p_identity": 4, "p_swap": 5}

# Phases, as locked in candidate_selection.json before any causal outcome was read.
PHASES = (
    ("write-in", 0, 2, "Содержимое компилируется в side stream."),
    ("hand-off", 2, 4, "Active stream наполняется, side уже зафиксирован."),
    ("parked", 4, 17, "Side почти не читают и не меняют."),
    ("rewrite", 17, 22, "Side переписывается перед поздним чтением."),
    ("late read", 22, 32, "Последние пять блоков читают продукт перезаписи."),
)
# Validated categorical pair (all six checks pass on the light surface).
ROLE_COLOR = {"masked": "#b43c5a", "visible": "#0b86b8"}
STREAM_COLOR = {"active (stream 0)": "#b43c5a", "side (stream 1)": "#0b86b8"}
# Sequential ramp for the ordered corruption levels: one hue, light -> dark.
NOISE_RAMP = ("#a8d5e8", "#6cb4d6", "#3a8fbd", "#1f6a9b", "#12466d")


def load_routing() -> dict:
    """ratio -> role -> metric -> per-interface mean."""
    table = {}
    for ratio in RATIOS:
        key = f"r{round(100 * ratio):02d}"
        with np.load(CAPTURES / f"{key}_all_positions.npz") as saved:
            router = saved["router"].astype(np.float64)
            mask = saved["mask"].reshape(-1)
        table[key] = {}
        for role, selected in (("masked", mask), ("visible", ~mask)):
            mean = router[:, selected, :].mean(1)
            local = {name: mean[:, index] for name, index in COORDS.items()}
            total = local["read_s0"] + local["read_s1"]
            local["side_share"] = local["read_s1"] / np.maximum(total, 1e-12)
            table[key][role] = local
    return table


def phase_of(index: int) -> str:
    for name, begin, end, _ in PHASES:
        if begin <= index < end:
            return name
    raise AssertionError(index)


def write_tables(routing: dict, drift: dict) -> list[dict]:
    rows = []
    for index, label in enumerate(INTERFACES):
        for role in ROLES:
            route = routing["r50"][role]
            geometry = drift["roles"][role]

            def value(key):
                item = geometry[key][index]
                return "" if item is None else round(float(item), 4)

            rows.append({
                "interface": label, "index": index, "phase": phase_of(index), "role": role,
                "read_s0": round(route["read_s0"][index], 4),
                "read_s1": round(route["read_s1"][index], 4),
                "side_read_share": round(route["side_share"][index], 4),
                "gain_s0": round(route["gain_s0"][index], 4),
                "gain_s1": round(route["gain_s1"][index], 4),
                "p_swap": round(route["p_swap"][index], 4),
                "step_cos_s0": value("cos_prev_s0"), "step_cos_s1": value("cos_prev_s1"),
                "norm_s0": value("norm_s0"), "norm_s1": value("norm_s1"),
                "side_cos_a0": value("cos_a0"), "side_cos_m0": value("cos_m0"),
                "cos_s0_s1": value("cos_active"),
            })
    with (OUTPUT / "stream_atlas.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def phase_summary(routing: dict, drift: dict) -> list[dict]:
    summary = []
    for name, begin, end, description in PHASES:
        span = list(range(begin, end))
        entry = {"phase": name, "interfaces": f"{INTERFACES[begin]}–{INTERFACES[end - 1]}",
                 "blocks": end - begin, "description": description, "roles": {}}
        for role in ROLES:
            route = routing["r50"][role]
            geometry = drift["roles"][role]

            def finite(key):
                values = np.array([np.nan if item is None else item
                                   for item in geometry[key]], dtype=float)[span]
                return values[np.isfinite(values)]

            step_side, step_active = finite("cos_prev_s1"), finite("cos_prev_s0")
            entry["roles"][role] = {
                "side_read_share": float(route["side_share"][span].mean()),
                "gain_side": float(route["gain_s1"][span].mean()),
                "gain_active": float(route["gain_s0"][span].mean()),
                "p_swap": float(route["p_swap"][span].mean()),
                "step_cos_side": float(np.median(step_side)) if step_side.size else float("nan"),
                "step_cos_active": float(np.median(step_active)) if step_active.size else float("nan"),
                "norm_side_end": float(geometry["norm_s1"][end - 1]),
                "norm_active_end": float(geometry["norm_s0"][end - 1]),
            }
        summary.append(entry)
    return summary


def bands(axis):
    shades = {"write-in": "#b43c5a", "hand-off": "#8a8f98", "parked": "#1f2430",
              "rewrite": "#d28d27", "late read": "#0b86b8"}
    alphas = {"write-in": .07, "hand-off": .04, "parked": .05, "rewrite": .11, "late read": .06}
    for name, begin, end, _ in PHASES:
        axis.axvspan(begin - .5, end - .5, color=shades[name], alpha=alphas[name], lw=0)


def render_figure(routing: dict, drift: dict):
    x = np.arange(32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), layout="constrained")

    axis = axes[0][0]
    bands(axis)
    for role in ROLES:
        axis.plot(x, routing["r50"][role]["side_share"], lw=2, marker="o", markersize=4,
                  color=ROLE_COLOR[role], label=role, markeredgecolor="white",
                  markeredgewidth=.5)
    axis.set_ylabel("доля чтения из side stream")
    axis.set_title("A · Кого читает router", loc="left", fontweight="bold", pad=32)
    axis.legend(frameon=False, loc="upper left")

    axis = axes[0][1]
    bands(axis)
    for role in ROLES:
        axis.plot(x, routing["r50"][role]["gain_s1"], lw=2, marker="o", markersize=4,
                  color=ROLE_COLOR[role], label=role, markeredgecolor="white",
                  markeredgewidth=.5)
    axis.set_ylabel("write gain в side stream")
    axis.set_title("B · Кто пишет в side stream", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")

    axis = axes[1][0]
    bands(axis)
    geometry = drift["roles"]["masked"]
    for name, key in (("active (stream 0)", "norm_s0"), ("side (stream 1)", "norm_s1")):
        axis.plot(x, geometry[key], lw=2, marker="o", markersize=4, color=STREAM_COLOR[name],
                  label=name, markeredgecolor="white", markeredgewidth=.5)
    axis.set_yscale("log")
    axis.set_ylabel("норма состояния (masked, log)")
    axis.set_title("C · Где лежит содержимое", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")

    axis = axes[1][1]
    bands(axis)
    for color, ratio in zip(NOISE_RAMP, RATIOS):
        key = f"r{round(100 * ratio):02d}"
        axis.plot(x, routing[key]["masked"]["side_share"], lw=2, color=color,
                  label=f"t={ratio:.1f}")
    axis.set_ylabel("доля чтения из side (masked)")
    axis.set_ylim(-.03, 1.12)
    axis.set_title("D · Устойчивость фаз к уровню шума", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper right", ncols=3, fontsize=9)

    for row in axes:
        for axis in row:
            axis.set_xticks(x[::2], [INTERFACES[i] for i in x[::2]], rotation=90, fontsize=7)
            axis.set_xlabel("interface   (a = attention, m = mixer)")
            axis.grid(alpha=.15)
            axis.set_xlim(-.6, 31.6)
    for name, begin, end, _ in PHASES:
        # The two opening phases are only two interfaces wide; stagger so labels never touch.
        height = 1.055 if name == "hand-off" else 1.015
        axes[0][0].text((begin + end - 1) / 2, height, name, ha="center", fontsize=8.5,
                        color="#5b6472", transform=axes[0][0].get_xaxis_transform())
    fig.suptitle("Атлас residual streams по глубине · frozen full-n=2, step "
                 f"{CHECKPOINT['checkpoint_step']}", fontsize=15)
    fig.savefig(OUTPUT / "stream_atlas.svg")
    fig.savefig(OUTPUT / "stream_atlas.png", dpi=170)
    plt.close(fig)


def fmt(value: dict) -> str:
    return f'{value["delta_ce"]:+.4f} [{value["ci95"][0]:+.4f}, {value["ci95"][1]:+.4f}]'


def render_html(routing, drift, rows, summary, causal, identity):
    phase_rows = []
    for entry in summary:
        for position, role in enumerate(ROLES):
            local = entry["roles"][role]
            first = (f'<td rowspan="2">{entry["phase"]}</td>'
                     f'<td rowspan="2">{entry["interfaces"]}</td>'
                     f'<td rowspan="2">{entry["blocks"]}</td>') if position == 0 else ""
            phase_rows.append(
                f"<tr>{first}<td>{role}</td>"
                f'<td>{local["side_read_share"]:.3f}</td>'
                f'<td>{local["gain_side"]:.3f}</td><td>{local["gain_active"]:.3f}</td>'
                f'<td>{local["p_swap"]:.4f}</td>'
                f'<td>{local["step_cos_side"]:.3f}</td><td>{local["step_cos_active"]:.3f}</td>'
                f'<td>{local["norm_side_end"]:.1f}</td><td>{local["norm_active_end"]:.1f}</td></tr>')

    detail_rows = []
    for row in rows:
        detail_rows.append("<tr>" + "".join(
            f"<td>{row[key]}</td>" for key in
            ("interface", "phase", "role", "read_s0", "read_s1", "side_read_share",
             "gain_s0", "gain_s1", "p_swap", "step_cos_s0", "step_cos_s1",
             "norm_s0", "norm_s1", "side_cos_a0", "side_cos_m0", "cos_s0_s1")) + "</tr>")

    triad_rows = []
    for ratio in (.1, .5, .9):
        key = f"r{round(100 * ratio):02d}"
        for role in ROLES:
            local = causal["ratios"][key][role]
            effects, contrasts = local["effects_vs_intact"], local["contrasts"]
            triad_rows.append(
                f"<tr><td>{ratio:.1f}</td><td>{role}</td>"
                f'<td>{fmt(effects["late_cut_side"])}</td>'
                f'<td>{fmt(effects["mid_shuffle_side"])}</td>'
                f'<td>{fmt(effects["mid_shuffle_side_rescue"])}</td>'
                f'<td>{contrasts["rescue_fraction"]["fraction"]:.1%}</td></tr>')

    control_rows = []
    for role in ROLES:
        effects = causal["ratios"]["r50"][role]["effects_vs_intact"]
        control_rows.append(
            f"<tr><td>{role}</td>" + "".join(
                f"<td>{fmt(effects[name])}</td>" for name in
                ("late_cut_side", "late_cut_active", "mid_shuffle_side",
                 "mid_shuffle_active", "prehead_zero_side")) + "</tr>")

    identity_rows = []
    for role in ROLES:
        local = identity["roles"][role]["effects_vs_intact"]
        shuffle = local["a8_shuffle"]["delta_ce"]
        cells = []
        for name in ("a8_shuffle", "a8_patch_m0_raw", "a8_patch_a0_raw",
                     "a8_patch_m0_normmatched", "a8_patch_a0_normmatched"):
            if name == "a8_shuffle":
                share = ""
            else:
                rescued = 1 - local[name]["delta_ce"] / shuffle
                share = (f' <span class="muted">({rescued:.0%} ущерба снято)</span>' if rescued > 0
                         else ' <span class="muted">(хуже, чем shuffle)</span>')
            cells.append(f"<td>{fmt(local[name])}{share}</td>")
        identity_rows.append(f"<tr><td>{role}</td>" + "".join(cells) + "</tr>")

    stale = causal["ratios"]["r50"]["masked"]["effects_vs_intact"]
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Атлас residual streams</title><style>
body{{font:15px/1.6 system-ui,-apple-system,sans-serif;max-width:1240px;margin:auto;padding:32px 22px;color:#1f2430;background:#fcfcfb}}
h1,h2,h3{{line-height:1.25}}h2{{margin-top:2.2em;border-bottom:2px solid #e6e9ee;padding-bottom:.3em}}
code{{background:#f1f3f5;padding:2px 5px;border-radius:3px;font-size:.92em}}
.lede{{border-left:5px solid #0b86b8;background:#f2f8fb;padding:16px 20px;border-radius:0 6px 6px 0}}
.note{{border-left:4px solid #d28d27;background:#fdf9f0;padding:13px 17px;border-radius:0 6px 6px 0}}
.warn{{border-left:4px solid #b43c5a;background:#fdf3f5;padding:13px 17px;border-radius:0 6px 6px 0}}
img{{width:100%;border:1px solid #e6e9ee;border-radius:10px;background:#fff}}
.tw{{overflow-x:auto;margin:1em 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 9px;border-bottom:1px solid #e6e9ee;text-align:right;white-space:nowrap}}
th{{background:#f7f8fa;position:sticky;top:0;font-weight:600}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.muted{{color:#6b7280;font-size:.88em}}
.scheme{{font:13.5px/1.75 ui-monospace,SFMono-Regular,monospace;background:#f7f8fa;padding:15px 17px;border-radius:8px;overflow-x:auto;border:1px solid #e6e9ee}}
.tall{{max-height:560px;overflow:auto}}
</style><body>
<h1>Что происходит с residual streams по слоям</h1>
<p class="muted">Frozen full-n=2 mHC-lite, step {CHECKPOINT['checkpoint_step']}
({CHECKPOINT['training_input_tokens'] / 1e9:.2f}B training tokens). Routing — 64 калибровочные
последовательности на каждый из пяти уровней шума; геометрия и причинные тесты — 128
discovery-последовательностей при t=.5, одни и те же маски во всех условиях.</p>

<div class="lede"><b>Модель использует второй поток как write-once буфер.</b>
Содержимое компилируется в него в первом блоке, затем проходит семь блоков нетронутым —
из него почти не читают и в него почти не пишут, пошаговый косинус ровно 1.000. Ближе к концу
он перекомпилируется в узком переходе <code>m8–m10</code> и только после этого его читают
последние пять блоков. Причинно необходимы обе части: порча содержимого буфера вредит,
а возврат чистого содержимого перед чтением её спасает.</div>

<h2>Пять фаз</h2>
<div class="scheme">a0 ─ m0 │ a1 ─ m1 │ a2 ─────────────── a8 │ m8 ── m10 │ a11 ─────────── m15 → mean readout
 write-in │ hand-off │      parked (7 блоков)     │  rewrite  │    late read
 контент  │  active   │  side не читают и         │ side      │ router читает
 в side   │  растёт   │  не меняют                │ переписан │ продукт перезаписи</div>
<div class="tw"><table><thead><tr><th>фаза</th><th>интерфейсы</th><th>блоков</th><th>роль</th>
<th>side read share</th><th>gain→side</th><th>gain→active</th><th>P(swap)</th>
<th>step cos side</th><th>step cos active</th><th>‖side‖ на выходе</th><th>‖active‖ на выходе</th>
</tr></thead><tbody>{''.join(phase_rows)}</tbody></table></div>
<p class="muted">Границы фаз взяты из <code>candidate_selection.json</code>, зафиксированного
по routing-профилям до того, как был посчитан любой causal-результат. Ни один CE в выбор границ
не входил.</p>

<h2>Атлас по глубине</h2>
<img src="stream_atlas.svg" alt="Четыре панели: чтение из side, запись в side, нормы потоков, устойчивость к шуму">
<p>Панель D показывает, что фазовая структура не артефакт одного уровня шума: профиль чтения
держит ту же форму при t от .1 до .9, сдвигается только высота позднего плато.</p>

<h2>Пошаговое изменение каждого потока</h2>
<img src="side_drift.svg" alt="Пошаговый косинус каждого потока по интерфейсам">
<p>Потоки работают в противофазе. В parked-фазе active меняется каждый шаг (медиана пошагового
косинуса .937 у masked), а side стоит неподвижно (медиана 1.000, минимум .993 на тринадцати
интерфейсах подряд). После <code>a8</code> они меняются ролями.</p>
<p class="note"><b>Поздние записи в side делает mixer, не attention.</b> В окне
<code>a11…m15</code> у masked пошаговый косинус side на attention-интерфейсах имеет медиану
.983, на mixer-интерфейсах — .945 (минимум .826). У visible контраст ещё резче: .999 против .981.</p>

<h2>Причинные тесты</h2>
<h3>Основная тройка и проверка по шуму</h3>
<p>ΔCE к intact на masked-целях, paired 95% sequence-bootstrap CI, 5000 draws.</p>
<div class="tw"><table><thead><tr><th>t</th><th>роль вмешательства</th><th>запрет позднего чтения side</th>
<th>порча parked state</th><th>+ возврат чистого state</th><th>ущерба снято</th></tr></thead>
<tbody>{''.join(triad_rows)}</tbody></table></div>

<h3>Controls при t=.5</h3>
<div class="tw"><table><thead><tr><th>роль</th><th>side read cut</th><th>active read cut</th>
<th>side shuffle</th><th>active shuffle</th><th>side→0 перед головой</th></tr></thead>
<tbody>{''.join(control_rows)}</tbody></table></div>
<p>У visible обнуление side непосредственно перед readout даёт ровно ноль — после последнего
слоя visible-позиции физически не могут влиять на masked logits, поэтому весь их эффект идёт
через позднюю обработку. У masked прямой вклад в logits существует отдельно
({fmt(stale['prehead_zero_side'])}) и его нельзя смешивать с поздним router-read.</p>

<h3>Идентичность parked state: патч на границе фазы</h3>
<p>Подстановка делается <b>после a8</b>, то есть на границе parked-фазы, дальше сеть сама
прогоняет <code>m8…m15</code>. Эталон неправильного содержимого — shuffle в той же точке.</p>
<div class="tw"><table><thead><tr><th>роль</th><th>shuffle (эталон)</th><th>патч m0</th>
<th>патч a0</th><th>патч m0, norm-matched</th><th>патч a0, norm-matched</th></tr></thead>
<tbody>{''.join(identity_rows)}</tbody></table></div>
<p class="warn"><b>Важно про глубину патча.</b> В первой версии аудита тот же патч ставился
перед <code>a11</code> и почти ничего не спасал. Это был артефакт глубины, а не содержимого:
к <code>a11</code> чистое состояние уже переписано (cos к m0 .672, норма в 2.55 раза больше),
поэтому раннее состояние там вне многообразия. На границе <code>a8</code> тот же патч m0
снимает 95.5% ущерба у masked.</p>

<h2>Что доказано, что нет</h2>
<h3>Доказано</h3>
<ul>
<li>Поздний retrieval side stream причинно необходим: запрет позднего чтения вредит, порча
parked state вредит, возврат чистого состояния перед чтением спасает. Все три условия с
положительными CI, держатся при t=.1 и .9.</li>
<li>Parked state — причинно подтверждённая память состояния после <code>m0</code>: замена восьми
блоков эволюции сырым m0-вектором стоит +0.0037 CE против +0.0820 за неправильное содержимое.
Norm-matching ничего не меняет, значит дело в направлении, а не в норме.</li>
<li>Физическое сохранение подтверждено и геометрически: тринадцать интерфейсов с пошаговым
косинусом ≥.993 и падением к m0 всего с .988 до .949.</li>
<li>Поздний router-read и прямой вклад в logits — разные каналы, они разведены контролем
с обнулением перед головой.</li>
</ul>
<h3>Не доказано</h3>
<ul>
<li>Это <b>не</b> «сохранили и позже прочитали как есть»: поздние блоки читают продукт
перезаписи <code>m8…m10</code>, а не сам припаркованный вектор (cos к m0 там .672 и ниже,
к <code>m15</code> — .109).</li>
<li>Мы не показали, <i>что именно</i> закодировано в буфере: ни probe, ни декодирования не делали.</li>
<li>Late-read cut перенаправляет side-массу в active, поэтому смешивает потерю side с удвоением
active. Чистое доказательство — пара disruption + rescue.</li>
<li>Масштаб: для masked side — второстепенный канал против active
({fmt(stale['mid_shuffle_side'])} против {fmt(stale['mid_shuffle_active'])} на shuffle).</li>
<li>Ничего не сказано про выбор attention keys; это не attention-source-switch анализ.</li>
</ul>

<h2>Полная таблица по 32 интерфейсам</h2>
<p class="muted">Routing при t=.5; геометрия при t=.5. Пустые ячейки — величины, не определённые
на первом интерфейсе. Машиночитаемая версия: <code>stream_atlas.csv</code>.</p>
<div class="tw tall"><table><thead><tr><th>iface</th><th>фаза</th><th>роль</th>
<th>read s0</th><th>read s1</th><th>side share</th><th>gain s0</th><th>gain s1</th><th>P(swap)</th>
<th>step cos s0</th><th>step cos s1</th><th>‖s0‖</th><th>‖s1‖</th>
<th>side cos a0</th><th>side cos m0</th><th>cos(s0,s1)</th></tr></thead>
<tbody>{''.join(detail_rows)}</tbody></table></div>

<h2>Провенанс</h2>
<p class="muted">Checkpoint <code>{escape(CHECKPOINT['checkpoint'])}</code>,
sha256 <code>{CHECKPOINT['checkpoint_sha256'][:16]}…</code>.
Источники: <code>residual_circuits_n2/captures/r*_all_positions.npz</code>,
<code>side_drift.json</code>, <code>final_results.json</code>, <code>side_identity.json</code>,
<code>candidate_selection.json</code>, <code>verification.json</code>.
Скрипты: <code>mechanism_audit_n2/side_retrieval.py</code>,
<code>side_identity.py</code>, <code>side_drift.py</code>, <code>stream_atlas.py</code>.</p>
</body></html>'''
    (OUTPUT / "stream_atlas.html").write_text(html)


def main():
    routing = load_routing()
    drift = json.loads((OUTPUT / "side_drift.json").read_text())
    causal = json.loads((OUTPUT / "final_results.json").read_text())
    identity = json.loads((OUTPUT / "side_identity.json").read_text())
    rows = write_tables(routing, drift)
    summary = phase_summary(routing, drift)
    render_figure(routing, drift)
    render_html(routing, drift, rows, summary, causal, identity)
    print("WROTE", OUTPUT / "stream_atlas.html", flush=True)
    for entry in summary:
        local = entry["roles"]["masked"]
        print(f"  {entry['phase']:<10}{entry['interfaces']:<10}"
              f"side_read={local['side_read_share']:.3f} gain_side={local['gain_side']:.3f} "
              f"step_cos_side={local['step_cos_side']:.3f} step_cos_active={local['step_cos_active']:.3f}")


if __name__ == "__main__":
    main()
