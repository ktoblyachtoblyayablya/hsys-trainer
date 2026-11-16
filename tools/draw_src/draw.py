#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later · Copyright (C) 2026 Пантелеев Г.Д.
"""Рисует схемы и графики img/svg/sv_*.svg для тренажёра.
Запуск из корня репозитория: python3 tools/draw_src/draw.py  (нужны numpy и matplotlib).
Графики работ 1–2 построены по замерам Google Benchmark (results.json из репозитория hsys),
данные Top500 — июньский список 2026 года (top500.org)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'img', 'svg')
os.makedirs(OUT, exist_ok=True)

# палитра: категориальные цвета в фиксированном порядке, текст — чернилами
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a63d2'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"
MONO = "DejaVu Sans Mono, Consolas, monospace"

plt.rcParams.update({
    'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'legend.frameon': False, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'svg.hashsalt': 'hsys'})


def _strip_doctype(path):
    """matplotlib пишет <!DOCTYPE svg …>; убираем — SVG без DTD проще встраивать."""
    import re
    with open(path, encoding='utf-8') as fh: s = fh.read()
    with open(path, 'w', encoding='utf-8') as fh: fh.write(re.sub(r'<!DOCTYPE[^>]*>\s*', '', s))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.svg'), format='svg', metadata={'Date': None})
    _strip_doctype(os.path.join(OUT, name + '.svg'))
    plt.close(fig)


# ---------------------------------------------------------------- графики

def chart_work1():
    n = 2 ** np.arange(10, 21)
    cpu = np.array([74.6, 139.1, 1464.0, 736.6, 1520.0, 2978.1, 6073.5, 24867.9, 60706.6, 118592.7, 814507.5])
    gpu = np.array([14905.4, 16642.2, 19652.3, 25479.4, 38931.2, 61507.1, 112694.4, 214848.4, 420870.2,
                    702217.7, 1267154.6])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.plot(n, cpu / 1e3, 'o-', color=S1, label='CPU, Eigen (VectorXf)', ms=5)
    ax.plot(n, gpu / 1e3, 's-', color=S2, label='GPU, add_gpu + копирования', ms=5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    ax.set_xlabel('n — число элементов вектора')
    ax.set_ylabel('время, мкс')
    ax.set_title('Работа 1: сложение векторов, время')
    ax.annotate('n = 2²⁰: 3 вектора по 4 МиБ = 12 МиБ\n= весь L3 процессора', xy=(2 ** 20, 814.5),
                xytext=(2 ** 12.2, 250), fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    ax.legend(loc='lower right')
    save(fig, 'sv_w1_time')

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.semilogx(n, cpu / gpu, 'o-', color=S2, ms=5, base=2)
    ax.axhline(1, color=INK2, lw=1, ls='--')
    ax.text(2 ** 10.1, 1.03, 'ускорение = 1: GPU и CPU наравне', fontsize=9.5, color=INK2, va='bottom')
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('n — число элементов вектора')
    ax.set_ylabel('T_CPU / T_GPU')
    ax.set_title('Работа 1: ускорение GPU относительно Eigen')
    ax.text(2 ** 10.1, 0.55, 'меньше 1 — GPU медленнее:\nв замер входят копирования\nпо PCIe', fontsize=9.5, color=INK2)
    save(fig, 'sv_w1_speed')


def chart_work2():
    n = np.array([16, 32, 64, 128, 256, 512, 1024])
    eig = np.array([294.8, 1898.5, 14828.5, 115045.0, 960826.7, 7227302.9, 58540458.9])
    gpu = np.array([3108.0, 3750.9, 5155.5, 11802.3, 77920.0, 586323.9, 5043052.2])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.plot(n, eig / 1e3, 'o-', color=S1, label='CPU, Eigen::MatrixXf', ms=5)
    ax.plot(n, gpu / 1e3, 's-', color=S2, label='GPU, kernel_matmul_naive (CUDA Events)', ms=5)
    ref = 2e-5 * n.astype(float) ** 3
    ax.plot(n, ref * gpu[-1] / 1e3 / ref[-1], ':', color=INK2, lw=1.4, label='∝ n³')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    ax.set_xlabel('n (матрицы n × n)')
    ax.set_ylabel('время, мкс')
    ax.set_title('Работа 2: умножение матриц, время')
    ax.legend(loc='upper left')
    save(fig, 'sv_w2_time')

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    sp = eig / gpu
    ax.semilogx(n, sp, 'o-', color=S2, ms=5, base=2)
    ax.axhline(1, color=INK2, lw=1, ls='--')
    for x, y in zip(n, sp):
        ax.annotate(f'{y:.1f}'.replace('.', ','), (x, y), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=9.5, color=INK)
    ax.set_ylim(0, 14.5)
    ax.set_xlabel('n (матрицы n × n)')
    ax.set_ylabel('T_Eigen / T_GPU')
    ax.set_title('Работа 2: ускорение наивного ядра относительно Eigen')
    save(fig, 'sv_w2_speed')


def chart_top10():
    # июнь 2026, Rmax в Пфлопс; цвет — чем считает система
    rows = [('LineShine (Китай)', 2198.4, 'cpu'), ('El Capitan (США)', 1809.0, 'amd'),
            ('Frontier (США)', 1353.0, 'amd'), ('Aurora (США)', 1012.0, 'intel'),
            ('JUPITER Booster (Германия)', 1000.0, 'nv'), ('HPC7, Eni (Италия)', 571.5, 'amd'),
            ('Eagle, Microsoft (США)', 561.2, 'nv'), ('HPC6, Eni (Италия)', 477.9, 'amd'),
            ('Fugaku (Япония)', 442.0, 'cpu'), ('Alps (Швейцария)', 434.9, 'nv')]
    col = {'amd': S2, 'nv': S3, 'intel': S1, 'cpu': '#8a8986'}
    lab = {'amd': 'AMD Instinct', 'nv': 'NVIDIA', 'intel': 'Intel GPU Max', 'cpu': 'без ускорителей'}
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, (name, r, k) in zip(y, rows):
        ax.barh(yi, r, color=col[k], height=0.62, label=lab[k])
        ax.text(r + 25, yi, f'{r:,.0f}'.replace(',', ' '), va='center', fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(1000, color=INK2, lw=1, ls='--')
    ax.text(1015, -0.95, '1 Эфлопс', fontsize=9, color=INK2)
    ax.set_ylim(-1.3, 9.6)
    ax.set_xlabel('Rmax (HPL), Пфлопс')
    ax.set_title('Top500, июнь 2026: первая десятка')
    ax.grid(axis='y', visible=False)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc='lower right', title='ускорители', title_fontsize=9.5)
    ax.set_xlim(0, 2500)
    save(fig, 'sv_top10')


def chart_top1():
    pts = [(1993, 0.0597, 'CM-5'), (1997, 1.068, 'ASCI Red'), (2002, 35.86, 'Earth Simulator'),
           (2004, 70.72, 'Blue Gene/L'), (2008, 1026, 'Roadrunner'), (2010, 2566, 'Тяньхэ-1A'),
           (2011, 10510, 'K computer'), (2012, 17590, 'Titan'), (2013, 33860, 'Тяньхэ-2'),
           (2016, 93010, 'Sunway TaihuLight'), (2018, 143500, 'Summit'), (2020, 442010, 'Fugaku'),
           (2022, 1102000, 'Frontier'), (2024, 1742000, 'El Capitan'), (2026, 2198400, 'LineShine')]
    gpu = {'Тяньхэ-1A', 'Titan', 'Summit', 'Frontier', 'El Capitan', 'Roadrunner'}
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = [p[0] for p in pts]
    yv = [p[1] for p in pts]
    ax.semilogy(x, yv, '-', color=GRID, lw=2, zorder=1)
    lead = {'Earth Simulator': (1996.2, 2500), 'Roadrunner': (2001.5, 3.0e4), 'Тяньхэ-1A': (2011.3, 120),
            'K computer': (2013.6, 600), 'Titan': (2016.0, 3000), 'Тяньхэ-2': (2017.4, 1.2e4)}
    for X, Y, nm in pts:
        c = S2 if nm in gpu else S1
        ax.semilogy(X, Y, 'o', color=c, ms=6, zorder=2)
        if nm in lead:
            ax.annotate(nm, (X, Y), xytext=lead[nm], fontsize=8.5, color=INK, ha='center', va='center',
                        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8, shrinkA=2, shrinkB=4))
        else:
            ax.annotate(nm, (X, Y), textcoords='offset points', xytext=(3, 6), ha='left', va='bottom', fontsize=8.5,
                        color=INK, rotation=32, rotation_mode='anchor')
    ax.semilogy([], [], 'o', color=S2, label='гибридные (Cell или GPU)')
    ax.semilogy([], [], 'o', color=S1, label='без ускорителей')
    for v, t in [(1, 'Тфлопс'), (1e3, 'Пфлопс'), (1e6, 'Эфлопс')]:
        ax.axhline(v, color=INK2, lw=0.8, ls=':')
        ax.text(1992.3, v * 1.25, '1 ' + t, fontsize=8.5, color=INK2)
    ax.set_ylabel('Rmax, Тфлопс (лог. шкала)')
    ax.set_title('№ 1 списка Top500 по годам')
    ax.set_xlim(1991.5, 2029)
    ax.set_ylim(0.02, 1e9)
    ax.legend(loc='upper left')
    save(fig, 'sv_top1')


def chart_amdahl():
    p = np.logspace(0, 12, 200, base=2)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for a, c in [(0.01, S1), (0.05, S3), (0.1, S4), (0.25, S2)]:
        s = 1 / (a + (1 - a) / p)
        ax.semilogx(p, s, color=c, base=2, label=f'α = {a:g}'.replace('.', ',') + f'  (предел {1 / a:g})')
    ax.set_xlabel('p — число процессоров (нитей, узлов)')
    ax.set_ylabel('ускорение S(p)')
    ax.set_title('Закон Амдала: S = 1 / (α + (1 − α) / p)')
    ax.legend(loc='upper left')
    save(fig, 'sv_amdahl')


def chart_roofline():
    # NVIDIA Tesla T4 (Turing, 40 SM): 8,1 Тфлопс FP32, 65 Тфлопс FP16 на тензорных ядрах, 320 ГБ/с
    bw, fp32, tc = 320.0, 8100.0, 65000.0
    i = np.logspace(-2, 3, 400)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.loglog(i, np.minimum(bw * i, fp32), color=S1, label='FP32 (CUDA-ядра)')
    ax.loglog(i, np.minimum(bw * i, tc), color=S3, ls='--', label='FP16 на тензорных ядрах')
    marks = [(1 / 12, 'сложение векторов\n1 флоп / 12 байт', (8, -26), 'left'),
             (0.25, 'наивное умножение\n2 флопа / 8 байт', (8, -26), 'left'),
             (4.0, 'shared memory, плитка 16:\n≈ 4 флопа на байт', (-10, 12), 'right')]
    for x, t, off, ha in marks:
        y = min(bw * x, fp32)
        ax.plot(x, y, 'o', color=S2, ms=6)
        ax.annotate(t, (x, y), textcoords='offset points', xytext=off, ha=ha, fontsize=8.5, color=INK)
    ax.text(0.011, 4e4, 'без учёта кэшей L1/L2', fontsize=8.5, color=INK2)
    ax.axvline(fp32 / bw, color=INK2, lw=0.8, ls=':')
    ax.text(fp32 / bw * 1.1, 60, f'точка перегиба\n{fp32 / bw:.0f} флоп/байт', fontsize=8.5, color=INK2)
    ax.set_xlabel('арифметическая интенсивность, флоп / байт')
    ax.set_ylabel('Гфлопс')
    ax.set_title('Модель roofline для Tesla T4: 320 ГБ/с, 8,1 Тфлопс FP32')
    ax.legend(loc='lower right')
    ax.set_ylim(2, 1.2e5)
    save(fig, 'sv_roofline')


# ---------------------------------------------------------------- схемы (SVG вручную)

class Svg:
    PAL = {
        'blue': ('#e8f0fd', '#1d5bbf'), 'green': ('#e3f4ea', '#1d7a47'), 'orange': ('#fdeee6', '#c4501b'),
        'gray': ('#f2f2f0', '#8a8986'), 'amber': ('#fdf3d8', '#a87400'), 'violet': ('#efe9fb', '#6b46c1'),
        'white': ('#ffffff', '#8a8986'), 'red': ('#fde8e4', '#b42318')}

    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []

    def rect(self, x, y, w, h, c='blue', rx=6, dash=False, sw=1.4, fill=None):
        f, s = self.PAL[c]
        d = ' stroke-dasharray="5 4"' if dash else ''
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or f}" '
                       f'stroke="{s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, t, size=13, anchor='middle', weight=400, fill=INK, mono=False, italic=False, halo=False):
        fam = MONO if mono else FONT
        st = ' font-style="italic"' if italic else ''
        if halo:
            st += ' stroke="#ffffff" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
        t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.el.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" text-anchor="{anchor}" '
                       f'font-weight="{weight}" fill="{fill}"{st}>{t}</text>')

    def box(self, x, y, w, h, lines, c='blue', size=13, rx=6, dash=False, mono=False, first_bold=True):
        self.rect(x, y, w, h, c, rx, dash)
        lines = [lines] if isinstance(lines, str) else lines
        lh = size * 1.3
        y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size * 0.35
        for i, t in enumerate(lines):
            self.text(x + w / 2, y0 + i * lh, t, size=size if i == 0 else size - 1.5,
                      weight=600 if (i == 0 and first_bold) else 400, fill=INK if i == 0 else INK2, mono=mono)

    def line(self, x1, y1, x2, y2, arrow=True, color=INK2, sw=1.5, dash=False, both=False):
        d = ' stroke-dasharray="5 4"' if dash else ''
        m = ' marker-end="url(#ah)"' if arrow else ''
        if both:
            m += ' marker-start="url(#ahs)"'
        self.el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, arrow=True, color=INK2, sw=1.5, dash=False, fill='none'):
        da = ' stroke-dasharray="5 4"' if dash else ''
        m = ' marker-end="url(#ah)"' if arrow else ''
        self.el.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{m}/>')

    def raw(self, s):
        self.el.append(s)

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img"><defs>'
                '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
                '<marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
                f'</defs><rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        with open(os.path.join(OUT, name + '.svg'), 'w', encoding='utf-8') as f:
            f.write(head + '\n'.join(self.el) + '</svg>\n')


def d_cpu_gpu():
    s = Svg(760, 330)
    s.text(190, 24, 'CPU: несколько сложных ядер', 15, weight=700)
    s.text(570, 24, 'GPU: тысячи простых ядер', 15, weight=700)
    s.rect(20, 36, 340, 214, 'gray', rx=10)
    for i, (x, y) in enumerate([(34, 50), (196, 50), (34, 118), (196, 118)]):
        s.rect(x, y, 150, 60, 'white', rx=5)
        s.box(x + 4, y + 4, 86, 52, ['Управление', 'предсказание', 'ветвлений'], 'amber', size=10)
        s.box(x + 94, y + 4, 52, 52, ['ALU', 'SIMD'], 'green', size=11)
    s.box(34, 186, 312, 52, ['Большой кэш L1 / L2 / L3', 'прячет задержку памяти'], 'blue', size=12)
    s.box(20, 262, 340, 36, ['DRAM (DDR): ~0,1 ТБ/с'], 'violet', size=12)
    s.rect(400, 36, 340, 214, 'gray', rx=10)
    for r in range(7):
        y = 46 + r * 25
        s.rect(410, y, 14, 19, 'amber', rx=2, sw=1)
        for c in range(18):
            s.rect(428 + c * 16.5, y, 13, 19, 'green', rx=2, sw=1)
    s.box(410, 224, 320, 20, ['L2 (единицы–десятки МБ)'], 'blue', size=10.5)
    s.box(400, 262, 340, 36, ['GDDR / HBM: 0,3–8 ТБ/с'], 'violet', size=12)
    s.text(380, 320, 'CPU оптимизирован под задержку одной нити, GPU — под пропускную способность тысяч нитей (SIMT)',
           12, fill=INK2)
    s.save('sv_cpu_gpu')


def d_grid():
    s = Svg(760, 350)
    s.text(185, 22, 'Сетка (grid): gridDim = (3, 2)', 14, weight=700)
    s.rect(20, 34, 330, 200, 'gray', rx=10)
    for by in range(2):
        for bx in range(3):
            hl = (bx, by) == (1, 1)
            s.box(32 + bx * 106, 46 + by * 94, 96, 82, [f'Блок ({bx}, {by})'], 'orange' if hl else 'blue', size=12)
    s.text(565, 22, 'Блок (1, 1): blockDim = (4, 3)', 14, weight=700)
    s.rect(400, 34, 340, 200, 'orange', rx=10)
    for ty in range(3):
        for tx in range(4):
            s.box(414 + tx * 81, 48 + ty * 60, 72, 50, ['нить', f'({tx}, {ty})'], 'white', size=11.5)
    s.text(380, 266, 'kernel<<<dim3(3, 2), dim3(4, 3)>>>(...);   // 6 блоков по 12 нитей = 72 нити', 12.5, mono=True)
    s.text(380, 292, 'x = blockIdx.x * blockDim.x + threadIdx.x    // 0 … 11', 12.5, mono=True)
    s.text(380, 314, 'y = blockIdx.y * blockDim.y + threadIdx.y    // 0 … 5', 12.5, mono=True)
    s.text(380, 340, 'Нити одного блока выполняются на одном SM и делят его shared memory; блоки независимы',
           12, fill=INK2)
    s.save('sv_grid')


def d_memhier():
    s = Svg(760, 400)
    s.rect(14, 14, 540, 230, 'gray', rx=10)
    s.text(30, 34, 'Устройство (GPU)', 13, anchor='start', weight=700)
    for b in range(2):
        x0 = 30 + b * 262
        s.rect(x0, 44, 248, 186, 'orange', rx=8)
        s.text(x0 + 124, 62, f'Блок ({b}, 0)', 12.5, weight=600)
        s.box(x0 + 12, 70, 224, 30, ['Разделяемая память (shared)'], 'amber', size=11.5)
        for t in range(2):
            tx = x0 + 12 + t * 116
            s.box(tx, 112, 108, 30, ['Регистры'], 'green', size=11.5)
            s.box(tx, 150, 108, 30, [f'Нить ({t}, 0)'], 'white', size=11.5)
            s.box(tx, 190, 108, 30, ['Локальная'], 'violet', size=11)
            s.line(tx + 54, 150, tx + 54, 142, arrow=True)
            s.line(tx + 54, 180, tx + 54, 190, arrow=True)
            s.line(tx + 54, 112, tx + 54, 100, arrow=True)
    s.box(30, 256, 508, 34, ['Глобальная память (global) — DRAM, видна всем нитям и хосту'], 'blue', size=12)
    s.box(30, 298, 250, 32, ['Константная (64 КБ, кэш)'], 'blue', size=11.5)
    s.box(288, 298, 250, 32, ['Текстурная (кэш, 2D)'], 'blue', size=11.5)
    s.box(590, 256, 156, 74, ['Память хоста', 'cudaMemcpy', 'по PCIe / NVLink'], 'gray', size=12)
    s.line(538, 273, 590, 273, both=True)
    s.text(650, 34, 'задержка доступа', 12, weight=700)
    for i, (t, v) in enumerate([('регистры', '≈ 1 такт'), ('shared, L1', '≈ 20–30 тактов'),
                                ('L2', '≈ 200 тактов'), ('global (DRAM)', '≈ 400–800 тактов'),
                                ('хост (PCIe)', 'мкс')]):
        s.text(586, 60 + i * 34, t, 12, anchor='start', weight=600)
        s.text(586, 76 + i * 34, v, 11.5, anchor='start', fill=INK2)
    s.text(380, 362, 'Регистры и локальная — у нити, shared — у блока, global / constant / texture — у всей сетки',
           12, fill=INK2)
    s.text(380, 384, 'Локальная память физически лежит в DRAM: туда «вытесняются» регистры (register spilling)',
           12, fill=INK2)
    s.save('sv_memhier')


def d_sm():
    s = Svg(760, 440)
    s.rect(10, 10, 740, 420, 'gray', rx=12)
    s.text(380, 32, 'Потоковый мультипроцессор (SM) Turing — как в Tesla T4', 14, weight=700)
    s.box(24, 42, 712, 26, ['L1-кэш инструкций'], 'white', size=11.5)
    for i in range(4):
        x = 24 + (i % 2) * 360
        y = 76 + (i // 2) * 150
        s.rect(x, y, 352, 142, 'white', rx=8)
        s.box(x + 8, y + 6, 336, 22, ['L0 I-кэш · планировщик варпов · диспетчер'], 'amber', size=10.5)
        s.box(x + 8, y + 32, 336, 22, ['Регистровый файл: 16 384 × 32 бит (64 КБ)'], 'violet', size=10.5)
        s.box(x + 8, y + 58, 80, 46, ['INT32', '× 16'], 'green', size=11)
        s.box(x + 92, y + 58, 80, 46, ['FP32', '× 16'], 'green', size=11)
        s.box(x + 176, y + 58, 168, 46, ['Тензорные ядра', '× 2 (FP16, INT8, INT4)'], 'orange', size=11)
        s.box(x + 8, y + 108, 164, 26, ['LD/ST × 4'], 'blue', size=10.5)
        s.box(x + 176, y + 108, 168, 26, ['SFU × 4 (sin, exp, rsqrt)'], 'blue', size=10.5)
    s.box(24, 378, 470, 40, ['L1-кэш данных / разделяемая память: 96 КБ', 'shared — до 64 КБ на SM'], 'amber',
          size=11.5)
    s.box(502, 378, 110, 40, ['TEX × 4'], 'blue', size=11.5)
    s.box(620, 378, 116, 40, ['RT-ядро'], 'blue', size=11.5)
    s.save('sv_sm')


def d_chip():
    s = Svg(760, 440)
    s.rect(10, 10, 740, 396, 'gray', rx=12)
    s.box(130, 20, 500, 26, ['Хост-интерфейс PCI Express 3.0 × 16'], 'violet', size=12)
    s.box(130, 52, 500, 26, ['GigaThread Engine — раздаёт блоки сетки по SM'], 'amber', size=12)
    for r in range(2):
        for g in range(3):
            x = 120 + g * 176
            y = 88 + r * 166
            s.rect(x, y, 168, 136, 'blue', rx=8)
            s.text(x + 84, y + 16, f'GPC {r * 3 + g}', 11.5, weight=700)
            for t in range(4):
                tx = x + 8 + (t % 2) * 78
                ty = y + 24 + (t // 2) * 54
                s.rect(tx, ty, 74, 48, 'white', rx=5)
                s.text(tx + 37, ty + 13, 'TPC', 9.5, weight=600, fill=INK2)
                s.box(tx + 4, ty + 18, 31, 26, ['SM'], 'green', size=10)
                s.box(tx + 39, ty + 18, 31, 26, ['SM'], 'green', size=10)
    s.box(120, 228, 520, 22, ['L2-кэш 4 МБ (общий для всех SM)'], 'orange', size=11.5)
    for side, x in (('л', 20), ('п', 652)):
        for k in range(4):
            s.box(x, 92 + k * 72, 88, 60, ['Контроллер', 'GDDR6', '32 бит'], 'violet', size=10)
    s.text(380, 428, 'TU104: 6 GPC × 4 TPC × 2 SM = 48 SM; в Tesla T4 включено 40 SM → 2560 CUDA-ядер, 320 тензорных',
           11.5, fill=INK2)
    s.save('sv_chip')


def d_toolchain():
    s = Svg(760, 340)
    s.box(14, 132, 92, 50, ['main.cu', 'kernel.cu'], 'gray', size=12, mono=True)
    s.box(130, 132, 92, 50, ['nvcc', 'разделяет'], 'amber', size=12)
    s.line(106, 157, 130, 157)
    s.box(250, 30, 140, 46, ['Код хоста', 'C++'], 'blue', size=12)
    s.box(414, 30, 140, 46, ['g++ / clang++', 'компилятор хоста'], 'blue', size=11.5)
    s.box(250, 236, 140, 46, ['Код устройства', '__global__, __device__'], 'green', size=11.5)
    s.box(414, 236, 108, 46, ['PTX', 'compute_75'], 'green', size=12)
    s.box(546, 236, 92, 46, ['ptxas'], 'green', size=12)
    s.box(660, 236, 88, 46, ['SASS', 'cubin sm_75'], 'green', size=11.5)
    s.path('M 222 150 C 236 150 236 53 250 53')
    s.path('M 222 164 C 236 164 236 259 250 259')
    s.line(390, 53, 414, 53)
    s.line(390, 259, 414, 259)
    s.text(402, 230, 'cicc (NVVM)', 10, fill=INK2)
    s.line(522, 259, 546, 259)
    s.line(638, 259, 660, 259)
    s.box(560, 128, 188, 58, ['fatbinary', 'SASS для sm_75 + PTX', 'встраивается в объектник'], 'orange', size=11.5)
    s.path('M 704 236 L 704 186')
    s.path('M 468 236 L 468 210 L 600 210 L 600 186', dash=True)
    s.box(400, 128, 136, 58, ['Компоновка', 'nvlink + ld'], 'gray', size=12)
    s.line(484, 76, 484, 128)
    s.line(560, 157, 536, 157)
    s.box(250, 128, 126, 58, ['Исполняемый', 'файл / .so'], 'gray', size=12)
    s.line(400, 157, 376, 157)
    s.text(380, 330, 'При запуске драйвер берёт готовый SASS для своей архитектуры, а если его нет — '
                     'JIT-компилирует PTX', 12, fill=INK2)
    s.text(380, 310, 'cuobjdump -sass / -ptx показывает, что лежит в fatbinary', 12, fill=INK2)
    s.save('sv_toolchain')


def d_cluster():
    s = Svg(760, 420)
    s.box(20, 20, 150, 46, ['Пользователи', 'ssh, scp'], 'gray', size=12)
    s.box(220, 20, 170, 56, ['Узел доступа (login)', 'сборка, qsub / sbatch'], 'blue', size=12)
    s.box(430, 20, 310, 56, ['Управляющий узел (head)', 'сервер очередей, планировщик, мониторинг'], 'amber',
          size=12)
    s.line(170, 43, 220, 43)
    for i in range(4):
        x = 20 + i * 150
        s.rect(x, 140, 138, 120, 'white', rx=8)
        s.text(x + 69, 158, f'Узел {i + 1}' if i < 3 else 'Узел N', 11.5, weight=700)
        s.box(x + 8, 166, 58, 34, ['CPU'], 'blue', size=11.5)
        s.box(x + 72, 166, 58, 34, ['CPU'], 'blue', size=11.5)
        for g in range(4):
            s.box(x + 8 + g * 31, 208, 28, 44, ['GPU'], 'green', size=8.5)
    s.text(625, 205, '…', 22, weight=700)
    s.box(650, 140, 96, 120, ['СХД', 'Lustre,', 'GPFS'], 'violet', size=12)
    ys = {'ib': 292, 'eth': 330, 'ipmi': 366}
    s.line(20, ys['ib'], 746, ys['ib'], arrow=False, color='#c4501b', sw=4)
    s.text(383, ys['ib'] + 20, 'Высокоскоростная сеть (интерконнект): InfiniBand, Slingshot — MPI и доступ к СХД',
           11.5, weight=600, halo=True)
    s.line(20, ys['eth'], 746, ys['eth'], arrow=False, color='#1d5bbf', sw=3)
    s.text(383, ys['eth'] + 18, 'Служебная сеть Ethernet: загрузка узлов, NFS, мониторинг, управление заданиями',
           11.5, weight=600, halo=True)
    s.line(20, ys['ipmi'], 746, ys['ipmi'], arrow=False, color='#8a8986', sw=3)
    s.text(383, ys['ipmi'] + 18, 'Сеть управления IPMI / BMC: питание, консоль, датчики', 11.5, weight=600, halo=True)
    for i in range(4):
        x = 20 + i * 150
        s.line(x + 40, 260, x + 40, ys['ib'] - 3, arrow=False, color='#c4501b', sw=2)
        s.line(x + 69, 260, x + 69, ys['eth'] - 2, arrow=False, color='#1d5bbf', sw=1.5)
        s.line(x + 100, 260, x + 100, ys['ipmi'] - 2, arrow=False, color='#8a8986', sw=1.5)
    s.line(698, 260, 698, ys['ib'] - 3, arrow=False, color='#c4501b', sw=2)
    s.text(480, 104, 'login и head тоже подключены ко всем сетям', 11, fill=INK2)
    s.text(383, 412, 'Гибридный кластер: в узле CPU и ускорители; задания попадают на узлы только через планировщик',
           11.5, fill=INK2)
    s.save('sv_cluster')


def d_node():
    s = Svg(760, 330)
    s.rect(10, 10, 740, 280, 'gray', rx=12)
    s.text(380, 30, 'Гибридный вычислительный узел', 14, weight=700)
    s.box(30, 48, 110, 50, ['DDR', 'ОЗУ узла'], 'violet', size=12)
    s.box(160, 48, 120, 50, ['CPU 0'], 'blue', size=13)
    s.box(480, 48, 120, 50, ['CPU 1'], 'blue', size=13)
    s.box(620, 48, 110, 50, ['DDR', 'ОЗУ узла'], 'violet', size=12)
    s.line(140, 73, 160, 73, both=True)
    s.line(600, 73, 620, 73, both=True)
    s.line(280, 73, 480, 73, both=True)
    s.text(380, 66, 'UPI / Infinity Fabric', 11, fill=INK2)
    s.box(160, 128, 120, 36, ['PCIe-коммутатор'], 'amber', size=11.5)
    s.box(480, 128, 120, 36, ['PCIe-коммутатор'], 'amber', size=11.5)
    s.line(220, 98, 220, 128, both=True)
    s.line(540, 98, 540, 128, both=True)
    xs = [40, 220, 400, 580]
    for i, x in enumerate(xs):
        s.box(x, 200, 140, 62, [f'GPU {i}', 'HBM: 2–8 ТБ/с'], 'green', size=12)
    for i, x in enumerate(xs):
        sw = 220 if i < 2 else 540
        s.line(sw, 164, x + 70, 200, both=True)
    s.path('M 180 231 L 220 231', arrow=False, color='#1d7a47', sw=3)
    s.path('M 360 231 L 400 231', arrow=False, color='#1d7a47', sw=3)
    s.path('M 540 231 L 580 231', arrow=False, color='#1d7a47', sw=3)
    s.text(380, 282, 'NVLink между GPU: сотни ГБ/с', 11.5, fill='#1d7a47', weight=600)
    s.box(300, 128, 160, 36, ['HCA InfiniBand'], 'orange', size=11.5)
    s.line(280, 146, 300, 146, both=True)
    s.text(380, 184, 'в интерконнект (200–400 Гбит/с)', 10.5, fill=INK2)
    s.text(380, 316, 'Узкое место — PCIe: 3.0 × 16 ≈ 16 ГБ/с, 4.0 × 16 ≈ 32 ГБ/с в одну сторону; память GPU — ТБ/с',
           11.5, fill=INK2)
    s.save('sv_node')


def diamond(s, x, y, filled, ang=90):
    # ромб на конце связи: (x, y) — острый угол у класса-владельца
    import math
    a = math.radians(ang)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -dy, dx
    p = [(x, y), (x + dx * 9 + px * 6, y + dy * 9 + py * 6), (x + dx * 18, y + dy * 18),
         (x + dx * 9 - px * 6, y + dy * 9 - py * 6)]
    pts = ' '.join(f'{u:.1f},{v:.1f}' for u, v in p)
    s.raw(f'<polygon points="{pts}" fill="{INK2 if filled else "#ffffff"}" stroke="{INK2}" stroke-width="1.4"/>')


def uml(s, x, y, w, name, fields, ops, c='blue', size=11):
    lh = size + 5
    h = 26 + len(fields) * lh + 8 + len(ops) * lh + 8
    s.rect(x, y, w, h, c, rx=4)
    s.text(x + w / 2, y + 18, name, 12.5, weight=700, mono=True)
    yy = y + 26
    s.line(x, yy, x + w, yy, arrow=False, color=Svg.PAL[c][1], sw=1)
    for i, f in enumerate(fields):
        s.text(x + 8, yy + 15 + i * lh, f, size, anchor='start', mono=True)
    yy += len(fields) * lh + 8
    s.line(x, yy, x + w, yy, arrow=False, color=Svg.PAL[c][1], sw=1)
    for i, o in enumerate(ops):
        s.text(x + 8, yy + 15 + i * lh, o, size, anchor='start', mono=True)
    return h


def d_dataview():
    s = Svg(760, 400)
    h = uml(s, 230, 14, 300, 'Vector<AtomT>  «фасад»',
            ['- data_ : shared_ptr<Data<AtomT>>', '- view_ : VectorView<AtomT>'],
            ['+ Vector(size)', '+ size(), data(), view()', '+ operator+(a, b) : Vector'], 'amber')
    hd = uml(s, 20, 210, 330, 'Data<AtomT>  «владелец, RAII»',
             ['- size_ : size_t', '- data_ : AtomT*'],
             ['+ Data(size)       // cudaMalloc', '+ копирование и перемещение',
              '+ copy_to_host / copy_from_host', '+ ~Data()          // cudaFree'], 'blue')
    hv = uml(s, 410, 210, 330, 'VectorView<AtomT>  «представление»',
             ['- data_ : AtomT*', '- size_ : size_t'],
             ['__host__ __device__ size()', '__host__ __device__ operator[](i)',
              'тривиально копируемый'], 'green')
    import math
    yb = 14 + h
    s.line(320, yb, 185, 210, arrow=False)
    diamond(s, 320, yb, False, ang=math.degrees(math.atan2(210 - yb, 185 - 320)))
    s.line(440, yb, 575, 210, arrow=False)
    diamond(s, 440, yb, True, ang=math.degrees(math.atan2(210 - yb, 575 - 440)))
    for i, t in enumerate(['агрегация (shared_ptr):', 'данные могут делить', 'несколько векторов']):
        s.text(20, 160 + i * 15, t, 11, anchor='start', fill=INK2)
    for i, t in enumerate(['композиция:', 'у каждого Vector', 'своё представление']):
        s.text(740, 160 + i * 15, t, 11, anchor='end', fill=INK2)
    s.box(410, 352, 330, 40, ['kernel_vecadd(VectorView a, b, c)', 'представления передаются по значению'],
          'gray', size=11)
    s.line(575, 210 + hv, 575, 352)
    s.text(185, 380, 'cudaMalloc / cudaFree — только в Data', 11.5, fill=INK2)
    s.save('sv_dataview')


def d_tiling():
    s = Svg(760, 412)
    T = 40
    ax, ay, aw, ah = 30, 200, 160, 160
    bx, by, bw, bh = 220, 20, 200, 160
    cx, cy = 220, 200
    s.rect(ax, ay, aw, ah, 'gray', rx=2)
    s.rect(bx, by, bw, bh, 'gray', rx=2)
    s.rect(cx, cy, bw, ah, 'gray', rx=2)
    for k in range(4):
        s.rect(ax + k * T, ay + T, T, T, 'blue' if k != 1 else 'orange', rx=1, sw=1)
        s.rect(bx + 2 * T, by + k * T, T, T, 'green' if k != 1 else 'orange', rx=1, sw=1)
    s.rect(cx + 2 * T, cy + T, T, T, 'amber', rx=1, sw=2)
    s.text(ax + aw / 2, ay + ah + 18, 'A (m × k)', 13, weight=700)
    s.text(bx - 8, by + 14, 'B (k × n)', 13, anchor='end', weight=700)
    s.text(cx + bw / 2, cy + ah + 18, 'C = A · B (m × n)', 13, weight=700)
    s.text(cx + 2 * T + 20, cy + T + 25, 'C', 12, weight=700)
    s.text(ax + 1.5 * T, ay + T - 6, 'плитка t', 10.5, fill='#c4501b', weight=600)
    s.text(bx + 3 * T + 6, by + 1.5 * T + 4, 'плитка t', 10.5, anchor='start', fill='#c4501b', weight=600)
    s.line(ax + 20, ay + T + 20, ax + aw - 10, ay + T + 20, color='#1d5bbf')
    s.line(bx + 2 * T + 20, by + 20, bx + 2 * T + 20, by + bh - 10, color='#1d7a47')
    s.rect(450, 20, 296, 124, 'amber', rx=8)
    s.text(598, 40, 'Разделяемая память блока 16 × 16', 12.5, weight=700)
    s.box(462, 50, 132, 44, ['a_tile[16][16]', '1 КиБ (float)'], 'blue', size=11, mono=True)
    s.box(602, 50, 132, 44, ['b_tile[16][16]', '1 КиБ (float)'], 'green', size=11, mono=True)
    s.text(598, 116, 'итого 2 КиБ на блок; каждый элемент', 11, fill=INK2)
    s.text(598, 132, 'из global читается в 16 раз реже', 11, fill=INK2)
    steps = ['1. нить грузит по элементу A и B', '2. __syncthreads() — плитки готовы',
             '3. 16 умножений-сложений из shared', '4. __syncthreads() — можно перезаписывать',
             '5. t += 16, повторить до конца k']
    s.rect(450, 158, 296, 150, 'white', rx=8)
    s.text(598, 178, 'Цикл по плиткам t = 0, 16, 32, …', 12.5, weight=700)
    for i, t in enumerate(steps):
        s.text(462, 202 + i * 22, t, 11, anchor='start')
    s.text(598, 334, 'Блок нитей 16 × 16 = 256 нитей', 12, weight=600)
    s.text(598, 352, 'считает одну плитку C 16 × 16', 12, fill=INK2)
    s.text(380, 402, 'Элементы за границей матриц заменяются нулями, поэтому размеры могут быть любыми (1, 127, 129 …)',
           11.5, fill=INK2)
    s.save('sv_tiling')


def d_wmma():
    s = Svg(760, 380)
    x0, y0, F = 200, 120, 40
    s.rect(30, y0, 150, 2 * F, 'blue', rx=2)
    s.text(105, y0 - 8, 'A: 32 строки', 11.5, weight=600)
    s.rect(x0, 20, 4 * F, 80, 'green', rx=2)
    s.text(x0 + 2 * F, 14, 'B: 64 столбца', 11.5, weight=600)
    for wy in range(2):
        for wx in range(4):
            s.box(x0 + wx * F, y0 + wy * F, F, F, [f'{wx},{wy}'], 'amber', size=10.5, rx=1)
    s.text(x0 + 2 * F, y0 + 2 * F + 18, 'Блок C 32 × 64: 8 варпов', 11.5, weight=600)
    s.text(x0 + 2 * F, y0 + 2 * F + 34, 'каждый — фрагмент 16 × 16', 11, fill=INK2)
    s.text(x0 + 2 * F, y0 + 2 * F + 50, 'blockDim = (4 · 32, 2) = 256 нитей', 11, fill=INK2, mono=True)
    s.rect(400, 14, 346, 268, 'white', rx=8)
    s.text(573, 34, 'Что делает один варп', 12.5, weight=700)
    lines = [('fill_fragment(acc, 0.0f)', 'аккумулятор float 16 × 16'),
             ('for p = 0; p < k; p += 16:', ''),
             ('  load_matrix_sync(a_frag, &a(row, p), lda)', 'half'),
             ('  load_matrix_sync(b_frag, &b(p, col), ldb)', 'half'),
             ('  mma_sync(acc, a_frag, b_frag, acc)', 'acc += A·B на тензорных ядрах'),
             ('store_matrix_sync(staging, acc, 16)', 'float → shared'),
             ('__syncwarp(); c = __float2half(staging)', 'запись в C (half)')]
    for i, (c, cm) in enumerate(lines):
        s.text(412, 58 + i * 32, c, 10.5, anchor='start', mono=True)
        if cm:
            s.text(412, 72 + i * 32, cm, 10, anchor='start', fill=INK2)
    s.box(30, 300, 716, 58, ['Ограничения WMMA', 'размеры кратны 16 · указатель выровнен на 32 байта · ldm кратен 8 '
                             'для half · все 32 нити варпа вызывают *_sync вместе'], 'gray', size=11.5)
    s.save('sv_wmma')


def d_reduce():
    s = Svg(760, 430)
    s.text(190, 22, '(1) kernel_vecred_nobr: дерево в shared', 12.5, weight=700)
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    W = 36
    rows = [vals, [8, 10, 6, 7], [14, 17], [31]]
    for r, row in enumerate(rows):
        y = 40 + r * 62
        for i, v in enumerate(row):
            x = 30 + i * W
            s.box(x, y, W - 6, 28, [str(v)], 'amber' if r else 'white', size=12)
        if r:
            st = len(row)
            s.text(30 + len(row) * W + 6, y + 12, f'stride {st}', 11, anchor='start', fill=INK2, mono=True)
            s.text(30 + len(row) * W + 6, y + 26, '+ sync', 10, anchor='start', fill='#b42318', mono=True)
            prev = rows[r - 1]
            for i in range(len(row)):
                s.line(30 + i * W + 17, y - 34, 30 + i * W + 17, y, color=INK2, sw=1.1)
                s.line(30 + (i + st) * W + 17, y - 34, 30 + i * W + 20, y, color='#c4501b', sw=1.1)
    s.text(190, 300, 'log₂256 = 8 шагов, после каждого __syncthreads:', 11, fill=INK2)
    s.text(190, 316, '1 + 8 = 9 барьеров на блок', 11, fill=INK2)
    s.text(570, 22, '(2) kernel_vecred_br: обмен регистрами', 12.5, weight=700)
    rows2 = [vals, [8, 10, 6, 7, '·', '·', '·', '·'], [14, 17, '·', '·'], [31, '·']]
    for r, row in enumerate(rows2):
        y = 40 + r * 62
        for i, v in enumerate(row):
            x = 400 + i * 38
            s.box(x, y, 32, 28, [str(v)], 'green' if r and v != '·' else 'white', size=12)
        if r:
            off = 4 // (2 ** (r - 1))
            s.text(752, y + 18, f'off {off}', 11, anchor='end', fill=INK2, mono=True)
            for i in range(off):
                s.line(400 + (i + off) * 38 + 16, y - 34, 400 + i * 38 + 19, y, color='#1d7a47', sw=1.1)
    s.text(570, 300, 'в варпе 32 нити: смещения 16, 8, 4, 2, 1 —', 11, fill=INK2)
    s.text(570, 316, 'без shared и без барьеров внутри варпа', 11, fill=INK2)
    s.text(570, 334, 'v += __shfl_down_sync(0xffffffff, v, off)', 10.5, mono=True)
    s.rect(20, 350, 720, 70, 'gray', rx=8)
    s.box(32, 362, 150, 46, ['вектор n', 'элементов'], 'white', size=11.5)
    s.box(240, 362, 190, 46, ['partial[blocks]', 'blocks ≤ 1024'], 'amber', size=11.5, mono=True)
    s.box(500, 362, 120, 46, ['result[0]'], 'green', size=11.5, mono=True)
    s.line(182, 385, 240, 385)
    s.line(430, 385, 500, 385)
    s.text(211, 378, 'проход 1', 10, fill=INK2)
    s.text(465, 378, 'проход 2', 10, fill=INK2)
    s.text(680, 380, 'блоки по', 10.5, fill=INK2)
    s.text(680, 396, '256 нитей', 10.5, fill=INK2)
    s.save('sv_reduce')


def d_conv():
    s = Svg(760, 400)
    P, cell = 2, 14
    n = 16
    x0, y0 = 30, 40
    s.text(x0 + (n + 2 * P) * cell / 2, 26, 'вход X[n][c_in] с отступом p (нули)', 12, weight=700)
    for i in range(n + 2 * P):
        for j in range(n + 2 * P):
            pad = i < P or j < P or i >= n + P or j >= n + P
            s.raw(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell}" height="{cell}" '
                  f'fill="{"#f2f2f0" if pad else "#ffffff"}" stroke="#d0cfcb" stroke-width="0.6"/>')
    ts = 9
    s.rect(x0, y0, ts * cell, ts * cell, 'orange', rx=0, sw=2.2, fill='none')
    s.rect(x0, y0, 3 * cell, 3 * cell, 'green', rx=0, sw=2, fill='rgba(29,122,71,0.15)')
    s.text(x0 + ts * cell + 6, y0 + 12, 'плитка входа', 11, anchor='start', fill='#c4501b', weight=600, halo=True)
    s.text(x0 + ts * cell + 6, y0 + 27, 'в shared', 11, anchor='start', fill='#c4501b', halo=True)
    s.text(x0 + 3 * cell / 2, y0 + 3 * cell + 14, 'k × k', 10.5, fill='#1d7a47', weight=600)
    s.rect(400, 20, 346, 200, 'white', rx=8)
    s.text(573, 42, 'Блок 16 × 16 нитей → 16 × 16 выходов', 12.5, weight=700)
    s.text(412, 70, 'tile = (bs − 1)·s + (k − 1)·d + 1', 12, anchor='start', mono=True)
    s.text(412, 94, 'k=3, s=1, d=1:  15 + 2 + 1 = 18', 11.5, anchor='start', mono=True)
    s.text(412, 114, 'k=7, s=2, d=1:  30 + 6 + 1 = 37', 11.5, anchor='start', mono=True)
    s.text(412, 134, 'k=3, s=2, d=2:  30 + 4 + 1 = 35', 11.5, anchor='start', mono=True)
    s.text(412, 162, 'H_out = (H + 2p − d(k − 1) − 1) / s + 1', 12, anchor='start', mono=True)
    s.text(412, 186, 'нить (ty, tx) читает', 11.5, anchor='start', fill=INK2)
    s.text(412, 204, 'tile[ty·s + r·d][tx·s + q·d]', 11.5, anchor='start', mono=True)
    s.rect(400, 232, 346, 112, 'gray', rx=8)
    s.text(573, 252, 'Сетка (grid)', 12.5, weight=700)
    s.text(412, 276, 'blockIdx.x, blockIdx.y — плитка выхода', 11.5, anchor='start', mono=True)
    s.text(412, 298, 'blockIdx.z = n · C_out + c_out', 11.5, anchor='start', mono=True)
    s.text(412, 322, 'цикл по c_in: загрузить → sync → Σ → sync', 11.5, anchor='start', mono=True)
    s.text(380, 376, 'Пример на рисунке: s = 1, p = 2, k = 3 — для блока 7 × 7 выходов нужна плитка 9 × 9 входа',
           11.5, fill=INK2)
    s.save('sv_conv')


def d_torque():
    s = Svg(760, 412)
    s.rect(14, 14, 200, 190, 'blue', rx=10)
    s.text(114, 34, 'Узел доступа', 12.5, weight=700)
    s.box(26, 46, 176, 70, ['job.pbs', '#PBS -l nodes=2:ppn=8:gpus=2', '#PBS -l walltime=01:00:00'], 'white',
          size=10.5, mono=True)
    s.text(114, 140, '$ qsub job.pbs', 11.5, mono=True)
    s.text(114, 160, '$ qstat -u $USER', 11.5, mono=True)
    s.text(114, 180, '$ qdel 1234', 11.5, mono=True)
    s.rect(270, 14, 300, 190, 'amber', rx=10)
    s.text(420, 34, 'Управляющий узел', 12.5, weight=700)
    s.box(286, 46, 268, 58, ['pbs_server', 'очереди, задания, состояние узлов'], 'white', size=11.5)
    s.box(286, 128, 268, 62, ['Планировщик', 'pbs_sched (FIFO) · Maui · Moab'], 'white', size=11.5)
    s.line(390, 104, 390, 128, both=True)
    s.text(398, 120, 'кого и куда запускать', 10, anchor='start', fill=INK2)
    s.line(202, 81, 286, 75)
    s.text(244, 70, 'qsub', 10.5, fill=INK2, mono=True)
    for i in range(3):
        x = 40 + i * 240
        s.rect(x, 250, 200, 92, 'green', rx=8)
        s.text(x + 100, 270, f'Вычислительный узел {i + 1}' if i < 2 else 'узел N', 11.5, weight=700)
        s.box(x + 12, 280, 176, 50, ['pbs_mom', 'запуск, контроль, ресурсы'], 'white', size=11)
        s.line(420, 204, x + 100, 250)
    s.text(210, 362, 'первый узел (mother superior) выполняет скрипт,', 10.5, fill=INK2)
    s.text(210, 376, 'список выделенных узлов — в $PBS_NODEFILE', 10.5, fill=INK2)
    s.box(600, 40, 146, 120, ['Состояния', 'Q — в очереди', 'R — выполняется', 'H — задержано',
                              'E — завершается', 'C — завершено'], 'gray', size=11)
    s.text(380, 402, 'Torque — ответвление OpenPBS: сервер, планировщик и MOM-демоны на узлах; Slurm — slurmctld и slurmd',
           11.5, fill=INK2)
    s.save('sv_torque')


def d_ganglia():
    s = Svg(760, 400)
    for c in range(2):
        x = 14 + c * 250
        s.rect(x, 14, 236, 196, 'green', rx=10)
        s.text(x + 118, 34, f'Кластер {"A" if c == 0 else "B"}', 12.5, weight=700)
        for i in range(3):
            s.box(x + 12 + i * 74, 46, 66, 52, ['узел', 'gmond'], 'white', size=11)
        s.raw(f'<ellipse cx="{x + 118}" cy="140" rx="98" ry="20" fill="#ffffff" stroke="#1d7a47" '
              f'stroke-dasharray="5 4" stroke-width="1.4"/>')
        s.text(x + 118, 136, 'мультикаст UDP 239.2.11.71:8649', 9.5, fill=INK2, mono=True)
        s.text(x + 118, 150, 'каждый gmond знает весь кластер', 9.5, fill=INK2)
        for i in range(3):
            s.line(x + 45 + i * 74, 98, x + 70 + i * 24, 122, both=True, sw=1.1)
        s.text(x + 118, 192, 'XML по TCP 8649', 10.5, fill=INK2, mono=True)
    s.box(530, 14, 216, 86, ['gmetad', 'опрашивает gmond по TCP,', 'пишет в RRD (rrdtool)'], 'amber', size=11.5)
    s.box(530, 120, 216, 62, ['Веб-интерфейс', 'gweb (PHP): графики из RRD'], 'blue', size=11.5)
    s.path('M 132 210 L 132 224 L 515 224 L 515 60 L 530 60')
    s.path('M 382 210 L 382 224', arrow=False)
    s.text(320, 220, 'gmetad опрашивает gmond', 10, fill=INK2, halo=True)
    s.line(638, 100, 638, 120)
    s.box(530, 196, 216, 40, ['Браузер администратора'], 'gray', size=11.5)
    s.line(638, 182, 638, 196)
    s.rect(14, 236, 500, 150, 'orange', rx=10)
    s.text(264, 256, 'Nagios: проверки и оповещения', 12.5, weight=700)
    s.box(26, 266, 150, 58, ['nagios', 'ядро, расписание'], 'white', size=11.5)
    s.box(190, 266, 150, 58, ['плагины', 'check_ping, check_load'], 'white', size=11)
    s.box(354, 266, 148, 58, ['NRPE на узлах', 'удалённые проверки'], 'white', size=11)
    s.line(176, 295, 190, 295)
    s.line(340, 295, 354, 295)
    s.text(264, 346, 'код возврата плагина: 0 OK · 1 WARNING · 2 CRITICAL · 3 UNKNOWN', 11, mono=True)
    s.text(264, 368, '→ оповещение по e-mail / SMS, обработчики событий', 11, fill=INK2)
    s.box(530, 256, 216, 110, ['Ganglia — «как идут дела»', 'метрики и тренды', 'Nagios — «что сломалось»',
                               'пороги и тревоги'], 'gray', size=11)
    s.save('sv_ganglia')


def d_backfill():
    s = Svg(760, 330)
    x0, y0, W, H = 70, 30, 620, 240
    nodes, hours = 8, 10
    cw, rh = W / hours, H / nodes
    s.rect(x0, y0, W, H, 'white', rx=2)
    for i in range(nodes):
        s.text(x0 - 8, y0 + (i + 0.5) * rh + 4, f'узел {i + 1}', 10.5, anchor='end', fill=INK2)
    for t in range(hours + 1):
        s.line(x0 + t * cw, y0, x0 + t * cw, y0 + H, arrow=False, color=GRID, sw=1)
        s.text(x0 + t * cw, y0 + H + 16, f'{t} ч', 10, fill=INK2)

    def job(t0, t1, n0, n1, c, lines, dash=False):
        s.box(x0 + t0 * cw + 1, y0 + n0 * rh + 1, (t1 - t0) * cw - 2, (n1 - n0) * rh - 2, lines, c, size=11,
              rx=4, dash=dash)

    job(0, 4, 0, 6, 'blue', ['A: выполняется', '6 узлов до 4 ч'])
    job(4, 9, 0, 8, 'amber', ['B: ждёт 8 узлов — резерв с 4 ч'], dash=True)
    job(0, 3, 6, 8, 'green', ['C: 2 узла, 3 ч', 'backfill'])
    s.box(x0 + 2, y0 + H + 26, 300, 26, ['D: 2 узла, 6 ч — ждёт: сдвинул бы B'], 'red', size=10.5, rx=4)
    s.text(x0 + W, y0 + H + 44, 'время →', 10.5, anchor='end', fill=INK2)
    s.text(380, 18, 'Обратное заполнение (backfilling): короткое задание C идёт вперёд, не задерживая B', 12,
           weight=700)
    s.save('sv_backfill')


def d_cloud():
    s = Svg(760, 360)
    s.box(20, 300, 720, 44, ['Оборудование: серверы с GPU (A100, H100, T4 …), сеть, СХД'], 'gray', size=12)
    s.box(20, 236, 720, 50, ['Гипервизор (KVM, VMware ESXi) или ОС узла для контейнеров',
                             'OpenStack / VMware vSphere / Kubernetes управляют пулом'], 'violet', size=11.5)
    items = [('PCIe passthrough', 'вся GPU — одной ВМ', 'blue'), ('vGPU (NVIDIA)', 'GPU делится между ВМ', 'blue'),
             ('MIG', 'до 7 изолированных', 'blue'), ('Контейнеры', 'NVIDIA Container Toolkit', 'green')]
    for i, (a, b, c) in enumerate(items):
        s.box(20 + i * 182, 160, 172, 62, [a, b], c, size=11.5)
        s.line(106 + i * 182, 236, 106 + i * 182, 222)
    s.box(20, 84, 352, 60, ['IaaS: ВМ и узлы с GPU по запросу', 'AWS P/G, Azure N, Yandex Cloud, Cloud.ru'],
          'amber', size=11.5)
    s.box(388, 84, 352, 60, ['PaaS / SaaS: ноутбуки, ML-платформы', 'Colab, Kaggle, SageMaker, DataSphere'],
          'amber', size=11.5)
    s.line(196, 160, 196, 144)
    s.line(564, 160, 564, 144)
    s.box(20, 14, 720, 52, ['Пользователь: портал, API, kubectl, Jupyter',
                            'платит за GPU-часы, получает ресурс за минуты'], 'white', size=11.5)
    s.line(380, 66, 380, 84)
    s.save('sv_cloud')


def d_timeline():
    s = Svg(760, 300)
    ev = [(1999, 'GeForce 256 —', '«первый GPU»', 1), (2001, 'GeForce 3:', 'шейдеры', -1),
          (2003, 'BrookGPU,', 'gpgpu.org (2002)', 1), (2006, 'G80, CTM;', 'AMD + ATI', -1),
          (2007, 'CUDA 1.0,', 'Tesla C870', 1), (2008, 'OpenCL 1.0', '', -1),
          (2010, 'Fermi;', 'Тяньхэ-1A № 1', 1), (2012, 'Kepler; Titan', '№ 1, GCN у AMD', -1),
          (2016, 'Pascal P100;', 'ROCm', 1), (2017, 'Volta V100:', 'тензорные ядра', -1),
          (2020, 'Ampere A100;', 'CDNA MI100', 1), (2022, 'Hopper H100;', 'Frontier — 1 Эфлопс', -1),
          (2024, 'Blackwell;', 'El Capitan MI300A', 1), (2026, 'Rubin, MI400', '', -1)]
    x0, x1, y = 40, 720, 150
    s.line(x0, y, x1 + 20, y, color=INK, sw=2)
    for k, (yr, a, b, side) in enumerate(ev):
        x = x0 + 12 + k * (x1 - x0 - 12) / (len(ev) - 1)
        s.raw(f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="#1d5bbf"/>')
        if side > 0:
            s.line(x, y - 6, x, y - 34, arrow=False, color=GRID, sw=1)
            s.text(x, y - 70, str(yr), 11, weight=700)
            s.text(x, y - 55, a, 10)
            s.text(x, y - 42, b, 10, fill=INK2)
        else:
            s.line(x, y + 6, x, y + 34, arrow=False, color=GRID, sw=1)
            s.text(x, y + 50, str(yr), 11, weight=700)
            s.text(x, y + 64, a, 10)
            s.text(x, y + 77, b, 10, fill=INK2)
    s.text(380, 24, 'От графических шейдеров к вычислительным ускорителям', 13, weight=700)
    s.text(380, 286, 'до 2006 г. GPGPU — «обман» графического конвейера: данные в текстурах, расчёт в шейдерах',
           11, fill=INK2)
    s.save('sv_timeline')


def d_coalesce():
    s = Svg(760, 300)
    s.text(380, 20, 'Обращения варпа к глобальной памяти', 13, weight=700)
    for row, (title, stride, c) in enumerate([('Объединённое: нить i → адрес i (a[i])', 1, 'green'),
                                              ('Со страйдом: нить i → адрес 8·i (a[8·i])', 8, 'red')]):
        y = 44 + row * 128
        s.text(20, y, title, 12, anchor='start', weight=600)
        for t in range(8):
            s.box(20 + t * 46, y + 10, 40, 26, [f'n{t}'], 'white', size=10.5)
        for seg in range(4):
            s.rect(20 + seg * 184, y + 74, 176, 26, 'gray', rx=3)
            s.text(108 + seg * 184, y + 92, f'сегмент 128 Б № {seg}', 10.5, fill=INK2)
        for t in range(8):
            ax = 20 + t * 46 + 20
            addr = t * stride
            bx = 20 + (addr // 32) * 184 + 6 + (addr % 32) * 5.3
            s.line(ax, y + 36, bx, y + 74, color=Svg.PAL[c][1], sw=1.2)
        s.text(740, y + 30, '1 транзакция' if stride == 1 else '2+ транзакций', 12, anchor='end', weight=700,
               fill=Svg.PAL[c][1])
        s.text(740, y + 48, '(для 32 нитей)' if stride == 1 else 'на 32 нити — 8 транзакций', 10.5,
               anchor='end', fill=INK2)
    s.text(380, 292, 'Соседние нити варпа должны читать соседние адреса — поэтому в работе 2 x-индекс нити = столбец',
           11.5, fill=INK2)
    s.save('sv_coalesce')


def d_divergence():
    s = Svg(760, 250)
    s.text(380, 20, 'Расхождение варпа: if (lane < 16) A(); else B();', 13, weight=700, mono=True)
    x0, cw = 110, 18
    s.text(100, 58, 'такт 1–k', 11, anchor='end', fill=INK2)
    s.text(100, 98, 'такт k+1–2k', 11, anchor='end', fill=INK2)
    for l in range(32):
        x = x0 + l * cw
        a = l < 16
        s.rect(x, 42, cw - 3, 24, 'green' if a else 'white', rx=2, sw=1)
        s.rect(x, 82, cw - 3, 24, 'white' if a else 'orange', rx=2, sw=1)
    s.text(x0 + 8 * cw, 138, 'A(): нити 0–15 активны, 16–31 замаскированы', 11.5, fill='#1d7a47')
    s.text(x0 + 24 * cw, 158, 'B(): наоборот', 11.5, fill='#c4501b')
    s.text(380, 198, 'Ветви выполняются по очереди — время ≈ A + B. Если условие одинаково для всего варпа', 11.5,
           fill=INK2)
    s.text(380, 216, '(например, зависит от номера варпа, а не нити), расхождения нет', 11.5, fill=INK2)
    s.text(380, 240, 'в работе 4 условие выхода одинаково для всех 32 нитей варпа — «варп выходит целиком»', 11.5,
           fill=INK2)
    s.save('sv_divergence')


def d_stack():
    s = Svg(760, 370)
    layers = [('Приложения пользователей', 'MPI + CUDA, PyTorch, пакеты (GROMACS, LAMMPS …)', 'white'),
              ('Менеджер ресурсов и планировщик', 'Torque/PBS + Maui/Moab, Slurm, LSF', 'amber'),
              ('Среда разработки и библиотеки', 'CUDA Toolkit, MPI (CUDA-aware), cuBLAS, cuDNN, NCCL, компиляторы',
               'green'),
              ('ОС узлов', 'Linux + драйвер NVIDIA, модули окружения (module load)', 'blue'),
              ('Оборудование', 'узлы CPU + GPU, InfiniBand, Ethernet, IPMI, СХД', 'gray')]
    for i, (a, b, c) in enumerate(layers):
        s.box(20, 14 + i * 64, 520, 56, [a, b], c, size=12)
    s.box(556, 14, 184, 120, ['Мониторинг', 'Ganglia, Nagios,', 'Zabbix, NVIDIA DCGM'], 'orange', size=12)
    s.box(556, 146, 184, 120, ['Развёртывание', 'и конфигурация', 'xCAT, Warewulf,', 'Base Command Manager'],
          'violet', size=12)
    s.box(556, 278, 184, 56, ['Учёт и пользователи', 'LDAP, accounting'], 'gray', size=11.5)
    s.text(380, 362, 'Система управления гибридным кластером — это стек, а не одна программа', 11.5, fill=INK2)
    s.save('sv_stack')


if __name__ == '__main__':
    for f in [chart_work1, chart_work2, chart_top10, chart_top1, chart_amdahl, chart_roofline,
              d_cpu_gpu, d_grid, d_memhier, d_sm, d_chip, d_toolchain, d_cluster, d_node, d_dataview, d_tiling,
              d_wmma, d_reduce, d_conv, d_torque, d_ganglia, d_backfill, d_cloud, d_timeline, d_coalesce,
              d_divergence, d_stack]:
        f()
    print('готово:', len(os.listdir(OUT)), 'файлов в', os.path.normpath(OUT))
