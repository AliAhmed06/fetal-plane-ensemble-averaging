import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']; INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e2dc'
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
NAMES = ['Abdomen', 'Brain', 'Cervix', 'Femur', 'Thorax', 'Other']
C = {'Train': ([509, 2138, 1106, 756, 1220, 3152], 1253), 'Validation': ([78, 367, 187, 105, 189, 396], 180), 'Test': ([124, 587, 333, 179, 309, 665], 359)}
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(6); wd = 0.26
for k, (sp, (c, pt)) in enumerate(C.items()):
    ax.bar(x + (k - 1) * (wd + 0.02), c, wd, color=SERIES[k], label=f"{sp} ({sum(c):,} images, {pt:,} patients)", zorder=3)
ax.set_xticks(x, NAMES); ax.set_ylabel("Images"); ax.grid(axis='y', color=GRID, zorder=0)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
ax.legend(frameon=False, fontsize=8, loc='upper left')
plt.savefig("/kaggle/working/paper_figures/fig1_class_distribution.png"); plt.close()

A = pd.read_csv("/kaggle/working/paper_figures/table_ablation.csv")
lab = {"Best single CNN (EfficientNet-B5)": "Best single CNN (EfficientNet-B5)", "All 9 CNNs, equal weights": "All nine CNNs, equal weights",
       "Top-5 CNNs, equal weights": "Top five, equal weights", "Top-5 re-selected on validation, equal": "Re-selected top five, equal weights",
       "Top-5, PSO weights": "Top five, PSO weights", "Top-5, random-search weights": "Top five, random-search weights",
       "Top-5, logistic stacking": "Top five, logistic stacking", "Top-5 + KAN head, equal weights": "Top five + KAN head, equal weights",
       "Top-5 + KAN head, PSO weights": "Top five + KAN head, PSO weights"}
A['label'] = A['label'].map(lab)
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(A) + 0.8)); yy = np.arange(len(A))[::-1]
ax.axvline(A.macro_f1.iloc[0], color=MUTED, lw=1, ls='--', zorder=1)
for y_, (_, r) in zip(yy, A.iterrows()):
    col = MUTED if 'single' in r.label else (SERIES[0] if ('equal' in r.label and 'KAN' not in r.label) else SERIES[1])
    ax.plot([r.lo, r.hi], [y_, y_], color=col, lw=2, zorder=2)
    ax.plot(r.macro_f1, y_, 'o', ms=8, color=col, mec='white', mew=2, zorder=3)
    ax.text(r.hi + 0.0012, y_, f"{r.macro_f1:.4f}", va='center', fontsize=7.5, color=INK)
ax.legend(handles=[Line2D([], [], color=MUTED, marker='o', lw=2, label='Best single CNN (reference)'),
                   Line2D([], [], color=SERIES[0], marker='o', lw=2, label='Equal-weight average of CNNs'),
                   Line2D([], [], color=SERIES[1], marker='o', lw=2, label='Learned weights, stacking or KAN head')],
          frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.18), ncol=3)
ax.set_yticks(yy, A.label); ax.set_xlabel("Test macro-F1 (95% CI, patient-level bootstrap)")
ax.grid(axis='x', color=GRID); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
plt.savefig("/kaggle/working/paper_figures/fig_ablation.png"); plt.close(); print("ok")
