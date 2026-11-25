import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, LINE = "#1f1f1e", "#6b6a63", "#b9b7ae"
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_BG = "#eb6834", "#fdeee6"
GREY_BG = "#f4f3ef"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, title, lines, fc=GREY_BG, ec=LINE, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.2, "\n".join(lines), ha="center", va="top", fontsize=7.4, color=INK,
            linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=MUTED, lw=1.3, shrinkA=2, shrinkB=2))


box(1, 11, 15, 27, "Fetal Planes DB", ["12,400 images", "1,792 patients", "6 classes", "public Zenodo", "release"])
box(19, 11, 21, 27, "Patient-disjoint split", ["by patient identifier", "", "Train: 1,253 patients,", "8,881 images",
                                               "Val: 180 patients,", "1,322 images", "Test: 359 patients,", "2,197 images"])
box(43.5, 25, 22, 17, "Nine fine-tuned CNNs", ["ImageNet initialisation,", "linear classifier,",
                                               "per-backbone lr / wd"], fc=BLUE_BG, ec=BLUE)
box(43.5, 4, 22, 17, "Extra head (ablation)", ["frozen ImageNet", "EfficientNet-B5 features",
                                               "-> KAN, MLP or linear head"], fc=ORANGE_BG, ec=ORANGE)
box(69, 11, 20, 27, "Fusion (fit on val)", ["equal weights", "val-F1 proportional", "PSO weights",
                                             "random search,", "same budget", "logistic stacking"])
box(92, 11, 17, 27, "Test evaluation", ["each image once", "macro-F1, MCC, AUC", "patient-level", "bootstrap 95% CIs",
                                        "paired tests vs", "best single CNN"])

arrow(16.4, 24.5, 18.6, 24.5)
arrow(40.4, 27, 43.1, 33)
arrow(40.4, 23, 43.1, 13)
arrow(65.9, 33, 68.6, 27)
arrow(65.9, 13, 68.6, 23)
arrow(89.4, 24.5, 91.6, 24.5)
ax.text(55.5, 1.0, "Test patients are never used for training, model selection, weight fitting or calibration.",
        ha="center", fontsize=7.8, color=MUTED, style="italic")
plt.savefig("/kaggle/working/paper_figures/fig_design.png", dpi=300, bbox_inches="tight", facecolor="white")
print("ok")
