import os, json, glob, re, zipfile, numpy as np, pandas as pd, torch, torch.nn as nn, timm
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
                             roc_curve, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, log_loss)

W = "/kaggle/working"; PD = f"{W}/natural_preds"; CK = f"{W}/checkpoints"; OUT = f"{W}/paper_figures"
os.makedirs(OUT, exist_ok=True)
CLASSES = ['abdomen', 'brain', 'cervix', 'femur', 'thorax', 'other']
NAMES = ['Abdomen', 'Brain', 'Cervix', 'Femur', 'Thorax', 'Other']
M9 = ['densenet169', 'resnet50', 'efficientnet_b0', 'mobilenet_v2', 'convnext_tiny',
      'resnext50_32x4d', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'inception_resnet_v2']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e2dc'
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                            'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                            'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                            'savefig.bbox': 'tight'})
num = {}

idx = {s: pd.read_csv(f"{PD}/{s}_index.csv") for s in ['train', 'val', 'test']}
yv, yt = idx['val'].label.values, idx['test'].label.values
PV = {m: np.load(f"{PD}/{m}_val.npy") for m in M9}
PT = {m: np.load(f"{PD}/{m}_test.npy") for m in M9}
ens_v, ens_t = np.mean([PV[m] for m in M9], 0), np.mean([PT[m] for m in M9], 0)

def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def ece(y, p, bins=15):
    c, pr = p.max(1), p.argmax(1); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (c > lo) & (c <= hi)
        if m.any(): e += m.mean() * abs((pr[m] == y[m]).mean() - c[m].mean())
    return e
lv = np.log(np.clip(ens_v, 1e-12, 1))
Ts = np.linspace(0.3, 5, 471)
T = float(Ts[np.argmin([log_loss(yv, softmax(lv / t), labels=range(6)) for t in Ts])])
cal_t = softmax(np.log(np.clip(ens_t, 1e-12, 1)) / T)
num['calibration'] = dict(T=T, ece_raw=ece(yt, ens_t), ece_cal=ece(yt, cal_t),
                          nll_raw=log_loss(yt, ens_t, labels=range(6)), nll_cal=log_loss(yt, cal_t, labels=range(6)))

pred = ens_t.argmax(1)
p, r, f, s = precision_recall_fscore_support(yt, pred, labels=range(6))
auc_c = [roc_auc_score((yt == i).astype(int), ens_t[:, i]) for i in range(6)]
cm = confusion_matrix(yt, pred, labels=range(6))
num['headline'] = dict(n_test=int(len(yt)), acc=accuracy_score(yt, pred), balanced_acc=balanced_accuracy_score(yt, pred),
                       macro_f1=f1_score(yt, pred, average='macro'), weighted_f1=f1_score(yt, pred, average='weighted'),
                       macro_precision=float(p.mean()), macro_recall=float(r.mean()), mcc=matthews_corrcoef(yt, pred),
                       auc_macro=float(np.mean(auc_c)), n_errors=int((pred != yt).sum()))
num['per_class'] = {NAMES[i]: dict(support=int(s[i]), precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                                   auc=float(auc_c[i])) for i in range(6)}
num['confusion_matrix'] = cm.tolist()
off = [(cm[i, j], NAMES[i], NAMES[j]) for i in range(6) for j in range(6) if i != j]
num['largest_confusions'] = [dict(true=a, pred=b, n=int(n)) for n, a, b in sorted(off, reverse=True)[:5]]

H = -(cal_t * np.log(cal_t + 1e-12)).sum(1); order = np.argsort(H); corr = (pred[order] == yt[order])
n_err = (~corr).sum(); rej = {}
for keep in [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]:
    k = int(round(keep * len(order)))
    rej[f"keep_{int(keep*100)}"] = dict(acc=float(corr[:k].mean()), errors_flagged_share=float((~corr[k:]).sum() / n_err))
num['rejection'] = rej

fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(6); wd = 0.26
for k, (sp, col) in enumerate(zip(['train', 'val', 'test'], SERIES[:3])):
    c = np.bincount(idx[sp].label.values, minlength=6)
    b = ax.bar(x + (k - 1) * (wd + 0.02), c, wd, color=col, label=f"{sp.capitalize()} ({c.sum():,} images, "
               f"{idx[sp].patient.nunique()} patients)", zorder=3)
ax.set_xticks(x, NAMES); ax.set_ylabel("Images"); ax.grid(axis='y', color=GRID, zorder=0)
ax.legend(frameon=False, fontsize=8, loc='upper left')
plt.savefig(f"{OUT}/fig1_class_distribution.png"); plt.close()

fig, ax = plt.subplots(figsize=(4.8, 4.2))
rowp = cm / cm.sum(1, keepdims=True)
ax.imshow(rowp, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#3987e5', '#104281']),
          vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cm[i, j]}\n{rowp[i, j]*100:.1f}%" if cm[i, j] else "0", ha='center', va='center', fontsize=7,
                color='white' if rowp[i, j] > 0.55 else INK)
ax.set_xticks(range(6), NAMES, rotation=30, ha='right'); ax.set_yticks(range(6), NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.spines[:].set_visible(False)
plt.savefig(f"{OUT}/fig_confusion_matrix.png"); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 4.2))
for i in range(6):
    fp, tp, _ = roc_curve((yt == i).astype(int), ens_t[:, i])
    ax.plot(fp, tp, lw=2, color=SERIES[i], label=f"{NAMES[i]} (AUC {auc_c[i]:.4f})")
ax.plot([0, 1], [0, 1], ls='--', lw=1, color=MUTED)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(color=GRID)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
ins = ax.inset_axes([0.38, 0.42, 0.34, 0.3])
for i in range(6):
    fp, tp, _ = roc_curve((yt == i).astype(int), ens_t[:, i]); ins.plot(fp, tp, lw=1.5, color=SERIES[i])
ins.set_xlim(0, 0.1); ins.set_ylim(0.85, 1.0); ins.tick_params(labelsize=6)
plt.savefig(f"{OUT}/fig_roc.png"); plt.close()

keeps = np.linspace(0.5, 1.0, 51)
accs = [corr[:int(round(k * len(order)))].mean() for k in keeps]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(keeps * 100, np.array(accs) * 100, lw=2, color=SERIES[0])
for k in [0.9, 0.8]:
    a = corr[:int(round(k * len(order)))].mean() * 100
    ax.plot(k * 100, a, 'o', ms=8, color=SERIES[0], mec='white', mew=2)
    ax.annotate(f"{a:.2f}% at {int(k*100)}% kept", (k * 100, a), xytext=(6, -14), textcoords='offset points',
                ha='left', fontsize=7.5, color=INK)
ax.invert_xaxis(); ax.set_xlabel("Predictions kept (%) — least confident removed first")
ax.set_ylabel("Accuracy on kept (%)"); ax.grid(color=GRID)
plt.savefig(f"{OUT}/fig_rejection.png"); plt.close()

def load_E(path, tag):
    E = pd.read_csv(path); E['run'] = tag; return E
E1 = load_E(f"{W}/revision_outputs/E_main_ablation_table.csv", "original top 5")
rows = [("Best single CNN (EfficientNet-B5)", E1, "single: tf_efficientnet_b5_ns"),
        ("All 9 CNNs, equal weights", E1, "all9 equal-weight"),
        ("Top-5 CNNs, equal weights", E1, "top5 equal-weight"),
        ("Top-5, PSO weights", E1, "top5 PSO"),
        ("Top-5, random-search weights", E1, "top5 random-simplex search (equal budget)"),
        ("Top-5, logistic stacking", E1, "top5 logistic stacking"),
        ("Top-5 + KAN head, equal weights", E1, "top5+KAN equal-weight"),
        ("Top-5 + KAN head, PSO weights", E1, "top5+KAN PSO")]
p2 = f"{W}/revision_outputs_reselected/E_main_ablation_table.csv"
if os.path.exists(p2):
    E2 = load_E(p2, "re-selected"); rows.insert(3, ("Top-5 re-selected on validation, equal", E2, "top5 equal-weight"))
abl = []
for lab, E, key in rows:
    r_ = E[E.config == key].iloc[0]
    rng_ = lambda t: tuple(map(float, re.fullmatch(r"\s*(-?\d+\.\d+)-(-?\d+\.\d+)\s*", str(t)).groups()))
    lo, hi = rng_(r_.macroF1_95ci)
    d = rng_(r_['dMacroF1_vs_best_single_95ci'])
    abl.append(dict(label=lab, macro_f1=float(r_.test_macro_f1), lo=lo, hi=hi, acc=float(r_.test_acc),
                    acc_ci=r_.acc_95ci, mcc=float(r_.test_mcc), auc=float(r_.test_auc), val_macro_f1=float(r_.val_macro_f1),
                    d_lo=d[0], d_hi=d[1]))
A = pd.DataFrame(abl); A.to_csv(f"{OUT}/table_ablation.csv", index=False)
num['ablation'] = abl
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(A) + 0.8))
yy = np.arange(len(A))[::-1]
ref = A.macro_f1.iloc[0]
ax.axvline(ref, color=MUTED, lw=1, ls='--', zorder=1)
for y_, (_, r_) in zip(yy, A.iterrows()):
    col = SERIES[0] if 'equal' in r_.label and 'KAN' not in r_.label else (MUTED if 'single' in r_.label else SERIES[1])
    ax.plot([r_.lo, r_.hi], [y_, y_], color=col, lw=2, zorder=2)
    ax.plot(r_.macro_f1, y_, 'o', ms=8, color=col, mec='white', mew=2, zorder=3)
    ax.text(r_.hi + 0.0015, y_, f"{r_.macro_f1:.4f}", va='center', fontsize=7.5, color=INK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=MUTED, marker='o', lw=2, label='Best single CNN (reference)'),
                   Line2D([], [], color=SERIES[0], marker='o', lw=2, label='Equal-weight average of CNNs'),
                   Line2D([], [], color=SERIES[1], marker='o', lw=2, label='Learned weights, stacking or extra KAN head')],
          frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.6 / len(A)), ncol=3)
ax.set_yticks(yy, A.label); ax.set_xlabel("Test macro-F1 (95% CI, patient-level bootstrap)")
ax.grid(axis='x', color=GRID); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
plt.savefig(f"{OUT}/fig_ablation.png"); plt.close()

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c//r, 1), nn.ReLU(), nn.Conv2d(c//r, c, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(c, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        if len(x.shape) == 2: return x
        x = x * self.ca(x); return x * self.sa(x)
class SAF(nn.Module):
    def __init__(self, c):
        super().__init__(); self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c); self.bn = nn.BatchNorm2d(c); self.act = nn.SiLU()
    def forward(self, x):
        if len(x.shape) == 2: return x
        return self.act(self.bn(self.conv(x)))
class Hybrid(nn.Module):
    def __init__(self, b, c, d):
        super().__init__(); self.backbone = b; self.custom = c; self.classifier = nn.Linear(d, 6)
    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, tuple): f = f[0]
        if len(f.shape) == 4:
            f = self.custom(f); f = f.mean(dim=[2, 3])
        else:
            f = self.custom(f)
        return self.classifier(f)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bb = timm.create_model('tf_efficientnet_b5_ns', pretrained=False, num_classes=0)
model = Hybrid(bb, nn.Sequential(SAF(2048), CBAM(2048)), 2048).to(dev)
model.load_state_dict(torch.load(f"{CK}/models/tf_efficientnet_b5_ns.pt", map_location=dev)); model.eval()
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])
fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
rng = np.random.default_rng(0)
for i, ax in enumerate(axes):
    cand = np.where((yt == i) & (pred == i))[0]; j = int(rng.choice(cand))
    name = str(idx['test'].image_name.iloc[j]); path = f"{W}/data/{CLASSES[i]}/{name}.png"
    img = Image.open(path).convert('RGB'); x_ = tf(img).unsqueeze(0).to(dev)
    g = cam(input_tensor=x_, targets=None)[0]
    ax.imshow(show_cam_on_image(np.asarray(img, dtype=np.float32) / 255, g, use_rgb=True)); ax.axis('off')
    ax.set_title(NAMES[i], fontsize=9, color=INK)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_gradcam_test.png"); plt.close()

probe = torch.randn(2, 3, 224, 224, device=dev)
with torch.no_grad():
    feats = model.backbone(probe)
    num['attention_check'] = dict(backbone_output_shape=list(feats.shape),
                                  custom_module_changes_features=bool((model.custom(feats) - feats).abs().max() > 0))

import torchvision.models as tvm
def eff_params(name):
    if name in ['densenet169', 'resnet50', 'efficientnet_b0', 'mobilenet_v2', 'resnext50_32x4d']:
        b = getattr(tvm, name)(weights=None)
        if hasattr(b, 'fc') and name != 'densenet169' and name in ['resnet50', 'resnext50_32x4d']: d = b.fc.in_features; b.fc = nn.Identity()
        elif name == 'densenet169': d = b.classifier.in_features; b.classifier = nn.Identity()
        else: d = b.classifier[1].in_features; b.classifier = nn.Identity()
    else:
        b = timm.create_model(name, pretrained=False, num_classes=0); d = b.num_features
    return sum(p.numel() for p in b.parameters()) + d * 6 + 6
num['effective_params_M'] = {m: round(eff_params(m) / 1e6, 2) for m in M9}

lat = {}
for m in M9:
    if m in ['densenet169', 'resnet50', 'efficientnet_b0', 'mobilenet_v2', 'resnext50_32x4d']:
        b = getattr(tvm, m)(weights=None)
        if m == 'densenet169': b.classifier = nn.Linear(b.classifier.in_features, 6)
        elif m in ['resnet50', 'resnext50_32x4d']: b.fc = nn.Linear(b.fc.in_features, 6)
        else: b.classifier[1] = nn.Linear(b.classifier[1].in_features, 6)
    else:
        b = timm.create_model(m, pretrained=False, num_classes=6)
    b = b.to(dev).eval(); xb = torch.randn(16, 3, 224, 224, device=dev)
    with torch.no_grad():
        for _ in range(5): b(xb)
        if dev.type == 'cuda': torch.cuda.synchronize()
        import time; t0 = time.time()
        for _ in range(30): b(xb)
        if dev.type == 'cuda': torch.cuda.synchronize()
    lat[m] = (time.time() - t0) / (30 * 16) * 1000; del b
num['latency_ms_per_image_bs16'] = {m: round(v, 2) for m, v in lat.items()}
num['ensemble_all9'] = dict(params_M=round(sum(num['effective_params_M'].values()), 2),
                            ms_per_image_sequential=round(sum(lat.values()), 2),
                            gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu')

tr = {}
for m in M9:
    h = json.load(open(f"{CK}/history/{m}.json")) if os.path.exists(f"{CK}/history/{m}.json") else None
    hp = json.load(open(f"{CK}/hyperparams/{m}.json")) if os.path.exists(f"{CK}/hyperparams/{m}.json") else {}
    tr[m] = dict(lr=hp.get('lr'), wd=hp.get('wd'),
                 epochs_run=len(h['val_acc']) if h else None,
                 best_val_acc_epoch=int(np.argmax(h['val_acc'])) + 1 if h else None,
                 val_macro_f1=f1_score(yv, PV[m].argmax(1), average='macro'),
                 test_macro_f1=f1_score(yt, PT[m].argmax(1), average='macro'),
                 test_acc=accuracy_score(yt, PT[m].argmax(1)))
num['backbones'] = tr

json.dump(num, open(f"{OUT}/paper_numbers.json", "w"), indent=2, default=float)
with zipfile.ZipFile(f"{W}/paper_figures.zip", "w") as z:
    for fpath in glob.glob(f"{OUT}/*"): z.write(fpath, os.path.basename(fpath))
print(json.dumps(num, indent=1, default=float))
print("\nDownload /kaggle/working/paper_figures.zip (figures + paper_numbers.json)")
