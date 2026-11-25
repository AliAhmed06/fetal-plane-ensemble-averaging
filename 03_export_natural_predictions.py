import os, json, pickle, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import timm
from sklearn.model_selection import GroupShuffleSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CKPT_DIR = "/kaggle/working/checkpoints"
OUT = "/kaggle/working/natural_preds"
os.makedirs(OUT, exist_ok=True)
CLASSES = ['abdomen', 'brain', 'cervix', 'femur', 'thorax', 'other']
NUM_CLASSES = len(CLASSES)
SPLIT_VERSION = "v4_simple_patient_70_10_20"
RAW_CSV = "/kaggle/input/fetal-planes-db/FETAL_PLANES_DB_data.csv"
RAW_IMG_DIR = "/kaggle/input/fetal-planes-db/Images/"
DATA_DIR = "/kaggle/working/data"
MODEL_LIST = ['densenet169', 'resnet50', 'efficientnet_b0', 'mobilenet_v2', 'convnext_tiny',
              'resnext50_32x4d', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'inception_resnet_v2']
MAPPING = {'Fetal abdomen': 'abdomen', 'Fetal brain': 'brain', 'Maternal cervix': 'cervix',
           'Fetal femur': 'femur', 'Fetal thorax': 'thorax', 'Other': 'other'}

def prepare_data():
    for c in CLASSES: os.makedirs(f"{DATA_DIR}/{c}", exist_ok=True)
    df = pd.read_csv(RAW_CSV, sep=';')
    df['cls'] = df['Plane'].map(MAPPING)
    df = df.dropna(subset=['cls'])
    for _, r in df.iterrows():
        dst = f"{DATA_DIR}/{r['cls']}/{r['Image_name']}.png"
        if os.path.exists(dst):
            continue
        p = os.path.join(RAW_IMG_DIR, f"{r['Image_name']}.png")
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('L')
                img.crop((35, 20, img.width - 40, img.height - 20)).resize((224, 224), Image.LANCZOS).save(dst)
            except Exception:
                continue
    return df

raw_df = prepare_data()
patient_col = 'Patient_num' if 'Patient_num' in raw_df.columns else [c for c in raw_df.columns if c.strip().lower().startswith('patient')][0]
recs = []
for _, r in raw_df.iterrows():
    fpath = f"{DATA_DIR}/{r['cls']}/{r['Image_name']}.png"
    if os.path.exists(fpath):
        recs.append((fpath, CLASSES.index(r['cls']), r[patient_col], r['Image_name'],
                     int(r['Train']) if 'Train' in raw_df.columns else -1))
paths_all = np.array([r[0] for r in recs]); labels_all = np.array([r[1] for r in recs])
groups_all = np.array([r[2] for r in recs]); names_all = np.array([r[3] for r in recs])
official_train = np.array([r[4] for r in recs])
print(f"{len(recs)} images, {len(set(groups_all))} patients")

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
trainval0, test0 = next(gss1.split(paths_all, labels_all, groups=groups_all))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
tr_rel, va_rel = next(gss2.split(paths_all[trainval0], labels_all[trainval0], groups=groups_all[trainval0]))
nat = {"train": trainval0[tr_rel], "val": trainval0[va_rel], "test": test0}

with open(f"{CKPT_DIR}/dataset_split.pkl", "rb") as f:
    sp = pickle.load(f)
assert sp['split_version'] == SPLIT_VERSION, "pickle is not the v4 split"
old = {k: [sp['samples'][i] for i in sp[f'{k}_idx']] for k in ["train", "val", "test"]}
print("\nVerification against dataset_split.pkl")
for k in ["train", "val", "test"]:
    nat_set, old_set = set(paths_all[nat[k]]), set(old[k])
    assert old_set <= nat_set, f"{k}: training pipeline used images outside the rebuilt split -> STOP, split differs"
    n_pat = len(set(groups_all[nat[k]]))
    exp = sp.get(f'n_{k}_patients')
    assert exp is None or n_pat == exp, f"{k}: patient count {n_pat} != pickle {exp}"
    print(f"  {k:5s}: natural {len(nat_set):5d} unique images ({n_pat} patients) | training pipeline "
          f"{len(old[k]):5d} rows = {len(old_set):5d} unique images "
          f"({len(old[k]) - len(old_set)} duplicate rows, {len(nat_set - old_set)} real images never used)")
for a in nat:
    for b in nat:
        if a < b:
            assert not set(groups_all[nat[a]]) & set(groups_all[nat[b]])
print("  Rebuild matches the pickle; zero patient overlap. Safe to evaluate the saved checkpoints.\n")

tab = pd.DataFrame({k: pd.Series(labels_all[nat[k]]).value_counts().reindex(range(6)).fillna(0).astype(int).values
                    for k in nat}, index=CLASSES)
tab["original"] = pd.Series(labels_all).value_counts().reindex(range(6)).values
for k in nat:
    tab[f"{k}_used_rows"] = pd.Series([sp['labels'][i] for i in sp[f'{k}_idx']]).value_counts().reindex(range(6)).fillna(0).astype(int).values
tab.loc["total"] = tab.sum()
tab.to_csv(f"{OUT}/table1_natural_vs_used.csv")
print(tab.to_string(), "\n")

for k in nat:
    pd.DataFrame({"image_name": names_all[nat[k]], "label": labels_all[nat[k]],
                  "patient": groups_all[nat[k]], "official_train": official_train[nat[k]]}) \
        .to_csv(f"{OUT}/{k}_index.csv", index=False)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction, 1),
                                nn.ReLU(), nn.Conv2d(channels//reduction, channels, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=7, padding=3), nn.Sigmoid())
    def forward(self, x):
        if len(x.shape) == 2: return x
        x = x * self.ca(x); return x * self.sa(x)

class SpeckleAdaptiveFilter(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, groups=ch); self.bn = nn.BatchNorm2d(ch); self.act = nn.SiLU()
    def forward(self, x):
        if len(x.shape) == 2: return x
        return self.act(self.bn(self.conv(x)))

class MultiScaleFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, 1); self.conv3 = nn.Conv2d(in_ch, in_ch//2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, in_ch//2, 5, padding=2); self.fuse = nn.Conv2d(in_ch//2 * 3, in_ch, 1)
    def forward(self, x):
        if len(x.shape) == 2: return x
        return self.fuse(torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1))

def get_feat_dim(backbone, name=""):
    if hasattr(backbone, 'num_features'): return backbone.num_features
    if hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'): return backbone.classifier.in_features
    if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'): return backbone.fc.in_features
    raise ValueError(name)

class HybridModel(nn.Module):
    def __init__(self, backbone, custom, feat_dim, num_classes):
        super().__init__()
        self.backbone = backbone; self.custom = custom; self.classifier = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple): feats = feats[0]
        if len(feats.shape) == 4:
            feats = self.custom(feats); feats = feats.mean(dim=[2, 3])
        else:
            feats = self.custom(feats)
        return self.classifier(feats)

def build_model(name):
    if name == 'densenet169':
        b = models.densenet169(weights=None); d = b.classifier.in_features; b.classifier = nn.Identity(); c = CBAM(d)
    elif name == 'resnet50':
        b = models.resnet50(weights=None); d = b.fc.in_features; b.fc = nn.Identity(); c = CBAM(d)
    elif name == 'efficientnet_b0':
        b = models.efficientnet_b0(weights=None); d = b.classifier[1].in_features; b.classifier = nn.Identity(); c = CBAM(d)
    elif name == 'mobilenet_v2':
        b = models.mobilenet_v2(weights=None); d = b.classifier[1].in_features; b.classifier = nn.Identity(); c = SpeckleAdaptiveFilter(d)
    elif name == 'convnext_tiny':
        b = timm.create_model('convnext_tiny', pretrained=False, num_classes=0); d = get_feat_dim(b); c = CBAM(d)
    elif name == 'resnext50_32x4d':
        b = models.resnext50_32x4d(weights=None); d = b.fc.in_features; b.fc = nn.Identity(); c = CBAM(d)
    elif name == 'tf_efficientnet_b4_ns':
        b = timm.create_model('tf_efficientnet_b4_ns', pretrained=False, num_classes=0); d = get_feat_dim(b); c = MultiScaleFusion(d)
    elif name == 'tf_efficientnet_b5_ns':
        b = timm.create_model('tf_efficientnet_b5_ns', pretrained=False, num_classes=0); d = get_feat_dim(b)
        c = nn.Sequential(SpeckleAdaptiveFilter(d), CBAM(d))
    elif name == 'inception_resnet_v2':
        b = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=0); d = get_feat_dim(b); c = CBAM(d)
    return HybridModel(b, c, d, NUM_CLASSES)

eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class PathDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return eval_tf(Image.open(self.paths[i]).convert('RGB'))

def loader(k):
    return DataLoader(PathDS(list(paths_all[nat[k]])), batch_size=32, shuffle=False, num_workers=2)

@torch.no_grad()
def run(model, k):
    return np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy() for x in loader(k)])

for name in MODEL_LIST:
    if os.path.exists(f"{OUT}/{name}_test.npy"):
        print(f"[skip] {name}"); continue
    m = build_model(name).to(device)
    m.load_state_dict(torch.load(f"{CKPT_DIR}/models/{name}.pt", map_location=device)); m.eval()
    for k in ["val", "test"]:
        np.save(f"{OUT}/{name}_{k}.npy", run(m, k))
    yv, yt = labels_all[nat["val"]], labels_all[nat["test"]]
    pv, pt = np.load(f"{OUT}/{name}_val.npy"), np.load(f"{OUT}/{name}_test.npy")
    from sklearn.metrics import f1_score
    print(f"{name:24s} natural val macro-F1={f1_score(yv, pv.argmax(1), average='macro'):.4f}  "
          f"natural test macro-F1={f1_score(yt, pt.argmax(1), average='macro'):.4f}  "
          f"acc={np.mean(pt.argmax(1) == yt):.4f}")
    del m; torch.cuda.empty_cache()

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_range=(-1, 1)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_noise, self.scale_base, self.scale_spline = scale_noise, scale_base, scale_spline
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                self.scale_spline * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        return torch.linalg.lstsq(A, B).solution.permute(2, 0, 1).contiguous()
    def b_splines(self, x):
        grid = self.grid; x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()
    def forward(self, x):
        base = F.linear(self.base_activation(x), self.base_weight)
        spl = F.linear(self.b_splines(x).view(x.size(0), -1),
                       (self.spline_weight * self.spline_scaler.unsqueeze(-1)).view(self.out_features, -1))
        return base + spl

class FastKAN(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([KANLinear(i, o) for i, o in zip(dims, dims[1:])])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

if not os.path.exists(f"{OUT}/b5emb_test.npy"):
    b5 = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=0).to(device).eval()
    with torch.no_grad():
        for k in ["train", "val", "test"]:
            np.save(f"{OUT}/b5emb_{k}.npy",
                    np.concatenate([b5(x.to(device)).cpu().numpy() for x in loader(k)]))
    del b5; torch.cuda.empty_cache()

E = {k: np.load(f"{OUT}/b5emb_{k}.npy") for k in ["train", "val", "test"]}
inside = ((E["train"] >= -2.2) & (E["train"] < 2.2)).mean()
print(f"\nB5 feature range: min={E['train'].min():.3f} max={E['train'].max():.3f} "
      f"mean={E['train'].mean():.3f}; share of values inside the KAN spline support [-2.2, 2.2): {inside:.3f}; "
      f"inside the nominal grid [-1, 1]: {((E['train'] >= -1) & (E['train'] <= 1)).mean():.3f}")

KAN_PATH = f"{CKPT_DIR}/kan_model.pt"
kan = FastKAN([E["train"].shape[1], 128, NUM_CLASSES]).to(device)
if os.path.exists(KAN_PATH):
    kan.load_state_dict(torch.load(KAN_PATH, map_location=device))
    print("Loaded saved KAN checkpoint.")
else:
    print("kan_model.pt not found -> retraining the KAN as in 02_train_kan_head.py (seed 42).")
    row_of = {p: i for i, p in enumerate(paths_all[nat["train"]])}
    old_train_rows = np.array([row_of[p] for p in old["train"]])
    Xtr = torch.tensor(E["train"][old_train_rows], dtype=torch.float32, device=device)
    ytr = torch.tensor(np.array([sp['labels'][i] for i in sp['train_idx']]), dtype=torch.long, device=device)
    assert len(Xtr) == len(ytr) == 12936, f"expected 12,936 training rows, got {len(Xtr)}"
    torch.manual_seed(SEED); np.random.seed(SEED)
    kan = FastKAN([E["train"].shape[1], 128, NUM_CLASSES]).to(device)
    opt_k = torch.optim.AdamW(kan.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(6):
        kan.train(); perm = torch.randperm(len(Xtr), device=device); tot = 0.0
        for i in range(0, len(Xtr), 512):
            j = perm[i:i + 512]
            opt_k.zero_grad(); loss = crit(kan(Xtr[j]), ytr[j]); loss.backward(); opt_k.step()
            tot += loss.item()
        print(f"  KAN epoch {ep + 1}/6  total_loss={tot:.4f}")
    torch.save(kan.state_dict(), KAN_PATH)
    print(f"KAN saved to {KAN_PATH}")
kan.eval()
with torch.no_grad():
    for k in ["val", "test"]:
        np.save(f"{OUT}/KAN_{k}.npy",
                torch.softmax(kan(torch.tensor(E[k], dtype=torch.float32, device=device)), 1).cpu().numpy())
yt = labels_all[nat["test"]]
print(f"KAN natural test acc={np.mean(np.load(f'{OUT}/KAN_test.npy').argmax(1) == yt):.4f}")

json.dump({
    "META_CSV": RAW_CSV,
    "PRED_DIR": OUT,
    "OUT_DIR": "/kaggle/working/revision_outputs",
}, open("/kaggle/working/audit_config.json", "w"), indent=2)
print("\nDone. Next: python 04_ablation_study.py /kaggle/working/audit_config.json")
