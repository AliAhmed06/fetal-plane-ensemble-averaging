import os, gc, json, pickle, warnings, random, numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import pyswarms as ps
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
gc.collect(); torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 10
HP_PSO_PARTICLES = 4
HP_PSO_ITERS = 2
TARGET_PER_CLASS = 3080
SPLIT_VERSION = "v4_simple_patient_70_10_20"

CLASSES = ['abdomen', 'brain', 'cervix', 'femur', 'thorax', 'other']
NUM_CLASSES = len(CLASSES)
CLASS_NAMES_DISPLAY = [c.capitalize() for c in CLASSES]

CKPT_DIR = "/kaggle/working/checkpoints"
FIG_DIR  = "/kaggle/working/figures"
for d in [CKPT_DIR, f"{CKPT_DIR}/models", f"{CKPT_DIR}/probs", f"{CKPT_DIR}/probs_val",
          f"{CKPT_DIR}/history", f"{CKPT_DIR}/hyperparams", FIG_DIR]:
    os.makedirs(d, exist_ok=True)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction, 1),
                                 nn.ReLU(), nn.Conv2d(channels//reduction, channels, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=7, padding=3), nn.Sigmoid())
    def forward(self, x):
        if len(x.shape) == 2: return x
        x = x * self.ca(x); x = x * self.sa(x)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, 1)
        self.conv3 = nn.Conv2d(in_ch, in_ch//2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, in_ch//2, 5, padding=2)
        self.fuse = nn.Conv2d(in_ch//2 * 3, in_ch, 1)
    def forward(self, x):
        if len(x.shape) == 2: return x
        return self.fuse(torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1))

class SpeckleAdaptiveFilter(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()
    def forward(self, x):
        if len(x.shape) == 2: return x
        return self.act(self.bn(self.conv(x)))

def get_feat_dim(backbone, model_name=""):
    if hasattr(backbone, 'num_features'): return backbone.num_features
    if hasattr(backbone, 'head') and hasattr(backbone.head, 'fc') and hasattr(backbone.head.fc, 'in_features'):
        return backbone.head.fc.in_features
    if hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
        return backbone.classifier.in_features
    if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
        return backbone.fc.in_features
    name_lower = model_name.lower()
    if 'convnext_tiny' in name_lower: return 768
    if 'efficientnet_b4' in name_lower: return 1792
    if 'efficientnet_b5' in name_lower: return 2048
    if 'inception_resnet_v2' in name_lower: return 1536
    if 'resnext' in name_lower: return 2048
    raise ValueError(f"Could not determine feature dim for {model_name}")

class HybridModel(nn.Module):
    def __init__(self, backbone, custom, feat_dim, num_classes):
        super().__init__()
        self.backbone = backbone; self.custom = custom
        self.classifier = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, tuple): feats = feats[0]
        if len(feats.shape) == 4:
            feats = self.custom(feats); feats = feats.mean(dim=[2, 3])
        else:
            feats = self.custom(feats)
        return self.classifier(feats)

_pretrained_backbone_cache = {}

def create_model(name, num_classes=NUM_CLASSES):
    cached = _pretrained_backbone_cache.get(name)
    use_pretrained = cached is None

    if name == 'densenet169':
        backbone = models.densenet169(pretrained=use_pretrained); feat_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity(); custom = CBAM(feat_dim)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=use_pretrained); feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity(); custom = CBAM(feat_dim)
    elif name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=use_pretrained); feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity(); custom = CBAM(feat_dim)
    elif name == 'mobilenet_v2':
        backbone = models.mobilenet_v2(pretrained=use_pretrained); feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity(); custom = SpeckleAdaptiveFilter(feat_dim)
    elif name == 'convnext_tiny':
        backbone = timm.create_model('convnext_tiny', pretrained=use_pretrained, num_classes=0)
        feat_dim = get_feat_dim(backbone, name); custom = CBAM(feat_dim)
    elif name == 'resnext50_32x4d':
        backbone = models.resnext50_32x4d(pretrained=use_pretrained); feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity(); custom = CBAM(feat_dim)
    elif name == 'tf_efficientnet_b4_ns':
        backbone = timm.create_model('tf_efficientnet_b4_ns', pretrained=use_pretrained, num_classes=0)
        feat_dim = get_feat_dim(backbone, name); custom = MultiScaleFusion(feat_dim)
    elif name == 'tf_efficientnet_b5_ns':
        backbone = timm.create_model('tf_efficientnet_b5_ns', pretrained=use_pretrained, num_classes=0)
        feat_dim = get_feat_dim(backbone, name); custom = nn.Sequential(SpeckleAdaptiveFilter(feat_dim), CBAM(feat_dim))
    elif name == 'inception_resnet_v2':
        backbone = timm.create_model('inception_resnet_v2', pretrained=use_pretrained, num_classes=0)
        feat_dim = get_feat_dim(backbone, name); custom = CBAM(feat_dim)
    else:
        raise ValueError(f"Unknown model: {name}")

    if use_pretrained:
        _pretrained_backbone_cache[name] = {k: v.clone() for k, v in backbone.state_dict().items()}
    else:
        backbone.load_state_dict(cached)

    return HybridModel(backbone, custom, feat_dim, num_classes).to(device)

RAW_CSV = "/kaggle/input/fetal-planes-db/FETAL_PLANES_DB_data.csv"
RAW_IMG_DIR = "/kaggle/input/fetal-planes-db/Images/"

def prepare_data():
    base = '/kaggle/working/data'
    os.makedirs(base, exist_ok=True)
    for c in CLASSES: os.makedirs(f"{base}/{c}", exist_ok=True)
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"CSV not found at: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, sep=';')
    mapping = {'Fetal abdomen': 'abdomen', 'Fetal brain': 'brain', 'Maternal cervix': 'cervix',
               'Fetal femur': 'femur', 'Fetal thorax': 'thorax', 'Other': 'other'}
    df['cls'] = df['Plane'].map(mapping)
    df = df.dropna(subset=['cls'])
    for _, r in df.iterrows():
        p = os.path.join(RAW_IMG_DIR, f"{r['Image_name']}.png")
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('L')
                cropped = img.crop((35, 20, img.width - 40, img.height - 20)).resize((224, 224), Image.LANCZOS)
                cropped.save(f"{base}/{r['cls']}/{r['Image_name']}.png")
            except Exception:
                continue
    return base, df

def find_column(df, candidates):
    normalized = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    return None

print("Regenerating preprocessed image folder from raw input (idempotent)...")
data_dir, raw_df = prepare_data()
print("CSV columns detected:", list(raw_df.columns))

state_path = f"{CKPT_DIR}/.pipeline_state.json"
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
if state.get('split_version') != SPLIT_VERSION:
    print(f"Pipeline state marker shows split_version={state.get('split_version')!r}, "
          f"current code expects {SPLIT_VERSION!r}.")
    print("Clearing all downstream artifacts that could be stale under a different split "
          "(hyperparameters are preserved)...")
    import glob
    patterns_to_clear = [
        f"{CKPT_DIR}/models/*.pt", f"{CKPT_DIR}/probs/*.npz", f"{CKPT_DIR}/probs_val/*.npz",
        f"{CKPT_DIR}/history/*.json", f"{CKPT_DIR}/b5_features.npz", f"{CKPT_DIR}/kan_model.pt",
        f"{CKPT_DIR}/kan_model_partial.pt", f"{CKPT_DIR}/kan_optimizer_partial.pt",
        f"{CKPT_DIR}/kan_history_partial.json", f"{CKPT_DIR}/individual_f1.json",
        f"{CKPT_DIR}/final_results.json", f"{CKPT_DIR}/ablation_and_stats.json",
        f"{CKPT_DIR}/dataset_split.pkl", f"{CKPT_DIR}/labels_global.npy", f"{CKPT_DIR}/labels_global_val.npy",
    ]
    n_removed = 0
    for pattern in patterns_to_clear:
        for fp in glob.glob(pattern):
            os.remove(fp); n_removed += 1
    print(f"Removed {n_removed} stale file(s). Hyperparameters in checkpoints/hyperparams/ were kept.")
    json.dump({'split_version': SPLIT_VERSION}, open(state_path, 'w'))
else:
    print("Pipeline state matches the current split version, existing checkpoints are safe to reuse.")

split_path = f"{CKPT_DIR}/dataset_split.pkl"
existing_split = None
if os.path.exists(split_path):
    with open(split_path, "rb") as f:
        existing_split = pickle.load(f)
    if existing_split.get('split_version') != SPLIT_VERSION:
        print(f"Found an incompatible dataset_split.pkl (version={existing_split.get('split_version')!r}).")
        print(f"Rebuilding the patient-level split ({SPLIT_VERSION}).")
        existing_split = None
    else:
        print("Found existing dataset split checkpoint (correct version), reusing it.")

if existing_split is not None:
    samples = existing_split['samples']; sample_labels = existing_split['labels']
    train_idx = existing_split['train_idx']; val_idx = existing_split['val_idx']; test_idx = existing_split['test_idx']
    balanced_counts = existing_split['balanced_counts']; original_counts = existing_split['original_counts']
else:
    patient_col = find_column(raw_df, ['Patient_num', 'Patient_number', 'Patient_Number',
                                        'patient_number', 'PatientID', 'Patient_ID', 'patient'])
    if patient_col is None:
        raise RuntimeError(
            "Could not find a patient-identifier column in the CSV (checked common name variants). "
            f"Actual columns are: {list(raw_df.columns)}. Add the real column name to the "
            "candidate list in find_column() before proceeding."
        )
    print(f"Using '{patient_col}' for patient-level splitting.")

    img_records = []
    for _, r in raw_df.iterrows():
        cls = r.get('cls')
        if pd.isna(cls):
            continue
        fpath = f"{data_dir}/{cls}/{r['Image_name']}.png"
        if not os.path.exists(fpath):
            continue
        img_records.append((fpath, CLASSES.index(cls), r[patient_col]))

    paths_all = np.array([r[0] for r in img_records])
    labels_all = np.array([r[1] for r in img_records])
    groups_all = np.array([r[2] for r in img_records])

    original_counts = pd.Series([CLASSES[l] for l in labels_all]).value_counts().reindex(CLASSES).fillna(0).astype(int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    trainval_idx0, test_idx0 = next(gss1.split(paths_all, labels_all, groups=groups_all))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
    train_idx0_rel, val_idx0_rel = next(gss2.split(paths_all[trainval_idx0], labels_all[trainval_idx0],
                                                     groups=groups_all[trainval_idx0]))
    train_idx0 = trainval_idx0[train_idx0_rel]
    val_idx0 = trainval_idx0[val_idx0_rel]

    g_train, g_val, g_test = set(groups_all[train_idx0]), set(groups_all[val_idx0]), set(groups_all[test_idx0])
    assert not (g_train & g_val), "Patient overlap between train and val"
    assert not (g_train & g_test), "Patient overlap between train and test"
    assert not (g_val & g_test), "Patient overlap between val and test"
    print(f"Patient-disjoint split confirmed: {len(g_train)} train patients, "
          f"{len(g_val)} val patients, {len(g_test)} test patients, zero overlap.")

    def oversample_within_split(idx_subset, target_per_class):
        samples_local, labels_local = [], []
        for ci in range(NUM_CLASSES):
            class_paths = paths_all[idx_subset][labels_all[idx_subset] == ci].tolist()
            n = len(class_paths)
            if n == 0:
                continue
            reps = (target_per_class // n) + 1
            repeated = (class_paths * reps)[:target_per_class]
            samples_local.extend(repeated)
            labels_local.extend([ci] * len(repeated))
        return samples_local, labels_local

    train_target = int(round(TARGET_PER_CLASS * 70 / 100))
    val_target = int(round(TARGET_PER_CLASS * 10 / 100))
    test_target = int(round(TARGET_PER_CLASS * 20 / 100))
    train_samples, train_labels = oversample_within_split(train_idx0, train_target)
    val_samples, val_labels_ = oversample_within_split(val_idx0, val_target)
    test_samples, test_labels_ = oversample_within_split(test_idx0, test_target)

    samples = train_samples + val_samples + test_samples
    sample_labels = train_labels + val_labels_ + test_labels_
    train_idx = np.arange(0, len(train_samples))
    val_idx = np.arange(len(train_samples), len(train_samples) + len(val_samples))
    test_idx = np.arange(len(train_samples) + len(val_samples), len(samples))

    balanced_counts = {CLASSES[i]: int(np.sum(np.array(train_labels) == i)) for i in range(NUM_CLASSES)}

    with open(split_path, "wb") as f:
        pickle.dump({'samples': samples, 'labels': sample_labels, 'train_idx': train_idx,
                     'val_idx': val_idx, 'test_idx': test_idx, 'balanced_counts': balanced_counts,
                     'original_counts': original_counts, 'classes': CLASSES,
                     'split_version': SPLIT_VERSION,
                     'n_train_patients': len(g_train), 'n_val_patients': len(g_val), 'n_test_patients': len(g_test)},
                    f)
    print(f"Saved dataset split checkpoint to {split_path}")
    print(f"  train: {len(train_samples)} images ({len(g_train)} patients)")
    print(f"  val:   {len(val_samples)} images ({len(g_val)} patients)")
    print(f"  test:  {len(test_samples)} images ({len(g_test)} patients)")

labels_global_val = np.array([sample_labels[i] for i in val_idx])
labels_global_test = np.array([sample_labels[i] for i in test_idx])
np.save(f"{CKPT_DIR}/labels_global_val.npy", labels_global_val)
np.save(f"{CKPT_DIR}/labels_global.npy", labels_global_test)

bc_series = pd.Series(balanced_counts).reindex(CLASSES)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(CLASS_NAMES_DISPLAY, pd.Series(original_counts).reindex(CLASSES).values, color=sns.color_palette("tab10", NUM_CLASSES))
axes[0].set_title("(a) Original class distribution"); axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(CLASS_NAMES_DISPLAY, bc_series.values, color=sns.color_palette("tab10", NUM_CLASSES))
axes[1].set_title("(b) Balanced training-set distribution"); axes[1].tick_params(axis='x', rotation=30)
plt.suptitle("Fig. 1: Class distribution of fetal ultrasound plane categories")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_class_distribution.png", dpi=300, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(3*NUM_CLASSES, 3))
for i, c in enumerate(CLASSES):
    cls_files = [s for s, l in zip(samples, sample_labels) if l == i]
    if cls_files:
        axes[i].imshow(Image.open(cls_files[0]), cmap='gray')
    axes[i].set_title(f"({chr(97+i)}) {c.capitalize()}"); axes[i].axis('off')
plt.suptitle("Fig. 2: Sample images of preprocessed dataset")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig2_sample_images.png", dpi=300, bbox_inches='tight'); plt.close()
print("Saved Fig 1 and Fig 2.")

train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                                transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class IndexedDataset(Dataset):
    def __init__(self, idx, tf):
        self.idx, self.tf = idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        img = Image.open(samples[j]).convert('RGB')
        return self.tf(img), sample_labels[j]

train_loader = DataLoader(IndexedDataset(train_idx, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(IndexedDataset(val_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(IndexedDataset(test_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def optimize_hyperparams(name):
    print(f"-> Running PSO for {name}...")
    def fitness(particles):
        costs = []
        for p in particles:
            lr, wd = 10 ** p[0], 10 ** p[1]
            model = create_model(name)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x.to(device)); preds.append(out.argmax(1).cpu()); trues.append(y)
            costs.append(-f1_score(torch.cat(trues), torch.cat(preds), average='weighted'))
            del model; gc.collect(); torch.cuda.empty_cache()
        return np.array(costs)
    opt = ps.single.GlobalBestPSO(n_particles=HP_PSO_PARTICLES, dimensions=2, options={'c1':0.7,'c2':0.9,'w':0.85},
                                   bounds=([-5,-6],[-2,-3]))
    _, pos = opt.optimize(fitness, iters=HP_PSO_ITERS, verbose=False)
    return 10 ** pos[0], 10 ** pos[1]

model_list = ['densenet169', 'resnet50', 'efficientnet_b0', 'mobilenet_v2',
              'convnext_tiny', 'resnext50_32x4d', 'tf_efficientnet_b4_ns',
              'tf_efficientnet_b5_ns', 'inception_resnet_v2']

f1_path = f"{CKPT_DIR}/individual_f1.json"
individual_f1 = json.load(open(f1_path)) if os.path.exists(f1_path) else {}

def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_mode: optimizer.zero_grad()
            out = model(x); loss = criterion(out, y)
            if train_mode: loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return total_loss / n, correct / n

for name in model_list:
    probs_path = f"{CKPT_DIR}/probs/{name}.npz"
    if os.path.exists(probs_path) and name in individual_f1 and 'val_f1' in individual_f1.get(name, {}):
        print(f"[skip] {name} already checkpointed (val_f1={individual_f1[name]['val_f1']:.4f}, "
              f"test_f1={individual_f1[name]['test_f1']:.4f}).")
        continue

    print(f"\n{'='*70}\nProcessing: {name}  (this run will stop once this ONE model finishes)\n{'='*70}")

    hp_path = f"{CKPT_DIR}/hyperparams/{name}.json"
    if os.path.exists(hp_path):
        hp = json.load(open(hp_path))
        best_lr, best_wd = hp['lr'], hp['wd']
        print(f"Loaded cached hyperparameters for {name}: lr={best_lr:.2e}, wd={best_wd:.2e}")
    else:
        best_lr, best_wd = optimize_hyperparams(name)
        json.dump({'lr': best_lr, 'wd': best_wd}, open(hp_path, 'w'))

    model = create_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)

    partial_model_path = f"{CKPT_DIR}/models/{name}_partial.pt"
    partial_opt_path   = f"{CKPT_DIR}/models/{name}_optimizer_partial.pt"
    partial_hist_path  = f"{CKPT_DIR}/history/{name}_partial.json"

    if os.path.exists(partial_model_path) and os.path.exists(partial_hist_path):
        print(f"Found a partial checkpoint for {name}, resuming instead of restarting.")
        model.load_state_dict(torch.load(partial_model_path, map_location=device))
        if os.path.exists(partial_opt_path):
            optimizer.load_state_dict(torch.load(partial_opt_path, map_location=device))
        hist = json.load(open(partial_hist_path))
        start_epoch = len(hist['train_acc'])
        best_val_acc = max(hist['val_acc']) if hist['val_acc'] else 0.0
        epochs_no_improve = (len(hist['val_acc']) - 1 - int(np.argmax(hist['val_acc']))) if hist['val_acc'] else 0
        print(f" -> resuming at epoch {start_epoch + 1}/{EPOCHS}, best_val_acc so far={best_val_acc:.4f}")
    else:
        hist = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        start_epoch, best_val_acc, epochs_no_improve = 0, 0.0, 0

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, None)
        hist['train_loss'].append(tr_loss); hist['train_acc'].append(tr_acc)
        hist['val_loss'].append(va_loss);   hist['val_acc'].append(va_acc)
        print(f" Epoch {epoch+1}/{EPOCHS}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        torch.save(model.state_dict(), partial_model_path)
        torch.save(optimizer.state_dict(), partial_opt_path)
        json.dump(hist, open(partial_hist_path, "w"))

        if va_acc > best_val_acc:
            best_val_acc, epochs_no_improve = va_acc, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f" Early stopping at epoch {epoch+1} (no val_acc improvement in {PATIENCE} epochs).")
                break

    model.eval()
    def get_raw_probs(loader):
        out_probs = []
        with torch.no_grad():
            for x, _ in loader:
                out_probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        return np.concatenate(out_probs)

    raw_probs_val = get_raw_probs(val_loader)
    raw_probs_test = get_raw_probs(test_loader)

    def cal_obj(w):
        costs = []
        for particle in w:
            scale = abs(particle[0]) + 1e-8; temp = abs(particle[1])
            scaled = (raw_probs_val ** (1/(1+temp))) * scale
            scaled /= scaled.sum(axis=1, keepdims=True)
            costs.append(-f1_score(labels_global_val, np.argmax(scaled, axis=1), average='weighted'))
        return np.array(costs)

    cal_opt = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options={'c1':0.7,'c2':0.9,'w':0.85},
                                       bounds=(np.zeros(2), np.array([8.0, 3.0])))
    _, best_pos = cal_opt.optimize(cal_obj, iters=10, verbose=False)
    scale, temp = abs(best_pos[0]), abs(best_pos[1])

    calibrated_val = (raw_probs_val ** (1/(1+temp))) * scale
    calibrated_val /= calibrated_val.sum(axis=1, keepdims=True)
    calibrated_test = (raw_probs_test ** (1/(1+temp))) * scale
    calibrated_test /= calibrated_test.sum(axis=1, keepdims=True)

    np.savez(f"{CKPT_DIR}/probs_val/{name}.npz", probs=calibrated_val)
    np.savez(probs_path, probs=calibrated_test)

    val_f1 = f1_score(labels_global_val, np.argmax(calibrated_val, axis=1), average='weighted')
    test_f1 = f1_score(labels_global_test, np.argmax(calibrated_test, axis=1), average='weighted')
    individual_f1[name] = {'val_f1': val_f1, 'test_f1': test_f1}
    json.dump(individual_f1, open(f1_path, "w"))

    torch.save(model.state_dict(), f"{CKPT_DIR}/models/{name}.pt")
    json.dump(hist, open(f"{CKPT_DIR}/history/{name}.json", "w"))
    for p in [partial_model_path, partial_opt_path, partial_hist_path]:
        if os.path.exists(p): os.remove(p)

    print(f"-> {name}  val_f1={val_f1:.4f}  test_f1={test_f1:.4f}  (checkpointed)\n")
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*70}\nStopping here on purpose, {name} is fully done and saved.")
    print("Save/persist the checkpoints/ folder now, then rerun this SAME cell")
    print("to continue with the next model.")
    print(f"{'='*70}")
    break
else:
    print("\nAll 9 CNNs are already trained.")

remaining = [m for m in model_list if not (os.path.exists(f"{CKPT_DIR}/probs/{m}.npz")
                                            and m in individual_f1 and 'val_f1' in individual_f1[m])]
if remaining:
    print(f"\nModels still remaining ({len(remaining)}/{len(model_list)}): {remaining}")
else:
    print("\nAll 9 CNNs trained and saved in:", CKPT_DIR)
