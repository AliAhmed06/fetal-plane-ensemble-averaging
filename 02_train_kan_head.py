import os, json, pickle, time, math, pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "/kaggle/working/checkpoints"
BATCH_SIZE = 32
KAN_EPOCHS = 6
KAN_BATCH = 512
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
CLASSES = ['abdomen', 'brain', 'cervix', 'femur', 'thorax', 'other']
SPLIT_VERSION = "v4_simple_patient_70_10_20"

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
    return base

print("Regenerating preprocessed image folder from raw input (idempotent)...")
prepare_data()

with open(f"{CKPT_DIR}/dataset_split.pkl", "rb") as f:
    split_data = pickle.load(f)
assert split_data.get('split_version') == SPLIT_VERSION, \
    "dataset_split.pkl does not match the current split. Run 01_train_cnns.py first."
samples, sample_labels = split_data['samples'], split_data['labels']
train_idx, val_idx, test_idx = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']
CLASSES = split_data['classes']
NUM_CLASSES = len(CLASSES)

eval_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class IndexedDataset(Dataset):
    def __init__(self, idx, tf):
        self.idx, self.tf = idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(Image.open(samples[j]).convert('RGB')), sample_labels[j]

train_loader = DataLoader(IndexedDataset(train_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
val_loader   = DataLoader(IndexedDataset(val_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(IndexedDataset(test_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

feat_path = f"{CKPT_DIR}/b5_features.npz"
if os.path.exists(feat_path):
    print("Found cached EfficientNet-B5 features, loading instead of re-extracting.")
    d = np.load(feat_path)
    train_feats, train_labs, val_feats, val_labs, test_feats, test_labs = \
        d['train_feats'], d['train_labs'], d['val_feats'], d['val_labs'], d['test_feats'], d['test_labs']
else:
    print("Extracting EfficientNet-B5 features (train/val/test)...")
    backbone = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=0).to(device)
    backbone.eval()

    def extract(loader):
        feats, labs = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(backbone(x.to(device)).detach().cpu().numpy())
                labs.append(np.array(y))
        return np.concatenate(feats), np.concatenate(labs)

    train_feats, train_labs = extract(train_loader)
    val_feats, val_labs = extract(val_loader)
    test_feats, test_labs = extract(test_loader)
    np.savez(feat_path, train_feats=train_feats, train_labs=train_labs,
              val_feats=val_feats, val_labs=val_labs, test_feats=test_feats, test_labs=test_labs)
    print(f"Saved features to {feat_path}")

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_range=(-1, 1)):
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
    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()
    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        return torch.linalg.lstsq(A, B).solution.permute(2, 0, 1).contiguous()
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)
    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                  self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

class FastKAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(in_f, out_f, grid_size=grid_size, spline_order=spline_order)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

feat_dim = train_feats.shape[1]
kan_path = f"{CKPT_DIR}/kan_model.pt"
partial_kan_path = f"{CKPT_DIR}/kan_model_partial.pt"
partial_kan_opt_path = f"{CKPT_DIR}/kan_optimizer_partial.pt"
partial_kan_hist_path = f"{CKPT_DIR}/kan_history_partial.json"

kan = FastKAN([feat_dim, 128, NUM_CLASSES], grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER).to(device)
print(f"KAN device check: {next(kan.parameters()).device}")

X_train_t = torch.tensor(train_feats, dtype=torch.float32).to(device)
y_train_t = torch.tensor(train_labs, dtype=torch.long).to(device)

if os.path.exists(kan_path):
    print("Found existing fully-trained KAN checkpoint, loading it instead of retraining.")
    kan.load_state_dict(torch.load(kan_path, map_location=device))
else:
    kan_opt = torch.optim.AdamW(kan.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(partial_kan_path) and os.path.exists(partial_kan_hist_path):
        print("Found a partial KAN checkpoint, resuming instead of restarting.")
        kan.load_state_dict(torch.load(partial_kan_path, map_location=device))
        if os.path.exists(partial_kan_opt_path):
            kan_opt.load_state_dict(torch.load(partial_kan_opt_path, map_location=device))
        kan_hist = json.load(open(partial_kan_hist_path))
        start_epoch = len(kan_hist['loss'])
        print(f" -> resuming at epoch {start_epoch + 1}/{KAN_EPOCHS}")
    else:
        kan_hist = {'loss': []}
        start_epoch = 0

    print("Training KAN branch...")
    n_batches_per_epoch = (X_train_t.size(0) + KAN_BATCH - 1) // KAN_BATCH
    for epoch in range(start_epoch, KAN_EPOCHS):
        kan.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0
        t0 = time.time()
        for b, i in enumerate(range(0, X_train_t.size(0), KAN_BATCH)):
            idx = perm[i:i+KAN_BATCH]
            kan_opt.zero_grad()
            loss = criterion(kan(X_train_t[idx]), y_train_t[idx])
            loss.backward(); kan_opt.step()
            epoch_loss += loss.item()
            if (b + 1) % 5 == 0 or (b + 1) == n_batches_per_epoch:
                print(f"  epoch {epoch+1}/{KAN_EPOCHS}  batch {b+1}/{n_batches_per_epoch}  "
                      f"running_loss={epoch_loss/(b+1):.4f}  elapsed={time.time()-t0:.1f}s")
        kan_hist['loss'].append(epoch_loss)
        print(f" KAN epoch {epoch+1}/{KAN_EPOCHS}  total_loss={epoch_loss:.4f}  time={time.time()-t0:.1f}s")

        torch.save(kan.state_dict(), partial_kan_path)
        torch.save(kan_opt.state_dict(), partial_kan_opt_path)
        json.dump(kan_hist, open(partial_kan_hist_path, "w"))

    torch.save(kan.state_dict(), kan_path)
    for p in [partial_kan_path, partial_kan_opt_path, partial_kan_hist_path]:
        if os.path.exists(p): os.remove(p)

kan.eval()
with torch.no_grad():
    kan_probs_val = torch.softmax(kan(torch.tensor(val_feats, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    kan_probs_test = torch.softmax(kan(torch.tensor(test_feats, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
np.savez(f"{CKPT_DIR}/probs_val/KAN.npz", probs=kan_probs_val)
np.savez(f"{CKPT_DIR}/probs/KAN.npz", probs=kan_probs_test)

print("\nKAN head trained. Validation and test probabilities saved.")
