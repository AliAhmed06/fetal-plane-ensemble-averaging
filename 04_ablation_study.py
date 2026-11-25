import json, os, sys, math, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, log_loss)
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

CONFIG = dict(
    META_CSV="/kaggle/input/fetal-planes-db/FETAL_PLANES_DB_data.csv",
    PRED_DIR="/kaggle/working/natural_preds",
    OUT_DIR="/kaggle/working/revision_outputs",
    TOP5=["tf_efficientnet_b4_ns", "tf_efficientnet_b5_ns", "densenet169",
          "resnet50", "efficientnet_b0"],
    ALL9=["densenet169", "resnet50", "efficientnet_b0", "mobilenet_v2",
          "convnext_tiny", "resnext50_32x4d", "tf_efficientnet_b4_ns",
          "tf_efficientnet_b5_ns", "inception_resnet_v2"],
    KAN_NAME="KAN",
    N_BOOT=2000,
    PSO_SEEDS=10,
    SEED=0,
)
CLASSES = ["abdomen", "brain", "cervix", "femur", "thorax", "other"]
PLANE_MAP = {"Fetal abdomen": "abdomen", "Fetal brain": "brain",
             "Maternal cervix": "cervix", "Fetal femur": "femur",
             "Fetal thorax": "thorax", "Other": "other"}
rng0 = np.random.default_rng(CONFIG["SEED"])
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
REPORT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    REPORT.append(s)


def load_meta(path):
    m = pd.read_csv(path, sep=None, engine="python")
    m.columns = [c.strip() for c in m.columns]
    m["image_name"] = m["Image_name"].astype(str).str.strip()
    m["label"] = m["Plane"].map(PLANE_MAP)
    m["patient"] = m["Patient_num"].astype(str)
    return m


def norm_name(s):
    s = os.path.basename(str(s)).strip()
    return os.path.splitext(s)[0]


def natural_split_table(meta, train_p, val_p, test_p):
    out = []
    for name, ps in [("train", train_p), ("val", val_p), ("test", test_p)]:
        d = meta[meta.patient.isin(ps)]
        r = d.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
        r.update(split=name, patients=len(set(ps)), images=len(d))
        out.append(r)
    return pd.DataFrame(out)[["split", "patients", "images"] + CLASSES]


def metrics(y, p):
    yhat = p.argmax(1)
    out = dict(acc=accuracy_score(y, yhat),
               macro_f1=f1_score(y, yhat, average="macro"),
               weighted_f1=f1_score(y, yhat, average="weighted"),
               mcc=matthews_corrcoef(y, yhat))
    try:
        out["auc_macro_ovr"] = roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError:
        out["auc_macro_ovr"] = np.nan
    pc = np.clip(p, 1e-12, 1)
    out["nll"] = log_loss(y, pc / pc.sum(1, keepdims=True), labels=list(range(len(CLASSES))))
    out["ece"] = ece(y, p)
    return out


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def patient_bootstrap(y, preds: dict, patients, ref=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.asarray(patients)
    up = np.unique(pats)
    idx_of = {u: np.where(pats == u)[0] for u in up}
    res = {k: {"acc": [], "macro_f1": [], "mcc": []} for k in preds}
    diff = {k: {"acc": [], "macro_f1": []} for k in preds} if ref else None
    for _ in range(n):
        s = rng.choice(up, size=len(up), replace=True)
        ii = np.concatenate([idx_of[u] for u in s])
        yy = y[ii]
        cache = {}
        for k, p in preds.items():
            yh = p[ii].argmax(1)
            a, f = accuracy_score(yy, yh), f1_score(yy, yh, average="macro")
            cache[k] = (a, f)
            res[k]["acc"].append(a); res[k]["macro_f1"].append(f)
            res[k]["mcc"].append(matthews_corrcoef(yy, yh))
        if ref:
            for k in preds:
                diff[k]["acc"].append(cache[k][0] - cache[ref][0])
                diff[k]["macro_f1"].append(cache[k][1] - cache[ref][1])
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    out = {}
    for k in preds:
        out[k] = {f"{m}_ci": ci(res[k][m]) for m in res[k]}
        if ref:
            out[k].update({f"d_{m}_vs_{ref}_ci": ci(diff[k][m]) for m in diff[k]})
    return out


def mcnemar(y, pa, pb):
    a, b = pa.argmax(1) == y, pb.argmax(1) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def fuse(P, w):
    w = np.clip(np.asarray(w, float), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return np.tensordot(w, P, axes=1)


def wf1(y, p):
    return f1_score(y, p.argmax(1), average="weighted")


def pso_weights(Pv, yv, seed, n_particles=40, iters=30, w_in=0.9, c1=0.65, c2=0.95,
                lo=0.0, hi=10.0):
    rng = np.random.default_rng(seed)
    d = Pv.shape[0]
    x = rng.uniform(lo, hi, (n_particles, d)); v = np.zeros_like(x)
    f = np.array([wf1(yv, fuse(Pv, xi)) for xi in x])
    pb, pbf = x.copy(), f.copy(); g = pb[pbf.argmax()].copy(); gf = pbf.max()
    for _ in range(iters):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w_in * v + c1 * r1 * (pb - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
        f = np.array([wf1(yv, fuse(Pv, xi)) for xi in x])
        m = f > pbf; pb[m], pbf[m] = x[m], f[m]
        if pbf.max() > gf:
            g, gf = pb[pbf.argmax()].copy(), pbf.max()
    return g / g.sum(), gf


def dirichlet_search(Pv, yv, seed, n=1240):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(Pv.shape[0]), n)
    s = [wf1(yv, fuse(Pv, w)) for w in W]
    return W[int(np.argmax(s))], max(s)


def fit_temperature(logp_val, yv):
    best = (1.0, 1e9)
    for T in np.linspace(0.3, 5, 471):
        q = softmax(logp_val / T)
        nll = log_loss(yv, q, labels=list(range(q.shape[1])))
        if nll < best[1]:
            best = (T, nll)
    return best[0]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def power_scale(p, alpha):
    q = np.clip(p, 1e-12, 1) ** alpha
    return q / q.sum(1, keepdims=True)


def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def make_heads(in_dim, n_cls=6, hidden=128):
    import torch, torch.nn as nn, torch.nn.functional as F

    class KANLinear(nn.Module):
        def __init__(self, i, o, grid=5, k=3, rng_=(-1, 1)):
            super().__init__()
            self.i, self.o, self.g, self.k = i, o, grid, k
            h = (rng_[1] - rng_[0]) / grid
            gr = torch.arange(-k, grid + k + 1) * h + rng_[0]
            self.register_buffer("grid", gr.expand(i, -1).contiguous())
            self.base = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
            self.spline = nn.Parameter(torch.randn(o, i, grid + k) * 0.1 / math.sqrt(i))
            self.scaler = nn.Parameter(torch.ones(o, i))

        def bsplines(self, x):
            g = self.grid; x = x.unsqueeze(-1)
            b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
            for p in range(1, self.k + 1):
                b = ((x - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * b[..., :-1]
                     + (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p]) * b[..., 1:])
            return b

        def forward(self, x):
            base = F.linear(F.silu(x), self.base)
            sp = F.linear(self.bsplines(x).flatten(1),
                          (self.spline * self.scaler.unsqueeze(-1)).flatten(1))
            return base + sp

    kan = nn.Sequential(nn.LayerNorm(in_dim), KANLinear(in_dim, hidden), KANLinear(hidden, n_cls))
    kan_paper = nn.Sequential(KANLinear(in_dim, hidden), KANLinear(hidden, n_cls))
    mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(),
                        nn.Linear(hidden, n_cls))
    kp = sum(p.numel() for p in kan.parameters())
    wide = max(hidden, int((kp - n_cls) / (in_dim + 1 + n_cls)))
    mlp_wide = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, wide), nn.SiLU(),
                             nn.Linear(wide, n_cls))
    return {"KAN(2048-128-6) no input norm": kan_paper, "KAN(2048-128-6)": kan, "MLP(2048-128-6)": mlp, f"MLP-param-matched(w={wide})": mlp_wide}


def train_head(model, Xtr, ytr, Xva, yva, epochs=6, lr=1e-3, wd=1e-4, bs=128, seed=0,
               select_on_val=False, max_epochs=60, patience=10):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(dev)
    cw = torch.tensor(len(ytr) / (len(CLASSES) * np.bincount(ytr, minlength=len(CLASSES))),
                      dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    best, best_state, bad = -1, None, 0
    n_ep = max_epochs if select_on_val else epochs
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad(); lossf(model(Xt[j].to(dev)), yt[j].to(dev)).backward(); opt.step()
        if select_on_val:
            model.eval()
            with torch.no_grad():
                f = wf1(yva, model(Xv).softmax(1).cpu().numpy())
            if f > best:
                best, bad = f, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_head(model, X):
    import torch
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=dev)).softmax(1).cpu().numpy()


def main():
    C = CONFIG
    meta = load_meta(C["META_CSV"]) if os.path.exists(C["META_CSV"]) else None

    pdir = C["PRED_DIR"]
    if not os.path.exists(f"{pdir}/test_index.csv"):
        say("No natural predictions found.")
        return finish()

    vi, ti = pd.read_csv(f"{pdir}/val_index.csv"), pd.read_csv(f"{pdir}/test_index.csv")
    lab = {c: i for i, c in enumerate(CLASSES)}
    yv = vi.label.map(lab).values if vi.label.dtype == object else vi.label.values
    yt = ti.label.map(lab).values if ti.label.dtype == object else ti.label.values
    if meta is not None:
        pm = dict(zip(meta.image_name.map(norm_name), meta.patient))
        pat_t = ti.image_name.map(lambda s: pm.get(norm_name(s), s)).values
    else:
        pat_t = ti.get("patient", ti.image_name).values
    assert ti.image_name.map(norm_name).is_unique, "test index has duplicate images, must be natural"
    say(f"\nNatural val: {len(yv)} images; natural test: {len(yt)} images, "
        f"{len(np.unique(pat_t))} patients")
    say("Natural test class counts:", dict(zip(CLASSES, np.bincount(yt, minlength=6))))

    names = [m for m in C["ALL9"] + [C["KAN_NAME"]] if os.path.exists(f"{pdir}/{m}_test.npy")]
    PV = {m: np.load(f"{pdir}/{m}_val.npy") for m in names}
    PT = {m: np.load(f"{pdir}/{m}_test.npy") for m in names}

    rows = {}
    for m in names:
        rows[f"single: {m}"] = (PV[m], PT[m])
    heads_test = {}
    if all(os.path.exists(f"{pdir}/b5emb_{s}.npy") for s in ["train", "val", "test"]):
        say("\n" + "=" * 78 + "\nPART D  KAN vs MLP vs LINEAR on frozen B5 embeddings\n" + "=" * 78)
        tri = pd.read_csv(f"{pdir}/train_index.csv")
        ytr = tri.label.map(lab).values if tri.label.dtype == object else tri.label.values
        assert tri.image_name.map(norm_name).is_unique, "train embeddings must be un-oversampled"
        Xtr, Xv, Xt = (np.load(f"{pdir}/b5emb_{s}.npy") for s in ["train", "val", "test"])
        say(f"Embedding values inside KAN spline support [-2.2, 2.2): {((Xtr >= -2.2) & (Xtr < 2.2)).mean():.3f}")
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
        heads = {"Linear (logistic regression)": (lr.predict_proba(Xv), lr.predict_proba(Xt), 6 * Xtr.shape[1] + 6)}
        for sched, kw in [("6ep", dict(epochs=6)), ("early-stop", dict(select_on_val=True))]:
            for hname, mdl in make_heads(Xtr.shape[1]).items():
                outs_v, outs_t = [], []
                for s in range(3):
                    mdl_s = make_heads(Xtr.shape[1])[hname]
                    mdl_s = train_head(mdl_s, Xtr, ytr, Xv, yv, seed=s, **kw)
                    outs_v.append(predict_head(mdl_s, Xv)); outs_t.append(predict_head(mdl_s, Xt))
                npar = sum(p.numel() for p in mdl.parameters())
                key = f"{hname} [{sched}]"
                heads[key] = (np.mean(outs_v, 0), np.mean(outs_t, 0), npar)
                f_seeds = [f1_score(yt, o.argmax(1), average="macro") for o in outs_t]
                say(f"{key:45s} params={npar:>9,d}  test macro-F1 per seed="
                    + ", ".join(f"{f:.4f}" for f in f_seeds))
        hd = []
        for k, (pv, pt, npar) in heads.items():
            hd.append(dict(head=k, params=npar, **{f"val_{a}": b for a, b in metrics(yv, pv).items()},
                           **{f"test_{a}": b for a, b in metrics(yt, pt).items()}))
            heads_test[k] = (pv, pt)
        pd.DataFrame(hd).to_csv(f"{C['OUT_DIR']}/D_heads_on_B5.csv", index=False)
        say(pd.DataFrame(hd)[["head", "params", "val_macro_f1", "test_macro_f1",
                              "test_acc", "test_nll"]].round(4).to_string(index=False))
        b5 = "tf_efficientnet_b5_ns"
        if b5 in PT and C["KAN_NAME"] in PT:
            e1, e2 = PT[b5].argmax(1) != yt, PT[C["KAN_NAME"]].argmax(1) != yt
            say(f"Error overlap B5 vs KAN: B5 errors={e1.sum()}, KAN errors={e2.sum()}, "
                f"both={int((e1 & e2).sum())}, prediction agreement="
                f"{(PT[b5].argmax(1) == PT[C['KAN_NAME']].argmax(1)).mean():.4f}")

    say("\n" + "=" * 78 + "\nPART C  FUSION ABLATIONS (weights fit on natural VAL only)\n" + "=" * 78)
    top5 = [m for m in C["TOP5"] if m in PV]
    best_single = max(top5, key=lambda m: wf1(yv, PV[m]))
    say("Best single model by val weighted-F1:", best_single)

    def add(name, members, how, seeds=1):
        Pv = np.stack([PV.get(m, heads_test.get(m, (None, None))[0]) for m in members])
        Pt = np.stack([PT.get(m, heads_test.get(m, (None, None))[1]) for m in members])
        if how == "equal":
            w = np.ones(len(members)) / len(members)
        elif how == "valf1":
            w = np.array([wf1(yv, PV.get(m, heads_test.get(m, (None,))[0])) for m in members]); w /= w.sum()
        elif how == "pso":
            ws = [pso_weights(Pv, yv, seed=s) for s in range(seeds)]
            w = ws[0][0]
            W = np.array([x[0] for x in ws])
            tf = [f1_score(yt, fuse(Pt, wi).argmax(1), average="macro") for wi in W]
            pd.DataFrame(W, columns=members).assign(val_wf1=[x[1] for x in ws], test_macroF1=tf) \
                .to_csv(f"{C['OUT_DIR']}/C_pso_seed_stability_{name.replace(' ', '_')}.csv", index=False)
            say(f"PSO stability over {seeds} seeds [{name}]: weight SD per expert = "
                + ", ".join(f"{m}:{s:.3f}" for m, s in zip(members, W.std(0)))
                + f" | test macro-F1 range {min(tf):.4f}-{max(tf):.4f}")
        elif how == "dirichlet":
            w, _ = dirichlet_search(Pv, yv, seed=0)
        elif how == "stack":
            lr = LogisticRegression(max_iter=3000, C=1.0)
            Xv_ = np.log(np.clip(Pv, 1e-9, 1)).transpose(1, 0, 2).reshape(len(yv), -1)
            Xt_ = np.log(np.clip(Pt, 1e-9, 1)).transpose(1, 0, 2).reshape(len(yt), -1)
            from sklearn.model_selection import cross_val_predict
            pv_cv = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0), Xv_, yv, cv=5,
                                      method="predict_proba")
            lr.fit(Xv_, yv)
            rows[name] = (pv_cv, lr.predict_proba(Xt_)); return
        rows[name] = (fuse(Pv, w), fuse(Pt, w))
        say(f"{name:55s} weights=" + ", ".join(f"{m}:{x:.3f}" for m, x in zip(members, w)))

    K = C["KAN_NAME"]
    add("top5 equal-weight", top5, "equal")
    add("top5 val-F1-proportional", top5, "valf1")
    add("top5 PSO", top5, "pso", seeds=C["PSO_SEEDS"])
    add("top5 random-simplex search (equal budget)", top5, "dirichlet")
    add("top5 logistic stacking", top5, "stack")
    if K in PV:
        add("top5+KAN equal-weight", top5 + [K], "equal")
        add("top5+KAN PSO", top5 + [K], "pso", seeds=C["PSO_SEEDS"])
        add("top5+KAN random-simplex search", top5 + [K], "dirichlet")
    for hk in heads_test:
        if hk.startswith("MLP(2048-128-6) [6ep]") or hk.startswith("Linear"):
            PV[hk], PT[hk] = heads_test[hk]
            add(f"top5+{hk} PSO", top5 + [hk], "pso", seeds=1)
    all9 = [m for m in C["ALL9"] if m in PV]
    add("all9 equal-weight", all9, "equal")

    say("\n" + "=" * 78 + "\nPART B  POWER SCALING p^(1/1.3)\n" + "=" * 78)
    full_key = "top5+KAN PSO" if K in PV else "top5 PSO"
    pv_full, pt_full = rows[full_key]
    q = power_scale(pt_full, 1 / 1.3)
    say(f"argmax identical after scaling: {(q.argmax(1) == pt_full.argmax(1)).all()}")
    say(f"mean entropy before={entropy(pt_full).mean():.4f}  after={entropy(q).mean():.4f} "
        f"(alpha<1 => flatter)")
    T = fit_temperature(np.log(np.clip(pv_full, 1e-12, 1)), yv)
    qT = softmax(np.log(np.clip(pt_full, 1e-12, 1)) / T)
    for lbl, pp in [("no scaling", pt_full), ("alpha=1/1.3 (as in manuscript)", q),
                    (f"temperature fit on val (T={T:.2f})", qT)]:
        m = metrics(yt, pp)
        say(f"{lbl:38s} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} "
            f"AUC={m['auc_macro_ovr']:.4f} NLL={m['nll']:.4f} ECE={m['ece']:.4f}")

    say("\n" + "=" * 78 + "\nPART E  RESULTS TABLE with patient-level bootstrap 95% CIs\n" + "=" * 78)
    ref = f"single: {best_single}"
    preds_t = {k: v[1] for k, v in rows.items()}
    boot = patient_bootstrap(yt, preds_t, pat_t, ref=ref, n=C["N_BOOT"], seed=C["SEED"])
    tab = []
    for k, (pv, pt) in rows.items():
        mv, mt = metrics(yv, pv), metrics(yt, pt)
        b = boot[k]
        n01, n10, pmc = mcnemar(yt, pt, preds_t[ref])
        tab.append({"config": k, "val_macro_f1": mv["macro_f1"], "test_acc": mt["acc"],
                    "acc_95ci": "%.4f-%.4f" % b["acc_ci"], "test_macro_f1": mt["macro_f1"],
                    "macroF1_95ci": "%.4f-%.4f" % b["macro_f1_ci"], "test_mcc": mt["mcc"],
                    "test_auc": mt["auc_macro_ovr"],
                    f"dMacroF1_vs_best_single_95ci": "%.4f-%.4f" % b[f"d_macro_f1_vs_{ref}_ci"],
                    "mcnemar_p_vs_best_single": pmc})
    tab = pd.DataFrame(tab)
    tab.to_csv(f"{C['OUT_DIR']}/E_main_ablation_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 50):
        say(tab.round(4).to_string(index=False))
    say("\nAn improvement is claimed only where the paired "
        "CI of the difference excludes 0.")
    return finish()


def finish():
    with open(f"{CONFIG['OUT_DIR']}/REPORT.txt", "w") as f:
        f.write("\n".join(REPORT))
    print(f"\nAll outputs in {CONFIG['OUT_DIR']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG.update(json.load(open(sys.argv[1])))
        os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    main()
