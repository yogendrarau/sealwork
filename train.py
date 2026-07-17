"""
Stage 2: Train the hand-seal MLP on data/landmarks.csv.

Model: 126 -> 256 -> 128 -> n_classes, ReLU + Dropout(0.3). Kept identical to the
MLP in web/export_onnx.py so the saved checkpoint round-trips through export
untouched (same module layout => same state_dict keys).

CLASSES — 12 seals, plus an optional 13th "none" class:
  "none" = transition frames (hands in motion between seals, half-formed shapes),
  collected via collect.py's none mode. The model learns transit-shapes -> none,
  so the web app's confidence gate rejects them instead of misfiring a wrong sign
  mid-combo. "none" is ALWAYS last; the 12 seal indices never shift. If the CSV
  has no none rows yet, a 12-class model is trained (identical to before) — the
  web app adapts off signs.json either way.

EVALUATION — two honest, different numbers:

  * default (temporal split): for each class, the FIRST 80% of its frames (in
    collection order) train, the LAST 20% test. This is deliberate. Hold-to-
    capture produces bursts of near-identical frames; a random split leaks
    almost-duplicate frames across train/test and inflates accuracy. A temporal
    split is the honest "how well did it actually learn this sign" number.

  * --lopo (leave-one-person-out): train on every subject but one, test on the
    held-out subject. This is the honest "does it work on a STRANGER's hands"
    number. Needs >= 2 subjects in the CSV (subject column). With one subject
    it can't run and says so.

Outputs (consumed by export_onnx.py):
    models/mlp.pt      {"state_dict":..., "signs":[...]}
    models/signs.json  the label order actually trained

Usage:
    python train.py                 # temporal split, save model
    python train.py --lopo          # + leave-one-person-out report
    python train.py --epochs 400 --device cuda
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

# Fixed label order — NEVER reorder. Must match signs.json / the web app.
BASE_SIGNS = ["rat", "ox", "tiger", "hare", "dragon", "snake",
              "horse", "ram", "monkey", "boar", "dog", "bird"]
NONE_LABEL = "none"          # transition class; always appended LAST if present

N_FEATURES = 126
MODEL_DIR = "models"
CSV_PATH = os.path.join("data", "landmarks.csv")


class MLP(nn.Module):
    # identical to web/export_onnx.py — do not diverge
    def __init__(self, in_dim=N_FEATURES, n_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_csv(path):
    """Return (X[N,126] float32, labels[N] str, subjects[N] str) in file (temporal) order."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[N_FEATURES - 1] == "R20_z", f"bad feature layout: {header[N_FEATURES-1]}"
        i_lb = header.index("label")
        i_sub = header.index("subject") if "subject" in header else None
        X, labels, subs = [], [], []
        for row in reader:
            if not any(row):
                continue
            X.append([float(row[c]) for c in range(N_FEATURES)])
            labels.append(row[i_lb])
            subs.append(row[i_sub] if i_sub is not None else "unknown")
    return np.asarray(X, dtype=np.float32), labels, subs


def temporal_split(y, n_classes, frac_train=0.8):
    """Per-label, first frac_train (in order) -> train, rest -> test. Returns index arrays."""
    tr, te = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]          # already in temporal order
        k = int(round(len(idx) * frac_train))
        tr.extend(idx[:k].tolist())
        te.extend(idx[k:].tolist())
    return np.array(sorted(tr)), np.array(sorted(te))


def train_model(Xtr, ytr, n_classes, epochs, lr, batch, device, seed=0, quiet=False,
                class_weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    n = len(Xtr)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            bi = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[bi]), ytr_t[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        if not quiet and (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep+1:4d}/{epochs}  train loss {tot/n:.4f}")
    return model


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(1).cpu().numpy()


def report(y_true, y_pred, signs, title):
    acc = (y_true == y_pred).mean()
    print(f"\n{title}: overall accuracy {acc*100:.2f}%  (n={len(y_true)})")
    print("  per-class:")
    for c, s in enumerate(signs):
        m = y_true == c
        if m.sum() == 0:
            print(f"    {s:7} —  (no test samples)")
            continue
        a = (y_pred[m] == c).mean()
        print(f"    {s:7} {a*100:5.1f}%  ({m.sum()})")
    idx = {s: i for i, s in enumerate(signs)}
    # known confusions — surface them explicitly. For a real sign eaten by the
    # none class, that's a sign the motion gate let settled frames through.
    pairs = [("tiger", "ram"), ("ram", "tiger"), ("dog", "boar"), ("boar", "dog")]
    if NONE_LABEL in idx:
        pairs += [(s, NONE_LABEL) for s in BASE_SIGNS]
    for a, b in pairs:
        if a not in idx or b not in idx:
            continue
        m = y_true == idx[a]
        if m.sum():
            conf = (y_pred[m] == idx[b]).mean()
            if conf > 0.02:
                print(f"  confusion: {a:6} predicted as {b:6} {conf*100:4.1f}% of the time")
    return acc


def run_lopo(X, y, subs, signs, args, device, class_weights=None):
    uniq = sorted(set(subs))
    print("\n" + "=" * 60)
    print("LEAVE-ONE-PERSON-OUT (honest 'works on strangers' number)")
    if len(uniq) < 2:
        print(f"  Only {len(uniq)} subject ({uniq}). LOPO needs >= 2 people.")
        print("  Subject tagging is wired up (collect.py) — collect another")
        print("  person's samples, then rerun with --lopo.")
        return
    subs = np.asarray(subs)
    accs = []
    for held in uniq:
        te = subs == held
        tr = ~te
        model = train_model(X[tr], y[tr], len(signs), args.epochs, args.lr,
                            args.batch, device, seed=args.seed, quiet=True,
                            class_weights=class_weights)
        a = report(y[te], predict(model, X[te], device), signs, f"held-out = {held}")
        accs.append(a)
    print(f"\nLOPO mean accuracy across {len(uniq)} people: {np.mean(accs)*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lopo", action="store_true", help="also run leave-one-person-out eval")
    ap.add_argument("--no-save", action="store_true", help="skip writing models/")
    ap.add_argument("--none-weight", type=float, default=0.5,
                    help="CE loss weight for the none class (<1 stops none from "
                         "claiming contested space and eating real signs; a missed "
                         "sign hurts gameplay more than a transit slipping through)")
    args = ap.parse_args()

    device = torch.device(args.device)
    X, labels, subs = load_csv(CSV_PATH)

    # 12-class or 13-class depending on whether none data exists yet
    has_none = NONE_LABEL in labels
    signs = BASE_SIGNS + ([NONE_LABEL] if has_none else [])
    idx = {s: i for i, s in enumerate(signs)}
    unknown = sorted({l for l in labels if l not in idx})
    if unknown:
        raise ValueError(f"unknown labels in CSV: {unknown}")
    y = np.asarray([idx[l] for l in labels], dtype=np.int64)

    print(f"loaded {len(X)} samples, {len(set(subs))} subject(s): {sorted(set(subs))}")
    n_none = labels.count(NONE_LABEL)
    if has_none:
        print(f"training {len(signs)}-class model (includes '{NONE_LABEL}': {n_none} samples)")
        if n_none < 300:
            print(f"  WARNING: only {n_none} none samples — aim for 500+ for a useful transition class")
    else:
        print(f"no '{NONE_LABEL}' samples — training the classic 12-class model.")
        print("  (collect transitions with collect.py's none mode to fix mid-combo misfires)")
    print(f"device: {device}")

    # inverse-frequency class weights: stops big/diverse classes (especially
    # none) from claiming boundary territory of small ones (monkey). The none
    # class is additionally discounted — a real sign eaten by none is a missed
    # cast, which hurts gameplay more than a transit frame slipping through.
    counts = np.bincount(y, minlength=len(signs)).astype(np.float64)
    cw = len(y) / (len(signs) * np.maximum(counts, 1.0))
    if has_none:
        cw[-1] *= args.none_weight
    cw = cw.tolist()
    print("class weights: inverse-frequency" + (f", none x{args.none_weight}" if has_none else ""))

    # ---- default: temporal split, train, evaluate, save ----
    tr, te = temporal_split(y, len(signs))
    print(f"\ntemporal split: {len(tr)} train / {len(te)} test")
    model = train_model(X[tr], y[tr], len(signs), args.epochs, args.lr,
                        args.batch, device, seed=args.seed, class_weights=cw)
    report(y[te], predict(model, X[te], device), signs, "TEMPORAL SPLIT (within-person)")

    if not args.no_save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        # retrain on ALL data for the shipped model (more data = better deploy),
        # but the number you trust is the temporal-split one above.
        print(f"\ntraining deploy model on all {len(X)} samples "
              f"({args.epochs} epochs, quiet — takes a couple of minutes)…")
        full = train_model(X, y, len(signs), args.epochs, args.lr, args.batch,
                           device, seed=args.seed, quiet=True, class_weights=cw)
        torch.save({"state_dict": full.state_dict(), "signs": signs},
                   os.path.join(MODEL_DIR, "mlp.pt"))
        with open(os.path.join(MODEL_DIR, "signs.json"), "w") as f:
            json.dump(signs, f)
        print(f"\nsaved -> {MODEL_DIR}/mlp.pt  ({len(signs)} classes, all {len(X)} samples)")
        print(f"saved -> {MODEL_DIR}/signs.json")
        print("next: python web/export_onnx.py   (from repo root; writes models/mlp.onnx + parity.json)")
        print("then copy models/mlp.onnx models/signs.json models/parity.json -> web/")

    if args.lopo:
        run_lopo(X, y, subs, signs, args, device, class_weights=cw)


if __name__ == "__main__":
    main()
