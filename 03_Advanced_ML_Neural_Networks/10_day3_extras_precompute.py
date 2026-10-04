"""
Předvýpočet dat pro sekce "Den 3: Extras" (Streamlit Cloud nemá TensorFlow ani PyTorch).

Výstupy:
1. data/day3_extras_heart_probs.json
   - Testovací pravděpodobnosti P(ahd=1) z Random Forest a XGBoost (srdce, split 70:30, seed 42)
   - Použití: Cost-Sensitive Threshold kalkulačka
2. data/mnist_mlp_weights.npz
   - Váhy MLP 784 -> 128 (ReLU) -> 10 (Softmax) natrénované v PyTorch (lokálně)
   - Inference v aplikaci probíhá čistě v numpy (≈ 0.1 ms na obrázek)
   - Trénink s lehkou augmentací (náhodné posuny ±2 px) => vyšší robustnost na ručně kreslené číslice
"""

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "03_Advanced_ML_Neural_Networks" / "data"


def precompute_heart_probabilities():
    data_path = BASE_DIR / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "heart_data_normalized.csv"
    df = pd.read_csv(data_path)
    X = df.drop("ahd_yes", axis=1)
    y = df["ahd_yes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Nejlepší parametry z GridSearchCV (XGBoost - Cvičení 1)
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic", gamma=1.0, learning_rate=0.2, max_depth=5,
        n_estimators=50, random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    xgb_clf.fit(X_train, y_train)

    payload = {
        "description": "Testovací pravděpodobnosti P(ahd=1) – heart_data_normalized.csv, split 70:30, random_state=42",
        "n_test": int(len(y_test)),
        "y_test": [int(v) for v in y_test.values],
        "proba_rf": [round(float(p), 5) for p in rf.predict_proba(X_test)[:, 1]],
        "proba_xgb": [round(float(p), 5) for p in xgb_clf.predict_proba(X_test)[:, 1]],
    }
    out = OUT_DIR / "day3_extras_heart_probs.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[heart] Saved {out} (n_test={payload['n_test']})")


def precompute_mnist_weights():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    npz_path = BASE_DIR / "scratch" / "mnist.npz"
    if not npz_path.exists():
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", npz_path)

    d = np.load(npz_path)
    X_train = torch.tensor(d["x_train"] / 255.0, dtype=torch.float32)
    y_train = torch.tensor(d["y_train"], dtype=torch.long)
    X_test = torch.tensor(d["x_test"] / 255.0, dtype=torch.float32)
    y_test = d["y_test"]

    torch.manual_seed(42)
    fc1 = nn.Linear(784, 128)
    fc2 = nn.Linear(128, 10)
    model = nn.Sequential(nn.Flatten(), fc1, nn.ReLU(), fc2)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_train)
    batch = 256
    epochs = 20
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            bx = X_train[idx]
            # Lehká augmentace: náhodný posun celé dávky o ±2 px (okraje MNIST jsou černé)
            dx, dy = np.random.randint(-2, 3, size=2)
            bx = torch.roll(bx, shifts=(int(dy), int(dx)), dims=(1, 2))
            opt.zero_grad()
            loss = loss_fn(model(bx), y_train[idx])
            loss.backward()
            opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_test).argmax(1).numpy() == y_test).mean()
            print(f"[mnist] epoch {ep:2d}: test acc = {acc * 100:.2f} %")

    model.eval()
    with torch.no_grad():
        test_acc = float((model(X_test).argmax(1).numpy() == y_test).mean())

    out = OUT_DIR / "mnist_mlp_weights.npz"
    np.savez_compressed(
        out,
        W1=fc1.weight.detach().numpy().T.astype(np.float32),  # (784, 128)
        b1=fc1.bias.detach().numpy().astype(np.float32),      # (128,)
        W2=fc2.weight.detach().numpy().T.astype(np.float32),  # (128, 10)
        b2=fc2.bias.detach().numpy().astype(np.float32),      # (10,)
        test_accuracy=np.array([test_acc], dtype=np.float32),
    )
    print(f"[mnist] Saved {out} | test acc {test_acc * 100:.2f} % | {time.time() - t0:.1f} s")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precompute_heart_probabilities()
    precompute_mnist_weights()
