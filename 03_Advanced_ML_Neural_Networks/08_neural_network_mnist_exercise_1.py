"""
Skript pro výpočet a předpočtení výsledků pro Cvičení 1: Neuronové sítě - Klasifikace číslic MNIST
Ukládá kompletní výsledky do 03_Advanced_ML_Neural_Networks/data/mnist_mlp_exercise_1_precomputed.json
"""

import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "03_Advanced_ML_Neural_Networks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "mnist_mlp_exercise_1_precomputed.json"

    # Načtení dat MNIST (buď z scratch nebo stažení)
    scratch_npz = base_dir / "scratch" / "mnist.npz"
    if not scratch_npz.exists():
        import urllib.request
        print("Downloading mnist.npz...")
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, scratch_npz)

    print(f"Loading MNIST from {scratch_npz}...")
    npz_data = np.load(scratch_npz)
    X_train_raw, y_train_raw = npz_data["x_train"], npz_data["y_train"]
    X_test_raw, y_test_raw = npz_data["x_test"], npz_data["y_test"]

    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    # Normalizace pixelů (0-255 -> 0.0-1.0)
    X_train = torch.tensor(X_train_raw / 255.0, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.long)
    X_test = torch.tensor(X_test_raw / 255.0, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.long)

    # Rozdělení na trénovací a validační sadu (15 % validace)
    val_size = int(len(X_train) * 0.15)
    train_size = len(X_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        TensorDataset(X_train, y_train), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Definice architektury dle zadání Keras: Flatten (784) -> Dense(128, ReLU) -> Dense(10, Softmax)
    class MNISTMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    torch.manual_seed(42)
    model = MNISTMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    print(f"Training MLP for {epochs} epochs...")
    t0 = time.time()

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            correct += (out.argmax(dim=1) == by).sum().item()
            total += len(by)
        tr_loss = total_loss / total
        tr_acc = correct / total

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * len(by)
                v_corr += (out.argmax(dim=1) == by).sum().item()
                v_tot += len(by)
        va_loss = v_loss / v_tot
        va_acc = v_corr / v_tot

        history["epoch"].append(ep)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:2d}/{epochs}: Train Acc={tr_acc*100:.2f}%, Val Acc={va_acc*100:.2f}%, Val Loss={va_loss:.4f}")

    training_duration_s = float(time.time() - t0)

    # Vyhodnocení na testovací sadě
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_preds = test_probs.argmax(axis=1)

    y_test_np = y_test_raw
    test_acc = float((test_preds == y_test_np).mean())
    cm = confusion_matrix(y_test_np, test_preds).tolist()
    report = classification_report(y_test_np, test_preds, output_dict=True)

    # Výběr 40 reprezentativních testovacích vzorků (včetně správných predikcí i typických záměn)
    sample_indices = []
    # Nejprve po 3 vzorcích z každé třídy 0-9
    for digit in range(10):
        digit_idx = np.where((y_test_np == digit) & (test_preds == digit))[0][:3]
        sample_indices.extend(digit_idx.tolist())
    # A 10 chybových predikcí (misclassifications) pro demonstraci diagnostiky
    error_idx = np.where(y_test_np != test_preds)[0][:10]
    sample_indices.extend(error_idx.tolist())

    sample_predictions = []
    for idx in sample_indices:
        sample_predictions.append({
            "test_index": int(idx),
            "true_label": int(y_test_np[idx]),
            "predicted_label": int(test_preds[idx]),
            "is_correct": bool(y_test_np[idx] == test_preds[idx]),
            "probabilities": [round(float(p), 4) for p in test_probs[idx]],
            "image_pixels": (X_test_raw[idx] / 255.0).tolist()
        })

    # Analýza počtu vah
    params_breakdown = {
        "flatten_layer": {"in": "28x28", "out": 784, "weights": 0},
        "hidden_layer_1": {"in": 784, "out": 128, "weights": 784 * 128 + 128}, # 100 480
        "output_layer": {"in": 128, "out": 10, "weights": 128 * 10 + 10},      # 1 290
        "total_trainable_params": 100480 + 1290                                  # 101 770
    }

    # Finální payload
    results = {
        "metadata": {
            "dataset": "MNIST Digits (0-9)",
            "n_train_total": 60000,
            "n_train_split": train_size,
            "n_val_split": val_size,
            "n_test": 10000,
            "batch_size": batch_size,
            "epochs": epochs,
            "training_duration_s": training_duration_s,
            "parameters_breakdown": params_breakdown
        },
        "history": history,
        "metrics": {
            "test_accuracy": test_acc,
            "confusion_matrix": cm,
            "classification_report": report
        },
        "sample_predictions": sample_predictions
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Precomputed results saved to {out_file}!")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Training Duration: {training_duration_s:.2f} s")


if __name__ == "__main__":
    main()
