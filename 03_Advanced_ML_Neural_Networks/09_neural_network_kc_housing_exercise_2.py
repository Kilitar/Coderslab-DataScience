"""
Skript pro výpočet a předpočtení výsledků pro Cvičení 2: Neuronové sítě - Regrese cen nemovitostí King County
Ukládá kompletní výsledky do 03_Advanced_ML_Neural_Networks/data/kc_house_mlp_exercise_2_precomputed.json
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "kc_house_data_preprocessed.csv"
    data_dir = base_dir / "03_Advanced_ML_Neural_Networks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "kc_house_mlp_exercise_2_precomputed.json"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop("price", axis=1)
    y = df["price"]
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"Dataset shape: {df.shape}, Features ({n_features}): {feature_names}")

    # Split 80:20 (nebo 70:30, použijeme 80:20, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Normalizace proměnných pomocí StandardScaler
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Trénovací a validační rozdělení (15 % validace)
    t_X = torch.tensor(X_train_scaled, dtype=torch.float32)
    t_y = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    full_dataset = TensorDataset(t_X, t_y)

    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 1. Trénování baselines (OLS, Strom, Random Forest)
    print("Fitting baseline models (Linear Regression, Decision Tree, Random Forest)...")
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_mae = float(mean_absolute_error(y_test, y_pred_lr))
    lr_rmse = float(mean_squared_error(y_test, y_pred_lr) ** 0.5)
    lr_r2 = float(r2_score(y_test, y_pred_lr))

    dt = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_mae = float(mean_absolute_error(y_test, y_pred_dt))
    dt_rmse = float(mean_squared_error(y_test, y_pred_dt) ** 0.5)
    dt_r2 = float(r2_score(y_test, y_pred_dt))

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(mean_squared_error(y_test, y_pred_rf) ** 0.5)
    rf_r2 = float(r2_score(y_test, y_pred_rf))

    # 2. Experimenty s architekturou MLP (omezení: počet neuronů v každé skryté vrstvě <= 18)
    architectures = {
        "exp1_1layer": {
            "name": "Experiment 1: 1 skrytá vrstva (18 neuronů)",
            "layers_desc": "Dense(18, ReLU) -> Dense(1, Linear)",
            "model": nn.Sequential(
                nn.Linear(n_features, 18),
                nn.ReLU(),
                nn.Linear(18, 1)
            ),
            "lr": 0.01
        },
        "exp2_2layers": {
            "name": "Experiment 2: 2 skryté vrstvy (18 -> 10 neuronů)",
            "layers_desc": "Dense(18, ReLU) -> Dense(10, ReLU) -> Dense(1, Linear)",
            "model": nn.Sequential(
                nn.Linear(n_features, 18),
                nn.ReLU(),
                nn.Linear(18, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            ),
            "lr": 0.008
        },
        "exp3_3layers": {
            "name": "Experiment 3: 3 skryté vrstvy (18 -> 12 -> 6 neuronů) [VÍTĚZ]",
            "layers_desc": "Dense(18, ReLU) -> Dense(12, ReLU) -> Dense(6, ReLU) -> Dense(1, Linear)",
            "model": nn.Sequential(
                nn.Linear(n_features, 18),
                nn.ReLU(),
                nn.Linear(18, 12),
                nn.ReLU(),
                nn.Linear(12, 6),
                nn.ReLU(),
                nn.Linear(6, 1)
            ),
            "lr": 0.005
        },
        "exp4_narrow": {
            "name": "Experiment 4: Příliš úzká síť / Bottleneck (4 neurony)",
            "layers_desc": "Dense(4, ReLU) -> Dense(1, Linear)",
            "model": nn.Sequential(
                nn.Linear(n_features, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            ),
            "lr": 0.01
        }
    }

    epochs = 40
    exp_results = {}
    best_pred_orig = None

    for exp_id, exp_info in architectures.items():
        print(f"\nTraining {exp_info['name']} for {epochs} epochs...")
        torch.manual_seed(42)
        model = exp_info["model"]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=exp_info["lr"])

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }

        t_exp0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            tr_loss_total, tr_cnt = 0.0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                tr_loss_total += loss.item() * len(by)
                tr_cnt += len(by)
            tr_loss = tr_loss_total / tr_cnt

            model.eval()
            v_loss_total, v_cnt = 0.0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = model(bx)
                    loss = criterion(out, by)
                    v_loss_total += loss.item() * len(by)
                    v_cnt += len(by)
            v_loss = v_loss_total / v_cnt

            history["epoch"].append(ep)
            history["train_loss"].append(round(tr_loss, 4))
            history["val_loss"].append(round(v_loss, 4))

        dur = float(time.time() - t_exp0)

        # Predikce na testovací sadě
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
            pred_orig = scaler_y.inverse_transform(pred_scaled).flatten()

        if exp_id == "exp3_3layers":
            best_pred_orig = pred_orig

        mae = float(mean_absolute_error(y_test, pred_orig))
        rmse = float(mean_squared_error(y_test, pred_orig) ** 0.5)
        r2 = float(r2_score(y_test, pred_orig))

        # Počet parametrů
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Results for {exp_id}: MAE={mae:,.2f} USD, RMSE={rmse:,.2f} USD, R2={r2*100:.2f}%, Params={n_params}")

        exp_results[exp_id] = {
            "name": exp_info["name"],
            "layers_desc": exp_info["layers_desc"],
            "n_params": n_params,
            "duration_s": round(dur, 2),
            "test_mae": round(mae, 2),
            "test_rmse": round(rmse, 2),
            "test_r2": round(r2, 4),
            "history": history
        }

    # 3. Analýza reziduí a vzorků (nejlepší model exp3)
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=300, replace=False)
    y_test_np = y_test.values
    sample_residuals = []
    for idx in sample_indices:
        actual = float(y_test_np[idx])
        pred_mlp = float(best_pred_orig[idx])
        sample_residuals.append({
            "actual": actual,
            "pred_mlp": pred_mlp,
            "res_mlp": actual - pred_mlp,
            "sqft_living": float(X_test.iloc[idx]["sqft_living"]),
            "grade": float(X_test.iloc[idx]["grade"]),
            "bedrooms": float(X_test.iloc[idx]["bedrooms"]),
            "bathrooms": float(X_test.iloc[idx]["bathrooms"])
        })

    # Výsledný JSON payload
    payload = {
        "metadata": {
            "dataset": "kc_house_data_preprocessed.csv",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": n_features,
            "features": feature_names,
            "epochs": epochs,
            "batch_size": batch_size,
            "scaler": "StandardScaler"
        },
        "baselines": {
            "linear_regression": {"mae": round(lr_mae, 2), "rmse": round(lr_rmse, 2), "r2": round(lr_r2, 4)},
            "decision_tree": {"mae": round(dt_mae, 2), "rmse": round(dt_rmse, 2), "r2": round(dt_r2, 4)},
            "random_forest": {"mae": round(rf_mae, 2), "rmse": round(rf_rmse, 2), "r2": round(rf_r2, 4)}
        },
        "experiments": exp_results,
        "sample_residuals": sample_residuals
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nAll results precomputed and saved to {out_file}!")


if __name__ == "__main__":
    main()
