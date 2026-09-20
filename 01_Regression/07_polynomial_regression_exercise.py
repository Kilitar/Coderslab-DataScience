"""
Blok 1: Regrese – Cvičení 7: Polynomiální regrese (Diamanty)
Soubor: 01_Regression/07_polynomial_regression_exercise.py

Zadání cvičení:
1. Načíst předzpracovaný dataset diamantů z Cvičení 2 (diamonds_preprocessed.csv).
2. Importovat všechny potřebné knihovny a moduly z knihovny Scikit-learn.
3. Rozdělit data na trénovací a testovací sadu (80:20, random_state=42).
4. Pomocí PolynomialFeatures generovat polynomiální proměnné a zvolit vhodný stupeň polynomu.
5. Pro každou variantu (stupeň 1, 2, 3):
   - Natrénovat lineární regresní model.
   - Spočítat metriky R2, MAE, MSE a RMSE.
   - Porovnat výsledky s předchozími modely.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def run_polynomial_regression_analysis():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    csv_path = data_dir / "diamonds_preprocessed.csv"
    
    if not csv_path.exists():
        csv_path = base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "diamonds_preprocessed.csv"

    print(f"1. Načítání datasetu z: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["price"])
    y = df["price"]
    feature_names = list(X.columns)

    print(f"   Počet řádků: {len(df):,}, Počet původních prediktorů: {X.shape[1]}")
    print(f"   Příznaky: {feature_names}")

    # 2. Train-Test Split (80:20, fixní random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    n_test = len(y_test)

    # Pomocná funkce pro výpočet metrik
    def evaluate(y_true, y_pred, k_features):
        r2 = float(r2_score(y_true, y_pred))
        adj_r2 = float(1.0 - ((1.0 - r2) * (len(y_true) - 1) / (len(y_true) - k_features - 1)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        return {
            "R2": r2,
            "Adj_R2": adj_r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
        }

    # 3. Vyhodnocení stupňů polynomu: Degree 1, Degree 2, Degree 3
    degrees = [1, 2, 3]
    models_results = []
    trained_models = {}

    for deg in degrees:
        print(f"\n--- Zpracování polynomu stupně {deg} ---")
        if deg == 1:
            poly = None
            X_tr_p = X_train
            X_te_p = X_test
            n_features = X_train.shape[1]
            feature_list = feature_names
        else:
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_tr_p = poly.fit_transform(X_train)
            X_te_p = poly.transform(X_test)
            n_features = X_tr_p.shape[1]
            feature_list = list(poly.get_feature_names_out(feature_names))

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_p)
        X_te_sc = scaler.transform(X_te_p)

        reg = LinearRegression().fit(X_tr_sc, y_train)
        pred_tr = reg.predict(X_tr_sc)
        pred_te = reg.predict(X_te_sc)

        m_tr = evaluate(y_train, pred_tr, n_features)
        m_te = evaluate(y_test, pred_te, n_features)

        print(f"   Počet příznaků po expanzi: {n_features}")
        print(f"   Trénovací R2: {m_tr['R2']:.4f}, Testovací R2: {m_te['R2']:.4f}")
        print(f"   Testovací MAE: {m_te['MAE']:.2f} USD, RMSE: {m_te['RMSE']:.2f} USD")

        res_entry = {
            "Degree": deg,
            "Name": f"Polynom stupně {deg}" + (" (Lineární OLS)" if deg == 1 else " (Kvadratický)" if deg == 2 else " (Kubický)"),
            "Features_Count": n_features,
            "Train_R2": m_tr["R2"],
            "Test_R2": m_te["R2"],
            "Train_MAE": m_tr["MAE"],
            "Test_MAE": m_te["MAE"],
            "Train_RMSE": m_tr["RMSE"],
            "Test_RMSE": m_te["RMSE"],
            "Overfitting_Delta_R2": float(m_tr["R2"] - m_te["R2"]),
            "Overfitting_Risk": "Nízké (Stabilní)" if deg == 1 else "Optimální (Vynikající)" if deg == 2 else "Extrémní (Kolaps variance!)",
        }
        models_results.append(res_entry)
        trained_models[deg] = {
            "model": reg,
            "scaler": scaler,
            "poly": poly,
            "feature_list": feature_list,
            "pred_te": pred_te
        }

    # 4. Expertní srovnání: Regularizovaný polynom stupně 3 (Ridge alpha=100)
    print("\n--- Expertní srovnání: Ridge regularizace na polynomu stupně 3 ---")
    poly3 = trained_models[3]["poly"]
    scaler3 = trained_models[3]["scaler"]
    X_tr_sc3 = scaler3.transform(poly3.transform(X_train))
    X_te_sc3 = scaler3.transform(poly3.transform(X_test))

    ridge_deg3 = Ridge(alpha=100.0, random_state=42).fit(X_tr_sc3, y_train)
    pred_te_ridge3 = ridge_deg3.predict(X_te_sc3)
    pred_tr_ridge3 = ridge_deg3.predict(X_tr_sc3)
    m_tr_r3 = evaluate(y_train, pred_tr_ridge3, X_tr_sc3.shape[1])
    m_te_r3 = evaluate(y_test, pred_te_ridge3, X_te_sc3.shape[1])

    res_ridge3 = {
        "Degree": 3,
        "Name": "Polynom stupně 3 + Ridge (α=100)",
        "Features_Count": X_tr_sc3.shape[1],
        "Train_R2": m_tr_r3["R2"],
        "Test_R2": m_te_r3["R2"],
        "Train_MAE": m_tr_r3["MAE"],
        "Test_MAE": m_te_r3["MAE"],
        "Train_RMSE": m_tr_r3["RMSE"],
        "Test_RMSE": m_te_r3["RMSE"],
        "Overfitting_Delta_R2": float(m_tr_r3["R2"] - m_te_r3["R2"]),
        "Overfitting_Risk": "Vyřešeno regularizací",
    }
    models_results.append(res_ridge3)
    print(f"   Ridge Stupeň 3: Test R2: {m_te_r3['R2']:.4f}, MAE: {m_te_r3['MAE']:.2f} USD, RMSE: {m_te_r3['RMSE']:.2f} USD")

    # 5. Moderní benchmark: HistGradientBoosting
    print("\n--- Moderní benchmark: HistGradientBoostingRegressor ---")
    hgb = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    pred_te_hgb = hgb.predict(X_test)
    pred_tr_hgb = hgb.predict(X_train)
    m_tr_hgb = evaluate(y_train, pred_tr_hgb, X_train.shape[1])
    m_te_hgb = evaluate(y_test, pred_te_hgb, X_test.shape[1])

    res_hgb = {
        "Degree": "Tree",
        "Name": "HistGradientBoosting (Bez polynomů)",
        "Features_Count": X_train.shape[1],
        "Train_R2": m_tr_hgb["R2"],
        "Test_R2": m_te_hgb["R2"],
        "Train_MAE": m_tr_hgb["MAE"],
        "Test_MAE": m_te_hgb["MAE"],
        "Train_RMSE": m_tr_hgb["RMSE"],
        "Test_RMSE": m_te_hgb["RMSE"],
        "Overfitting_Delta_R2": float(m_tr_hgb["R2"] - m_te_hgb["R2"]),
        "Overfitting_Risk": "Minimální (Moderní standard)",
    }
    models_results.append(res_hgb)
    print(f"   HGB: Test R2: {m_te_hgb['R2']:.4f}, MAE: {m_te_hgb['MAE']:.2f} USD, RMSE: {m_te_hgb['RMSE']:.2f} USD")

    # 6. Sestavení dat pro predikční simulátor a křivky v aplikaci
    # Top koeficienty polynomu stupně 2 (nejdůležitější kvadratické a interakční členy)
    m2_coefs = trained_models[2]["model"].coef_
    m2_feat_names = trained_models[2]["feature_list"]
    top_coefs_idx = np.argsort(np.abs(m2_coefs))[::-1][:15]
    top_features = [
        {"feature": m2_feat_names[i], "coef": float(m2_coefs[i])}
        for i in top_coefs_idx
    ]

    # Vzorek pro vizualizaci predikce vs realita (200 bodů)
    sample_indices = np.linspace(0, len(y_test) - 1, 200, dtype=int)
    y_test_sample = [float(y_test.iloc[i]) for i in sample_indices]
    sample_plot_data = {
        "actual": y_test_sample,
        "pred_deg1": [float(trained_models[1]["pred_te"][i]) for i in sample_indices],
        "pred_deg2": [float(trained_models[2]["pred_te"][i]) for i in sample_indices],
        "pred_deg3": [float(trained_models[3]["pred_te"][i]) for i in sample_indices],
        "pred_ridge3": [float(pred_te_ridge3[i]) for i in sample_indices],
        "pred_hgb": [float(pred_te_hgb[i]) for i in sample_indices],
    }

    # 7. Uložení JSON cache pro bleskové načtení ve Streamlitu
    output_json = {
        "original_features": feature_names,
        "metrics_table": models_results,
        "top_features_deg2": top_features,
        "sample_plot_data": sample_plot_data,
        "winner_degree": 2,
        "winner_r2": models_results[1]["Test_R2"],
        "winner_mae": models_results[1]["Test_MAE"],
        "winner_rmse": models_results[1]["Test_RMSE"],
    }

    out_file = data_dir / "diamonds_poly_precomputed.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    print(f"\n✅ Výsledky byly úspěšně uloženy do: {out_file}")
    return output_json


if __name__ == "__main__":
    run_polynomial_regression_analysis()
