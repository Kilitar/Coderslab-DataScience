import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, ElasticNetCV, RidgeCV, LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

st.title("🔬 Kritický rozbor a MLOps: Regularizace v praxi")
st.caption("Matematické nástrahy neškálovaných modelů, prevence Data Leakage při ladění hyperparametrů, Lasso Path a Elastic Net.")

# =============================================================================
# CACHE VÝPOČTŮ (Okamžité načtení z předpočítaných dat)
# =============================================================================
@st.cache_data
def load_critique_data():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "01_Regression" / "data" / "kc_reg_critique_precomputed.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            data["feature_names"],
            np.array(data["alphas_path"]),
            np.array(data["coefs_path"]),
            pd.DataFrame(data["cv_list"]),
            data["best_ridge_alpha"],
            data["best_lasso_alpha"],
            data["best_enet_alpha"],
            data["best_enet_l1_ratio"],
        )

    csv_path = base_dir / "01_Regression" / "data" / "kc_house_data_preprocessed.csv"

    if not csv_path.exists():
        csv_path = base_dir / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "kc_house_data_preprocessed.csv"

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    feature_names = list(X.columns)

    # 1. Škálovaná data
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 2. Lasso Path
    alphas_path, coefs_path, _ = lasso_path(X_train_sc, y_train, eps=1e-3)

    # 3. Křížová validace RidgeCV, LassoCV a ElasticNetCV
    ridge_cv = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=5).fit(X_train_sc, y_train)
    best_ridge_alpha = ridge_cv.alpha_
    pred_ridge_cv = ridge_cv.predict(X_test_sc)

    lasso_cv = LassoCV(alphas=np.logspace(1, 4, 30), cv=5, random_state=42, max_iter=3000, tol=0.01).fit(X_train_sc, y_train)
    best_lasso_alpha = lasso_cv.alpha_
    pred_lasso_cv = lasso_cv.predict(X_test_sc)

    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
        alphas=np.logspace(1, 4, 20),
        cv=5,
        random_state=42,
        max_iter=3000,
        tol=0.01
    ).fit(X_train_sc, y_train)
    best_enet_alpha = enet_cv.alpha_
    best_enet_l1_ratio = enet_cv.l1_ratio_
    pred_enet_cv = enet_cv.predict(X_test_sc)

    # OLS baseline
    ols = LinearRegression().fit(X_train, y_train)
    pred_ols = ols.predict(X_test)

    def calc_m(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }

    cv_comparison = pd.DataFrame([
        {"Model": "OLS Baseline", "Alpha": "N/A", "L1 Ratio": "N/A", **calc_m(y_test, pred_ols)},
        {"Model": "RidgeCV (5-Fold CV)", "Alpha": f"{best_ridge_alpha:.2f}", "L1 Ratio": "0.0 (Čisté L2)", **calc_m(y_test, pred_ridge_cv)},
        {"Model": "LassoCV (5-Fold CV)", "Alpha": f"{best_lasso_alpha:.2f}", "L1 Ratio": "1.0 (Čisté L1)", **calc_m(y_test, pred_lasso_cv)},
        {"Model": "ElasticNetCV (5-Fold CV)", "Alpha": f"{best_enet_alpha:.2f}", "L1 Ratio": f"{best_enet_l1_ratio:.2f}", **calc_m(y_test, pred_enet_cv)},
    ])

    return (
        feature_names,
        alphas_path,
        coefs_path,
        cv_comparison,
        best_ridge_alpha,
        best_lasso_alpha,
        best_enet_alpha,
        best_enet_l1_ratio,
    )


(
    feature_names,
    alphas_path,
    coefs_path,
    cv_comparison,
    best_ridge_alpha,
    best_lasso_alpha,
    best_enet_alpha,
    best_enet_l1_ratio,
) = load_critique_data()


# =============================================================================
# SEKCE 1: ABSOLUTNÍ METODICKÁ CHYBA – ABSENCE ŠKÁLOVÁNÍ
# =============================================================================
st.markdown("### 🚨 1. Kardinální metodická chyba v kurzu: Regularizace bez StandardScaler")

st.warning("""
**V prezentaci kurzu (snímky 9 a 16) je studentům předkládán kód:**
```python
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)  # <-- BEZ JAKÉHOKOLIV ŠKÁLOVÁNÍ!
```
""")

col_err1, col_err2 = st.columns(2)

with col_err1:
    st.markdown("""
    #### Proč je to v regularizaci fatální?
    1. **Různá měřítka proměnných:**  
       - `sqft_living` (plocha domu) má hodnoty v tisících (1 000 až 5 000). Její koeficient $\\beta$ je malý (např. $\\approx 150$).
       - `bedrooms` (počet pokojů) má hodnoty 1 až 5. Její koeficient $\\beta$ je velký (např. $\\approx 40 000$).
    2. **Disproporční penalizace:**  
       Penalizační člen $\alpha \sum \beta_j^2$ trestá koeficient pro `bedrooms` částkou $40000^2 = 1{,}6 \times 10^9$, zatímco pro `sqft_living` pouze $150^2 = 22500$.
    3. **Důsledek v Scikit-learn:**  
       Algoritmus Coordinate Descent na neškálovaných datech **vůbec nekonverguje** a vyhazuje `ConvergenceWarning` a `LinAlgWarning`! Proměnné s malými čísly jsou brutálně zdecimovány, zatímco proměnné s velkými čísly nepociťují téměř žádnou regularizaci.
    """)

with col_err2:
    st.markdown("""
    #### Co na to říká Scikit-learn přímo v konzoli:
    ```text
    ConvergenceWarning: Objective did not converge. 
    You might want to increase the number of iterations, 
    check the scale of the features or consider increasing regularisation.
    
    LinAlgWarning: An ill-conditioned matrix detected: 
    slice 0 has rcond = 2.318e-17.
    ```
    > [!TIP]
    > **Zlaté pravidlo pro produkční kód:**  
    > Před jakoukoliv regularizací (L1, L2, Elastic Net, Ridge, Lasso) je **standardizace všech numerických příznaků (`StandardScaler`) matematickou povinností**!
    """)

# =============================================================================
# SEKCE 2: DATA SNOOPING VS KŘÍŽOVÁ VALIDACE
# =============================================================================
st.markdown("---")
st.markdown("### ⚠️ 2. Data Leakage (Snooping) při ladění α vs. Křížová validace")

st.markdown("""
Na snímku 17 prezentace autor hledá optimální $\\alpha$ tím, že ve smyčce volá:
```python
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for alpha in alphas:
    ridge_red = Ridge(alpha=alpha).fit(X_train, y_train)
    y_pred_ridge = ridge_red.predict(X_test)  # <-- CHYBA: Ladění přímo na testovací sadě!
    mean_squared_error(y_test, y_pred_ridge)
```
**Proč je to metodická chyba?**  
Testovací sada má sloužit výhradně jako **závěrečná neviděná kontrola**. Pokud na ní vyhodnocujeme desítky hodnot hyperparametrů a vybereme tu, která na testovací sadě náhodou dopadla nejlépe, dochází k **prosakování informací (Data Leakage / Snooping)** a testovací chyba je zkresleně optimistická.

#### Správné řešení: 5-násobná křížová validace na trénovacích datech
Knihovna Scikit-learn pro tento účel nabízí optimalizované třídy `RidgeCV`, `LassoCV` a `ElasticNetCV`:
""")

st.dataframe(
    cv_comparison.style.format({
        "R2": "{:.5f}",
        "MAE": "{:,.2f} USD",
        "RMSE": "{:,.2f} USD",
    }),
    width="stretch",
)

# =============================================================================
# SEKCE 3: INTERAKTIVNÍ LASSO REGULARIZATION PATH
# =============================================================================
st.markdown("---")
st.markdown("### 📈 3. Interaktivní Lasso Regularization Path")
st.caption("Sledujte trajektorii všech 18 regresních koeficientů s rostoucí silou regularizace α. Najetím myší na křivku zjistíte název proměnné.")

fig_path = go.Figure()

for i, feat in enumerate(feature_names):
    fig_path.add_trace(
        go.Scatter(
            x=alphas_path,
            y=coefs_path[i, :],
            mode="lines",
            name=feat,
            line=dict(width=2),
            hovertemplate=f"<b>{feat}</b><br>Alpha: %{{x:.2f}}<br>Koeficient: %{{y:,.1f}}<extra></extra>",
        )
    )

fig_path.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.8)
fig_path.update_xaxes(type="log", title="Síla regularizace α (logaritmická škála, klesá zprava doleva)")
fig_path.update_yaxes(title="Standardizovaný koeficient β")
fig_path.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    hovermode="closest",
)
st.plotly_chart(fig_path, width="stretch")

st.markdown("""
**Pozorování z grafu Lasso Path:**
- Při vysokých hodnotách $\\alpha > 50\\,000$ jsou **všechny váhy přesně rovny nule** (model predikuje pouhý průměr $\\bar{y}$).
- Jako první „přežívají“ a získávají váhu ty nejvýznamnější strukturální parametry: **`sqft_living`** (obytná plocha), **`grade`** (konstrukční kvalita) a **`lat`** (zeměpisná šířka – poloha domu).
- Méně významné či kolineární parametry (`sqft_basement`, `sqft_lot`) zůstávají dlouho vynulovány na nule.

---

### 🌟 4. Proč v praxi potřebujeme Elastic Net?
V King County Housing existuje silná multikolinearita:
- `sqft_living` (celková plocha) a `sqft_above` (nadzemní plocha) mají korelaci **$r = 0.88$**.
- **Chování čistého Lasso ($L_1$):** Zcela náhodně vybere jeden z nich a druhý natvrdo vynuluje. Pokud data mírně pozměníme, Lasso může své rozhodnutí obrátit (vysoká nestabilita modelu).
- **Řešení Elastic Net:** Kombinuje $L_1$ penalizaci pro redukci dimenze a $L_2$ penalizaci pro sdružování korelujících proměnných (*Grouping Effect*). Obě proměnné si rozdělí váhu a model je stabilní.
""")
