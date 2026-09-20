from pathlib import Path
import streamlit as st

st.title("📥 Tahák Dne 1: Scikit-learn Cheatsheet ke stažení")
st.caption("Kompletní reprezentativní tahák do kapsy: Syntéza importů, vzorců, hyperparametrů, kompletní workflow a checklistů pro praktické využití a technické pohovory.")

base_dir = Path(__file__).resolve().parent.parent
summary_doc_path = base_dir / "01_Regression" / "theory" / "07_day_1_summary.md"

if summary_doc_path.exists():
    with open(summary_doc_path, "r", encoding="utf-8") as f:
        md_text = f.read()
else:
    md_text = "# Day 1 Summary Cheatsheet\nDokument se připravuje."

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 📄 Kompletní přehled v Markdownu / tisknutelném formátu")
    st.markdown("Stáhněte si ucelený souhrn všech kapitol, srovnávací tabulku i česko-anglický glosář pojmů:")
with col2:
    st.download_button(
        label="⬇️ Stáhnout Tahák (.md)",
        data=md_text,
        file_name="Day_1_Regression_Cheatsheet_CodersLab.md",
        mime="text/markdown",
        width="stretch"
    )

st.markdown("---")

tab_pipe, tab_code, tab_params, tab_math, tab_gotchas = st.tabs([
    "🚀 Kompletní End-to-End Workflow",
    "💻 5 klíčových modelů (Scikit-learn šablony)",
    "⚙️ Přehled hyperparametrů k ladění",
    "📐 Matematické vzorce metrik a penalizací",
    "⚠️ Top 5 chyb u pohovorů (Gotchas)"
])

# =============================================================================
# TAB 1: WORKFLOW
# =============================================================================
with tab_pipe:
    st.markdown("#### 🔄 Standardní 6krokové workflow regresního projektu v Pythonu:")
    st.code('''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Načtení dat a vyčištění
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 2. Rozdělení PŘED jakoukoliv transformací (Prevence Data Leakage!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Sestavení Pipeline (Scaler + Model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Ridge())
])

# 4. Křížová validace a ladění hyperparametrů
param_grid = {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

# 5. Predikce na testovací sadě
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 6. Vyhodnocení metrik
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R²: {r2:.4f} | RMSE: {rmse:,.1f} | MAE: {mae:,.1f}")''', language="python")

# =============================================================================
# TAB 2: 5 MODELŮ
# =============================================================================
with tab_code:
    st.markdown("#### 1. Základní OLS Lineární regrese:")
    st.code('''from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
intercept = model.intercept_
slopes = model.coef_''', language="python")

    st.markdown("#### 2. Ridge regrese (L2 regularizace – řeší multikolinearitu):")
    st.code('''from sklearn.linear_model import Ridge
# VŽDY použijte StandardScaler před Ridge!
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)''', language="python")

    st.markdown("#### 3. Lasso regrese (L1 regularizace – automatický výběr příznaků):")
    st.code('''from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
# Proměnné s nulovým koeficientem byly vyřazeny:
selected_features = X.columns[lasso.coef_ != 0]''', language="python")

    st.markdown("#### 4. Polynomiální regrese (Nelineární expanze):")
    st.code('''from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=10.0)
)
poly_model.fit(X_train, y_train)''', language="python")

    st.markdown("#### 5. Rozhodovací strom (CART – zachytí skoky a zóny):")
    st.code('''from sklearn.tree import DecisionTreeRegressor
# Strom NEPOTŘEBUJE StandardScaler!
tree = DecisionTreeRegressor(
    max_depth=10,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42
)
tree.fit(X_train, y_train)
importances = tree.feature_importances_''', language="python")

# =============================================================================
# TAB 3: HYPERPARAMETRY
# =============================================================================
with tab_params:
    st.markdown("#### ⚙️ Nejdůležitější parametry v Scikit-learn a co dělají:")
    st.markdown(r"""
    | Algoritmus | Hyperparametr | Výchozí hodnota | Co ovlivňuje | Doporučený směr ladění |
    | :--- | :--- | :---: | :--- | :--- |
    | **Ridge** | `alpha` | `1.0` | Síla L2 stlačování vah $\sum \beta^2$ | Větší $\alpha$ = menší variance (tlumí přeučení), zkoušet log škálu `[0.01, 0.1, 1, 10, 100]` |
    | **Lasso** | `alpha` | `1.0` | Síla L1 nulování vah $\sum |\beta|$ | Větší $\alpha$ = více nulových vah (agresivnější Feature Selection) |
    | **ElasticNet** | `l1_ratio` | `0.5` | Poměr mezi L1 a L2 penalizací | `0.0` = čistý Ridge, `1.0` = čisté Lasso, mezilehlé hodnoty = kombinace |
    | **Polynomial** | `degree` | `2` | Stupeň mocniny | **Nikdy nevolit > 3** na mnoha sloupcích (exploze dimenzí a paměti!) |
    | **DecisionTree** | `max_depth` | `None` | Maximální počet pater stromu | Klíčová ochrana proti overfittingu: omezit na `8 – 15` pater |
    | **DecisionTree** | `min_samples_leaf` | `1` | Minimální počet vzorků v listu | Zvýšit na `5 – 50`, aby list nereprezentoval jediný odlehlý dům |
    | **DecisionTree** | `min_samples_split` | `2` | Počet vzorků nutný k dalšímu dělení | Nastavit na cca dvojnásobek `min_samples_leaf` |
    | **DecisionTree** | `ccp_alpha` | `0.0` | Post-pruning komplexní parametr | Vybrat z `cost_complexity_pruning_path` pro matematické prořezání |
    """)

# =============================================================================
# TAB 4: VZORCE
# =============================================================================
with tab_math:
    st.markdown(r"""
    #### 1. Metriky kvality regrese:
    - **Koeficient determinace ($R^2$):**  
      $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
    - **Adjusted $R^2$ (penalizovaný za počet parametrů $p$):**  
      $$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$
    - **Mean Absolute Error (MAE):**  
      $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
    - **Root Mean Squared Error (RMSE):**  
      $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

    #### 2. Účelové funkce s penalizací (Ztrátové funkce):
    - **OLS:** $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$
    - **Lasso (L1):** $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p |\beta_j|$
    - **Ridge (L2):** $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p \beta_j^2$
    - **Cost-Complexity Pruning (CART):** $R_\alpha(T) = R(T) + \alpha |T|$
    """)

# =============================================================================
# TAB 5: GOTCHAS
# =============================================================================
with tab_gotchas:
    st.markdown("#### ⚠️ 5 nejčastějších chyb u technických pohovorů a v projektech:")
    st.markdown(r"""
    1. **Data Leakage při škálování:**  
       ❌ *Chyba:* Zavolání `scaler.fit_transform(X)` na celém datasetu před splitováním.  
       ✔️ *Správně:* `scaler.fit_transform(X_train)` a na testovací sadě POUZE `scaler.transform(X_test)`.
    2. **Škálování u Rozhodovacích stromů:**  
       ❌ *Chyba:* Zbytečné škálování dat před `DecisionTreeRegressor`.  
       ✔️ *Správně:* Stromy pracují s pořadím hodnot ($x_j \le s$), škálování nemá na strukturu stromu žádný vliv.
    3. **Extrapolace se stromy:**  
       ❌ *Chyba:* Použití stromu pro předpověď budoucího trendu růstu cen.  
       ✔️ *Správně:* Stromy neumí jít nad strop trénovací sady; pro trend zvolte lineární/polynomiální regresi.
    4. **Polynomiální expanze na kategorických proměnných:**  
       ❌ *Chyba:* Aplikace `PolynomialFeatures` na One-Hot zakódované sloupce (např. $\text{cut\_Ideal}^2$).  
       ✔️ *Správně:* Expanzi provádějte pouze na spojitých fyzikálních veličinách (`carat`, `x`, `y`, `z`).
    5. **Porovnávání $R^2$ mezi různými datasety:**  
       ❌ *Chyba:* Tvrdit, že model s $R^2 = 0.90$ na diamantech je lepší než model s $R^2 = 0.80$ na domech.  
       ✔️ *Správně:* $R^2$ závisí na vnitřním šumu v datech dané domény, nelze je absolutně srovnávat mezi sebou.
    """)
