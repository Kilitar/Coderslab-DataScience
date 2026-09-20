from pathlib import Path
import streamlit as st

st.title("📥 Tahák Dne 1: Scikit-learn Cheatsheet ke stažení")
st.caption("Kompletní reprezentativní tahák do kapsy: Syntéza importů, vzorců, hyperparametrů a checklistů pro praktické využití a technické pohovory.")

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

tab_code, tab_math = st.tabs([
    "💻 Rychlý kód: Nejpoužívanější Scikit-learn šablony",
    "📐 Matematické vzorce metrik a penalizací"
])

with tab_code:
    st.markdown("#### 1. Kompletní Pipeline pro Ridge a Lasso s normalizací:")
    st.code('''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# Pipeline zaručuje, že scaler se fituje POUZE na train sadě (prevence data leakage!)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Optimální alpha: {grid.best_params_}")''', language="python")

    st.markdown("#### 2. Decision Tree s prořezáním (Optimalizace hloubky a listů):")
    st.code('''from sklearn.tree import DecisionTreeRegressor

# Strom nepotřebuje StandardScaler!
tree = DecisionTreeRegressor(
    max_depth=12,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42
)
tree.fit(X_train, y_train)

# Feature importance (MDI)
importances = tree.feature_importances_''', language="python")

with tab_math:
    st.markdown(r"""
    #### Metriky přesnosti regrese:
    - **Koeficient determinace:** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
    - **Adjusted $R^2$:** $R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$
    - **Mean Absolute Error:** $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
    - **Root Mean Squared Error:** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$

    #### Účelové funkce s regularizací:
    - **Lasso (L1):** $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p |\beta_j|$
    - **Ridge (L2):** $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \sum_{j=1}^p \beta_j^2$
    - **Cost-Complexity Pruning (Tree):** $R_\alpha(T) = R(T) + \alpha |T|$
    """)
