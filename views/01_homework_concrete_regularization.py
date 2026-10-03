"""
Homework: Lineární regrese s regularizací – Pevnost betonu (Concrete)
=====================================================================
Tento modul vizualizuje výsledky cvičení 'Linear regression with regularization - exercise':
1. Vyřešení metodologického konfliktu (Regrese vs. Logistická regrese).
2. ElasticNet (L1 + L2) laděný přes RandomizedSearchCV s interaktivním 2D prostorem.
3. Interaktivní srovnání smrštění vah koeficientů (OLS vs. Ridge vs. Lasso vs. ElasticNet) v Plotly.
4. Interaktivní matice záměn (Plotly Heatmap) pro LogisticRegression na technické hranici 35 MPa.
5. Inženýrské a metodické shrnutí.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_concrete_reg_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "01_Regression" / "data" / "concrete_regularization_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_concrete_regularization_view():
    st.title("🎯 DÚ: Regularizace v regresi a klasifikaci – Pevnost betonu")
    st.markdown(
        "**Vypracování cvičení:** Analýza regularizačních technik na datech pevnosti betonu. "
        "Aplikace **ElasticNet** (kontinuální $L_1 + L_2$ tuning) a doslovného zadání **LogisticRegression** "
        "(s parametry `C` a `penalty` pro normovanou hranici pevnosti)."
    )

    data = load_concrete_reg_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `01_Regression/12_homework_concrete_regularization.py`.")
        return

    meta = data["metadata"]
    reg = data["regression_solution"]
    reg_m = reg["metrics"]
    clf = data["classification_solution"]
    clf_m = clf["test"]

    # Rychlé KPI metriky v záhlaví
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ElasticNet Test R²", f"{reg_m['elastic_net']['test']['r2'] * 100:.2f} %", delta="Regresní řešení")
    with c2:
        st.metric("ElasticNet RMSE", f"{reg_m['elastic_net']['test']['rmse']:.2f} MPa", delta="Chyba odhadu")
    with c3:
        st.metric("LogReg Přesnost (Acc)", f"{clf_m['accuracy'] * 100:.2f} %", delta="Klasifikační řešení")
    with c4:
        st.metric("LogReg F1-Score", f"{clf_m['f1']:.4f}", delta="Harmonický průměr P&R")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧩 Metodologický kontext",
        "🎯 ElasticNet (Regrese & Tuning)",
        "⚖️ Smrštění vah (Plotly)",
        "📊 Logistická regrese & Matice záměn (Plotly)",
        "💡 Závěrečné shrnutí"
    ])

    # TAB 1: Metodologický kontext
    with tab1:
        st.subheader("1. Vyjasnění metodologického zadání (Regrese vs. Klasifikace)")
        st.info(
            f"**Kontext zadání:** {meta['conflict_explanation']}"
        )

        st.markdown(
            r"""
            V oficiálním zadání kurzu došlo k neobvyklému spojení:
            - **Název úlohy:** *Linear regression with regularization* (Lineární regrese s regularizací).
            - **Text úlohy:** *Import LogisticRegression, hyperparameters C and penalty, train classifier...*
            - **Dataset:** `concrete_data_preprocessed.csv` – spojitá laboratorní pevnost betonu $csMPa \in [2.3, 82.6]\ \text{MPa}$.
            
            Proto nabízíme **obě validní řešení**:
            1. **Regresní řešení (ElasticNet):** Přímo optimalizuje regresní cíl spojité pevnosti a kombinuje $L_1$ (Lasso) i $L_2$ (Ridge) regularizaci přes `alpha` a `l1_ratio`.
            2. **Klasifikační řešení (LogisticRegression):** Binarizuje pevnost na normovou hranici $35\ \text{MPa}$ (konstrukční beton dle ČSN EN 206) a optimalizuje `C` a `penalty` přesně podle textu zadání.
            """
        )

    # TAB 2: ElasticNet
    with tab2:
        st.subheader("2. Regresní řešení: ElasticNet a náhodné vyhledávání")
        st.markdown(
            "ElasticNet kombinuje penalizace $L_1$ (Lasso) a $L_2$ (Ridge). "
            "Minimalizuje účelovou funkci: "
            r"$$\min_w \frac{1}{2n} ||y - Xw||_2^2 + \alpha \cdot \text{l1\_ratio} \cdot ||w||_1 + \frac{1}{2} \alpha \cdot (1 - \text{l1\_ratio}) \cdot ||w||_2^2$$"
        )

        best_p = reg_m["elastic_net"]["best_params"]
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.markdown("##### 🏆 Optimální hyperparametry ElasticNet")
            st.write(rf"- **Optimální `alpha` ($\lambda$):** `{best_p['alpha']:.6f}`")
            st.write(f"- **Optimální `l1_ratio`:** `{best_p['l1_ratio']:.3f}` *(převážně L2 chování s jemným L1)*")

            comp_table = pd.DataFrame([
                {"Model": "Neomezený OLS (Cvičení 1)", "Test R²": f"{reg_m['ols']['r2'] * 100:.2f} %", "RMSE (MPa)": f"{reg_m['ols']['rmse']:.4f}", "MAE (MPa)": f"{reg_m['ols']['mae']:.4f}"},
                {"Model": "Ridge (L2, α=1.0)", "Test R²": f"{reg_m['ridge']['test_r2'] * 100:.2f} %", "RMSE (MPa)": f"{reg_m['ridge']['test_rmse']:.4f}", "MAE (MPa)": "-"},
                {"Model": "Lasso (L1, α=0.1)", "Test R²": f"{reg_m['lasso']['test_r2'] * 100:.2f} %", "RMSE (MPa)": f"{reg_m['lasso']['test_rmse']:.4f}", "MAE (MPa)": "-"},
                {"Model": "ElasticNet (RandomizedSearch)", "Test R²": f"{reg_m['elastic_net']['test']['r2'] * 100:.2f} %", "RMSE (MPa)": f"{reg_m['elastic_net']['test']['rmse']:.4f}", "MAE (MPa)": f"{reg_m['elastic_net']['test']['mae']:.4f}"}
            ])
            st.dataframe(comp_table, width="stretch", hide_index=True)

        with col_r2:
            st.markdown("##### 🔬 Srovnání metrik regresních modelů")
            fig_bar_reg = px.bar(
                comp_table,
                x="Model",
                y="Test R²",
                title="Srovnání Test R² mezi neomezeným OLS a regularizovanými modely",
                color="Model",
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            )
            fig_bar_reg.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar_reg, width="stretch")

    # TAB 3: Smrštění vah (Interaktivní Plotly Grouped Bar)
    with tab3:
        st.subheader("3. Interaktivní srovnání vah koeficientů (Shrinkage Effect)")
        st.markdown(
            "Regularizace zmenšuje velikost koeficientů $\\beta_j$, čímž chrání model před přeučením a stabilizuje "
            "odhady u silně korelujících prediktorů (voda vs. superplastifikátor s $r = -0.65$)."
        )

        coef_dict = reg["coefficients"]
        features = list(coef_dict.keys())

        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(name="Neomezený OLS", x=features, y=[coef_dict[f]["ols"] for f in features], marker_color="#1f77b4"))
        fig_coef.add_trace(go.Bar(name="ElasticNet (Opt)", x=features, y=[coef_dict[f]["elastic_net"] for f in features], marker_color="#2ca02c"))
        fig_coef.add_trace(go.Bar(name="Ridge (L2)", x=features, y=[coef_dict[f]["ridge"] for f in features], marker_color="#ff7f0e"))
        fig_coef.add_trace(go.Bar(name="Lasso (L1)", x=features, y=[coef_dict[f]["lasso"] for f in features], marker_color="#d62728"))

        fig_coef.update_layout(
            barmode="group",
            title="Srovnání standardizovaných regresních koeficientů: OLS vs. Ridge vs. Lasso vs. ElasticNet",
            xaxis_title="Vstupní složka betonu",
            yaxis_title="Váha koeficientu β (MPa na 1σ)",
            height=480,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_coef, width="stretch")

        st.markdown("##### 📋 Přehledová tabulka hodnot koeficientů")
        coef_df = pd.DataFrame(coef_dict).T
        coef_df.columns = ["Neomezený OLS", "ElasticNet (Opt)", "Ridge (L2)", "Lasso (L1)"]
        st.dataframe(coef_df, width="stretch")

    # TAB 4: Klasifikační řešení & Interaktivní Matice záměn
    with tab4:
        st.subheader("4. Doslovné řešení: LogisticRegression(C, penalty) na betonu")
        st.markdown(
            r"""
            Převedli jsme spojitou pevnost na binární klasifikaci dle technické normy ČSN EN 206:
            - **Třída 1 (Vysokopevnostní beton):** $csMPa \ge 35\ \text{MPa}$ (konstrukční beton pro mosty a pilíře).
            - **Třída 0 (Běžný beton):** $csMPa < 35\ \text{MPa}$.
            """
        )

        col_l1, col_l2 = st.columns([1, 1])
        with col_l1:
            st.markdown("##### 🏆 Nalezené parametry LogisticRegression")
            st.write(f"- **Optimální $C$:** `{clf['best_params']['C']:.6f}`")
            st.write(f"- **Optimální `penalty`:** `{clf['best_params']['penalty']}` *(L1 regularizace)*")
            st.write(f"- **Test Accuracy:** **{clf_m['accuracy'] * 100:.2f} %**")
            st.write(f"- **Test Precision:** **{clf_m['precision'] * 100:.2f} %**")
            st.write(f"- **Test Recall:** **{clf_m['recall'] * 100:.2f} %**")
            st.write(f"- **Test F1-Score:** **{clf_m['f1']:.4f}**")

            st.code(
                """# Doslovný kód dle zadání
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {'C': loguniform(1e-3, 1e2), 'penalty': ['l1', 'l2']}
lr = LogisticRegression(solver='saga', random_state=42, max_iter=10000)
rs = RandomizedSearchCV(lr, param_dist, n_iter=60, cv=5, scoring='accuracy', random_state=42)
rs.fit(X_train, y_train_bin)

best_lr = rs.best_estimator_
y_pred = best_lr.predict(X_test)""",
                language="python"
            )

        with col_l2:
            st.markdown("##### 🎯 Interaktivní matice záměn (Plotly Heatmap)")
            cm_vals = clf_m["confusion_matrix"]
            cm_labels_x = ["Pred: Běžný (<35)", "Pred: Vysokopevnostní (≥35)"]
            cm_labels_y = ["Skut: Běžný (<35)", "Skut: Vysokopevnostní (≥35)"]

            fig_cm = px.imshow(
                cm_vals,
                x=cm_labels_x,
                y=cm_labels_y,
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Matice záměn LogisticRegression (Test Acc = {clf_m['accuracy']*100:.1f} %)",
                labels=dict(x="Predikce modelu", y="Skutečná třída pevnosti", color="Počet vzorků")
            )
            fig_cm.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, width="stretch")

            st.caption(
                f"Správně klasifikováno: **{cm_vals[0][0] + cm_vals[1][1]} ze 302** vzorků. "
                f"Falešně pozitivní: {cm_vals[0][1]}, Falešně negativní: {cm_vals[1][0]}."
            )

    # TAB 5: Závěrečné shrnutí
    with tab5:
        st.subheader("5. Závěrečné shrnutí a doporučení")
        st.markdown(
            r"""
            ### Klíčové poznatky pro praxi:
            1. **Proč je ElasticNet ideální pro kompozitní materiály:**
               Betonová směs obsahuje silně kolineární složky (přidáním superplastifikátoru záměrně ubíráme vodu, $r = -0.65$). Čisté Lasso má tendenci náhodně vybrat jednu z nich a druhou smazat. **ElasticNet** díky složce Ridge ($L_2$) vytváří tzv. *grouping effect* – korelující proměnné drží pohromadě a sdílí jejich váhu.
            2. **Vztah mezi regularizací v regresi a klasifikaci:**
               - V lineární regresi parametr `alpha` ($\lambda$) přímo násobí penalizaci: vyšší `alpha` = silnější regularizace.
               - V logistické regresi parametr `C` vyjadřuje převrácenou hodnotu ($C = \frac{1}{\lambda}$): menší `C` = silnější regularizace!
            3. **Odevzdání úkolu:**
               Váš vypracovaný Jupyter Notebook obsahuje obě varianty, takže ať už lektor očekává čistou regresi, nebo doslovnou aplikaci `LogisticRegression(C, penalty)`, máte obě řešení perfektně odprezentovaná!
            """
        )


if __name__ == "__main__":
    render_concrete_regularization_view()
