from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🔬 Cvičení 8: Expertní analýza & Diagnostika rozhodovacího stromu")
st.caption("Detailní metodologický rozbor: Proč strom poráží OLS, anatomie přeučení v listech, neschopnost extrapolace luxusních vil a Cost-Complexity Pruning (ccp_alpha).")

tab_comp, tab_overfit, tab_resid, tab_pruning = st.tabs([
    "⚔️ Proč strom překonal Lineární regresi?",
    "🔬 Anatomie přeučení: 35 pater vs. 14 pater",
    "📉 Diagnostika reziduí & Strop na luxusu",
    "✂️ Cost-Complexity Pruning (ccp_alpha)"
])

# =============================================================================
# TAB 1: PROČ STROM PŘEKONAL OLS
# =============================================================================
with tab_comp:
    st.markdown("### ⚔️ Proč rozhodovací strom dosáhl R² = 0.795 oproti OLS (R² ≈ 0.70)?")
    st.markdown(r"""
    V Cvičení 1 dosáhla základní OLS lineární regrese koeficientu determinace **$R^2 \approx 0.69 - 0.70$**.  
    Optimální rozhodovací strom na stejných datech dosáhl **$R^2 \approx 0.793 - 0.796$** (pokles RMSE o více než 35 000 USD).
    
    Čím to je, když jsme stromu nedodali žádné speciální polynomiální transformace?
    """)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        #### 1. Geografické shluky (`lat` & `long`)
        - **Lineární regrese:** Předpokládá, že cena se souřadnicemi roste přísně lineárně (např. *„čím severněji, tím vždy dražší“*).
        - **Rozhodovací strom:** Dokáže pravoúhlými řezy ohraničit konkrétní **bohaté enklávy** (např. poloostrov Bellevue, Medina, břehy Lake Washington), aniž by zvedal cenu v sousedních méně žádaných zónách.
        """)
    with col_a2:
        st.markdown("""
        #### 2. Skokový vliv kvality stavby (`grade`)
        - **Lineární regrese:** Přidává za každý stupeň `grade` konstantní částku.
        - **Rozhodovací strom:** Zachycuje nelineární zlom: rozdíl mezi grade 6 a 7 je běžný dům, ale přechod z grade 10 na 11 a 12 znamená zakázkové architektonické sídlo, kde cena roste exponenciálně.
        """)

    st.info(
        "💡 **Závěr:** Rozhodovací strom exceluje všude tam, kde data obsahují **geografické, socioekonomické nebo administrativní zóny a skokové tarify**."
    )

# =============================================================================
# TAB 2: ANATOMIE PŘEUČENÍ
# =============================================================================
with tab_overfit:
    st.markdown("### 🔬 Co se děje uvnitř stromu při přeučení?")
    st.markdown(r"""
    Podívejme se na dramatický rozdíl mezi výchozím stromem (Model 1) a prořezaným optimem (Model 3):
    """)

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        with st.container(border=True):
            st.markdown("#### ❌ Model 1: Neomezený strom")
            st.write("- **Hloubka:** 35 pater")
            st.write("- **Počet koncových listů:** 16 683 listů")
            st.write("- **Průměrný počet domů v listu:** ~1.04 domu!")
            st.write("- **Trénovací R²:** **1.0000** (dokonalá paměť)")
            st.write("- **Testovací R²:** **0.7048** (propad o 30 %)")
            st.error("Každý list reprezentuje jediný konkrétní prodaný dům. Model si zapamatoval i náhodné chyby a emoce kupujících.")

    with col_o2:
        with st.container(border=True):
            st.markdown("#### ✅ Model 3: Optimální strom")
            st.write("- **Hloubka:** 14 pater")
            st.write("- **Počet koncových listů:** 1 240 listů")
            st.write("- **Průměrný počet domů v listu:** ~14 domů")
            st.write("- **Trénovací R²:** **0.8889**")
            st.write("- **Testovací R²:** **0.7932** (rozdíl jen 9.5 %)")
            st.success("List reprezentuje statistický průměr mikro-lokality. Odolný vůči šumu.")

# =============================================================================
# TAB 3: DIAGNOSTIKA REZIDUÍ A STROP NA LUXUSU
# =============================================================================
with tab_resid:
    st.markdown("### 📉 Kde rozhodovací strom selhává? Past na luxusních vilách")
    st.markdown(r"""
    Zásadní slabinou stromů je, že **nikdy nedokážou předpovědět hodnotu vyšší než nejvyšší průměr v trénovacím listu**.
    
    Pokud se v testovacích datech objeví unikátní sídlo s cenou nad 4 000 000 USD, strom pro něj nemá žádnou speciální větev:
    - Zařadí ho do nejvyššího dostupného listu (kde je průměr např. 2 850 000 USD).
    - Výsledkem je **obrovská záporná chyba podhodnocení (Under-prediction bias)** u nejdražších nemovitostí.
    """)

    st.warning(
        "⚠️ **Doporučení pro produkci:** "
        "Pro odhady nemovitostí v luxusním segmentu (> 2 mil. USD) se v bankovní praxi nepoužívá čistý rozhodovací strom, "
        "ale buď **Log-transformovaný Gradient Boosting** nebo kombinace s OLS modelem (Stacking)."
    )

# =============================================================================
# TAB 4: COST-COMPLEXITY PRUNING
# =============================================================================
with tab_pruning:
    st.markdown("### ✂️ Teoretický doplněk: Prořezávání podle složitosti (ccp_alpha)")
    st.markdown(r"""
    Kromě omezení `max_depth` a `min_samples_leaf` nabízí Scikit-learn matematicky nejelegantnější metodu regulace:  
    **Minimal Cost-Complexity Pruning**.
    
    Definuje účelovou funkci penalizující počet listů $|T|$:
    $$R_\alpha(T) = R(T) + \alpha |T|$$
    kde:
    - $R(T)$ je celková chyba trénovacích dat (MSE součet),
    - $|T|$ je počet koncových listů (složitost stromu),
    - $\alpha \ge 0$ je penalizační koeficient složitosti (`ccp_alpha`).
    
    Při $\alpha = 0$ roste neomezený strom. S rostoucím $\alpha$ algoritmus automaticky odřezává větve, které nepřinášejí dostatečně velký pokles chyby, až po nejjednodušší strom.
    """)
