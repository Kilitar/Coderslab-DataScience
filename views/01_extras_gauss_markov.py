from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Diagnostika 4 předpokladů OLS")
st.caption("Gauss-Markovovy teorémy v praxi: Proč lineární regrese v King County naráží na limity normality, homoskedasticity a multikolinearity.")

@st.cache_data
def load_diag_data():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "01_Regression" / "data" / "day1_extras_precomputed.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

diag = load_diag_data()

tab1, tab2, tab3 = st.tabs([
    "📊 1. Normalita reziduí & Trychtýř (Homoskedasticita)",
    "⚡ 2. Multikolinearita & VIF faktor",
    "🛡️ 3. Gauss-Markov Checklist"
])

# =============================================================================
# TAB 1: NORMALITA A HETEROSKEDASTICITA
# =============================================================================
with tab1:
    st.markdown("### 📊 Rozdělení reziduí: Gaussovský zvon vs. Těžký pravý chvost")
    st.markdown(r"""
    Teoretický předpoklad lineární regrese zní: **Rezidua musí mít normální rozdělení $e \sim \mathcal{N}(0, \sigma^2)$**.  
    Zde je skutečný histogram reziduí OLS na testovacích datech King County:
    """)

    hist_data = diag["res_hist"]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=hist_data["bin_centers"],
        y=hist_data["counts"],
        name="Skutečná rezidua OLS",
        marker_color="#3B82F6"
    ))
    fig_hist.update_layout(
        title="Histogram reziduí (Chyba predikce = Skutečná cena - Predikce OLS)",
        xaxis_title="Reziduum v USD (Záporné = nadhodnoceno, Kladné = podhodnoceno)",
        yaxis_title="Počet nemovitostí",
        height=450
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.warning(
        r"⚠️ **Závěr diagnostiky:** Rozdělení má **výrazný pravý chvost (Right Skewness)**. "
        r"U běžných domů je chyba soustředěna kolem 0, ale u luxusních vil model chybuje o více než 1 milion USD! "
        r"Dochází k **heteroskedasticitě (rozšiřující se trychtýř chyb)**, což je důvod, proč se v praxi často používá logaritmická transformace ceny $\log(\text{price})$."
    )

# =============================================================================
# TAB 2: VIF MULTIKOLINEARITA
# =============================================================================
with tab2:
    st.markdown("### ⚡ Multikolinearita v King County: Variance Inflation Factor (VIF)")
    st.markdown(r"""
    **VIF (Variance Inflation Factor)** měří, jak moc je rozptyl odhadovaného koeficientu $\beta_j$ nafouknut 
    kvůli korelaci s ostatními prediktory:  
    $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
    - $\text{VIF} < 5$: Bezpečný prediktor
    - $5 \le \text{VIF} < 10$: Zvýšená korelace, doporučena obezřetnost
    - $\text{VIF} \ge 10$: **Závažná multikolinearita**, koeficienty v OLS jsou nestabilní!
    """)

    df_vif = pd.DataFrame(diag["vif_data"]).sort_values(by="vif", ascending=False)
    
    fig_vif = go.Figure(go.Bar(
        x=df_vif["vif"],
        y=df_vif["feature"],
        orientation="h",
        marker=dict(
            color=df_vif["vif"].apply(lambda v: "#EF4444" if v >= 10 else ("#F59E0B" if v >= 5 else "#10B981")),
            line=dict(color="#1E293B", width=1)
        )
    ))
    fig_vif.add_vline(x=5, line_dash="dash", line_color="#F59E0B", annotation_text="Pozor (VIF=5)")
    fig_vif.add_vline(x=10, line_dash="dash", line_color="#EF4444", annotation_text="Kritická hranice (VIF=10)")
    fig_vif.update_layout(
        title="VIF hodnoty pro prediktory v King County",
        xaxis_title="Hodnota VIF faktoru",
        yaxis_title="Příznak nemovitosti",
        height=450
    )
    st.plotly_chart(fig_vif, width="stretch")

    st.info(
        "🧠 **Proč `sqft_living` a `sqft_above` vykazují vysoký VIF?**  \n"
        "Protože `sqft_living` je v podstatě součtem `sqft_above` (nadzemní plocha) + `sqft_basement` (suterén)! "
        "Je to dokonalý učebnicový příklad multikolinearity, který vyřešila až **Ridge regularizace (L2)** v Cvičení 5."
    )

# =============================================================================
# TAB 3: CHECKLIST
# =============================================================================
with tab3:
    st.markdown("### 🛡️ Rychlý checklist pro datového analytika: 4 předpoklady OLS")
    st.markdown(r"""
    | Předpoklad | Co znamená v teorii | Jak se projevil na našich datech | Řešení v praxi |
    | :--- | :--- | :--- | :--- |
    | **1. Linearita parametrů** | Vztah mezi $X$ a $Y$ lze popsat lineární kombinací | U diamantů selhala (cena roste kubicky s karáty) | Polynomiální regrese, Decision Tree |
    | **2. Homoskedasticita** | Rozptyl chyb je konstantní $\text{Var}(\varepsilon) = \sigma^2$ | Porušena: u luxusních domů rozptyl chyb prudce roste | Logaritmická transformace $\log(y)$, WLS |
    | **3. Žádná multikolinearita** | Prediktory nejsou lineární kombinací jiných | Porušena: `sqft_living` = `sqft_above` + `basement` | Ridge (L2) regularizace, odstranění sloupce |
    | **4. Normalita chyb** | Rezidua pocházejí z normálního rozdělení | Pravostranně zešikmeno kvůli vilám u vody | Robustní regrese (Huber), stromy |
    """)
