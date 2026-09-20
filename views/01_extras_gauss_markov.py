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
    V klasickém lineárním modelu je předpoklad **normality chyb $\varepsilon \sim \mathcal{N}(0, \sigma^2)$** klíčový pro **statistické testy hypotéz** ($t$-testy významnosti koeficientů, $F$-testy spolehlivosti modelu a konfidenční intervaly).  
    *(Pozor na častý omyl: Pro samotný Gauss-Markovův teorém a vlastnost BLUE – nejlepší lineární nestranný odhad – normalita chyb **není vyžadována**, postačují sférické chyby, exogenita a linearita!)*  
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
        r"⚠️ **Závěr diagnostiky:** Rozdělení vykazuje **výrazný pravý chvost (Right Skewness)**. "
        r"U běžných domů je chyba soustředěna kolem nuly, ale u luxusních vil model podhodnocuje o více než 1 milion USD! "
        r"*(Metodické upřesnění: Samotný histogram reziduí nedokazuje heteroskedasticitu – k jejímu průkazu slouží bodový graf reziduí vůči predikované ceně $\hat{y}$, kde se objevuje rozšiřující se trychtýř rozptylu chyb).* "
        r"Proto se v praxi pro stabilizaci rozptylu často modeluje logaritmus ceny $\log(\text{price})$."
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
        "🧠 **Proč `sqft_living`, `sqft_above` a `sqft_basement` vykazují extrémní VIF?**  \n"
        "Hodnota **999** v grafu je **vizualizační strop (cap)**. Všech 21 613 nemovitostí v King County splňuje přesnou matematickou identitu:  \n"
        "$$\\text{sqft\\_living} = \\text{sqft\\_above} + \\text{sqft\\_basement}$$  \n"
        "Při společném zahrnutí všech tří proměnných do OLS je matice $\\mathbf{X}^T \\mathbf{X}$ singulární a teoretická hodnota je $\\text{VIF} = \\infty$ (dokonalá multikolinearita). "
        "Tento problém se v praxi řeší buď vynecháním součtového sloupce, nebo nasazením **Ridge regularizace (L2)**."
    )

# =============================================================================
# TAB 3: CHECKLIST
# =============================================================================
with tab3:
    st.markdown("### 🛡️ Teoretický rámec: Gauss-Markovův teorém (BLUE) vs. Statistická inference")
    st.markdown(r"""
    Podle **Gauss-Markovova teorému** je odhad metodou nejmenších čtverců (OLS) $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ 
    **nejlepším lineárním nestranným odhadem (BLUE – Best Linear Unbiased Estimator)**, pokud jsou splněny následující 4 podmínky:

    | Podmínka Gauss-Markov (BLUE) | Matematická formulace | Co znamená v praxi | Jak se projevila na našich datech |
    | :--- | :--- | :--- | :--- |
    | **1. Linearita v parametrech** | $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ | Model je lineární kombinací vah $\beta_j$ | U diamantů selhala bez transformace (cena roste kubicky) |
    | **2. Striktní exogenita** | $\mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}$ | Rezidua nemají systematický vztah k $X$ | Opomenutí lokality v King County vedlo k systematickým chybám |
    | **3. Plná sloupcová hodnost** | $\text{rank}(\mathbf{X}) = p$ (žádná perfektní multikolinearita) | Žádný sloupec není přesnou kopií/kombinací jiného | `sqft_living` $\approx$ `sqft_above` + `basement` (vysoký VIF $\to$ Ridge) |
    | **4. Sférická rezidua (Homoskedasticita & Nekorelovanost)** | $\text{Var}(\boldsymbol{\varepsilon} \mid \mathbf{X}) = \sigma^2 \mathbf{I}$ | Konstantní rozptyl chyb a nulová autokorelace $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ | Porušena: u luxusních domů rozptyl chyb prudce roste (trychtýř) |

    ---

    #### ⚠️ Důležité upřesnění k normalitě chyb:
    - **Normalita chyb $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ NENÍ podmínkou Gauss-Markovova teorému!**  
      I bez normality je OLS stále **BLUE** (má nejmenší rozptyl mezi všemi lineárními nestrannými odhady).
    - **Kdy je normalita nutná?**  
      Normalita je nezbytná pro **statistickou inferenci na konečných vzorcích**: výpočet $p$-hodnot v $t$-testech koeficientů, $F$-testech celého modelu a pro přesné konfidenční intervaly spolehlivosti. (U obřích vzorků díky Centrální limitní větě – CLT – konvergují odhady k normalitě asymptoticky).
    """)

