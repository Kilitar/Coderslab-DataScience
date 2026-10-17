"""
Den 5: Neřízené učení – Teorie detekce anomálií & Isolation Forest (iForest)
=============================================================================
Detailní matematický a algoritmický rozbor metody Isolation Forest:
1. Změna paradigmatu: Proč izolovat přímo namísto profilování normality.
2. Matematické odvození: Průměrná hloubka E(h(x)), normalizační BST konstanta c(n) a skóre s(x, n).
3. Interaktivní kalkulátor skóre anomálie na základě hloubky a velikosti vzorku.
4. Srovnávací studie: Isolation Forest vs. Local Outlier Factor (LOF) vs. One-Class SVM.
5. Produkční osvědčené postupy: Subsampling (psi=256), invariance vůči škálování a Extended Isolation Forest.
"""

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Teorie: Isolation Forest (iForest)", page_icon="📖", layout="wide")

st.title("📖 Teorie: Detekce anomálií & Isolation Forest (iForest)")
st.caption(
    "Pokročilý teoretický rozbor algoritmu Isolation Forest (Liu, Ting, Zhou, 2008): "
    "Konstrukce náhodných izolačních stromů, odvození normalizační konstanty c(n) přes harmonická čísla, "
    "interpretace anomálního skóre a srovnání s hustotními metodami."
)

tab_arch, tab_math, tab_calc, tab_comp, tab_prod = st.tabs([
    "🌲 Princip stromové izolace",
    "📐 Matematické odvození",
    "🧮 Kalkulátor skóre anomálie",
    "⚖️ iForest vs. LOF vs. OC-SVM",
    "🛡️ Produkční osvědčené postupy"
])

# -----------------------------------------------------------------------------
# TAB 1: Princip stromové izolace
# -----------------------------------------------------------------------------
with tab_arch:
    st.subheader("🌲 Změna paradigmatu: Přímá izolace namísto profilování normality")
    st.markdown(
        """
        Tradiční statistické a strojové metody přistupují k detekci odlehlých pozorování tak, že se nejprve 
        snaží definovat, co je **„normální“** (těžiště shluků, hustota pravděpodobnosti, kovarianční matice). 
        To je výpočetně náročné a v praxi náchylné k přeučení.
        
        **Isolation Forest (iForest)** přichází s revolučním konceptem:
        - Outliers jsou **vzácné** (v menšině).
        - Outliers mají **výrazně odlišné hodnoty atributů** (leží daleko od hustých center).
        """
    )

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        with st.container(border=True):
            st.markdown("### 🔴 Anomálie (Outliers)")
            st.markdown(
                """
                Leží v řídce osídlených oblastech. Náhodný řez provedený v prostoru znaků má vysokou pravděpodobnost, 
                že anomálii oddělí od zbytku dat téměř okamžitě. 
                
                **Výsledek:** Velmi malá hloubka stromu (krátká cesta $h(x)$ od kořene).
                """
            )
    with col_box2:
        with st.container(border=True):
            st.markdown("### 🔵 Běžná pozorování (Inliers)")
            st.markdown(
                """
                Jsou nahloučena v hustých klastrech. Aby náhodný řez izoloval jediný bod uvnitř shluku, 
                musí strom provést velké množství postupných jemných dělení. 
                
                **Výsledek:** Velká hloubka stromu (dlouhá cesta $h(x)$ od kořene).
                """
            )


    st.markdown("---")
    st.markdown("#### Schéma rekurzivního dělení prostoru:")

    # Vizuální demonstrace náhodných řezů v 2D
    np.random.seed(42)
    inliers_x = np.random.normal(loc=5, scale=1.0, size=150)
    inliers_y = np.random.normal(loc=5, scale=1.0, size=150)
    outlier_x = [1.2, 9.5]
    outlier_y = [8.8, 1.5]

    fig_cuts = go.Figure()
    fig_cuts.add_trace(go.Scatter(x=inliers_x, y=inliers_y, mode="markers", marker=dict(color="#3b82f6", size=7, opacity=0.6), name="Běžné body (Inliers)"))
    fig_cuts.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode="markers", marker=dict(color="#ef4444", size=12, line=dict(width=1.5, color="black")), name="Anomálie (Outliers)"))

    # Náhodné řezy izolující anomálii na [1.2, 8.8]
    fig_cuts.add_shape(type="line", x0=2.5, y0=0, x1=2.5, y1=10, line=dict(color="#dc2626", width=2.5, dash="dash"))
    fig_cuts.add_annotation(x=2.5, y=9.5, text="1. řez: x₁ < 2.5 (Anomálie izolována v hloubce 1!)", showarrow=True, arrowhead=2, arrowcolor="#dc2626", font=dict(color="#dc2626", size=11))

    fig_cuts.update_layout(
        title="Koncept náhodných řezů v 2D rovině (Anomálie je oddělena hned prvním řezem)",
        xaxis_title="Příznak x₁",
        yaxis_title="Příznak x₂",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_cuts, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: Matematické odvození
# -----------------------------------------------------------------------------
with tab_math:
    st.subheader("📐 Matematické odvození skóre anomálií")
    st.markdown(
        r"""
        V původním článku *Isolation Forest* (Liu, Ting, Zhou, 2008) autoři propojili strukturu náhodného izolačního stromu 
        s teoretickými vlastnostmi **binárních vyhledávacích stromů (Binary Search Trees - BST)**.
        
        ### 1. Průměrná hloubka neúspěšného hledání v BST: $c(n)$
        Pro vzorek o velikosti $n$ je průměrná hloubka neúspěšného vyhledávání v náhodně zkonstruovaném BST dána vzorcem:
        $$c(n) = 2 H(n-1) - \frac{2(n-1)}{n}$$
        kde $H(i)$ je $i$-té harmonické číslo:
        $$H(i) = \sum_{k=1}^i \frac{1}{k} \approx \ln(i) + \gamma$$
        Konstanta $\gamma \approx 0.5772156649$ je **Eulerova-Mascheroniho konstanta**.
        
        Hodnota $c(n)$ představuje **očekávanou hloubku stromu** pro typické pozorování v datasetu o velikosti $n$.
        
        ### 2. Anomální skóre: $s(x, n)$
        Skóre anomálie pro bod $x$ v populaci o velikosti $n$ při průměrné naměřené hloubce $E(h(x))$ napříč všemi stromy lesa je:
        $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
        
        ### 3. Rigorózní interpretace chování funkce $s(x, n)$:
        - **Pokud $E(h(x)) \to 0$:**
          $$s(x, n) \to 2^0 = 1.0$$
          Bod byl v průměru izolován v extrémně mělké hloubce. **Jednoznačná anomálie**.
        - **Pokud $E(h(x)) \to c(n)$:**
          $$s(x, n) \to 2^{-1} = 0.5$$
          Bod má průměrnou hloubku přesně rovnou teoretickému očekávání náhodného BST. **Typické pozorování**.
        - **Pokud $E(h(x)) \to n - 1$:**
          $$s(x, n) \to 2^{-\infty} \to 0.0$$
          Bod vyžadoval extrémní hloubku a leží v samém jádru nejhustšího shluku. **Naprosto bezpečný inlier**.
        """
    )

    st.markdown("---")
    # Křivka anomálního skóre v závislosti na poměru E(h(x)) / c(n)
    ratio = np.linspace(0, 3, 200)
    score = 2 ** (-ratio)

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(x=ratio, y=score, mode="lines", line=dict(color="#2563eb", width=3), name="Skóre s(x, n)"))
    fig_score.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="Normální bod (s = 0.5)", annotation_position="top right")
    fig_score.add_hline(y=0.6, line_dash="dot", line_color="#ef4444", annotation_text="Prahová zóna anomálií (s >= 0.6)", annotation_position="top right")

    fig_score.update_layout(
        title="Průběh funkce anomálního skóre s(x, n) = 2^(-E(h(x)) / c(n))",
        xaxis_title="Poměr hloubky k očekávané hodnotě: E(h(x)) / c(n)",
        yaxis_title="Skóre anomálie s(x, n)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_score, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: Kalkulátor skóre anomálie
# -----------------------------------------------------------------------------
with tab_calc:
    st.subheader("🧮 Interaktivní kalkulátor: Přepočet délky cesty na anomální skóre")
    st.markdown(
        """
        Vyzkoušejte si, jak se průměrná hloubka $E(h(x))$ promítá do oficiálního anomálního skóre $s(x, n)$ 
        dle původní práce Liu et al. (2008):
        """
    )

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        input_n = st.number_input(
            "Velikost podvzorku stromu (psi / max_samples)",
            min_value=4,
            max_value=10000,
            value=256,
            step=16,
            help="Standardní hodnota v Scikit-learn je psi = 256."
        )
    with c_col2:
        # Spočteme c(n)
        euler_gamma = 0.5772156649
        harmonic = math.log(input_n - 1) + euler_gamma
        cn = 2.0 * harmonic - (2.0 * (input_n - 1) / input_n)

        max_h = math.ceil(math.log2(input_n)) * 2
        input_eh = st.slider(
            "Průměrná hloubka izolace bodu E(h(x))",
            min_value=1.0,
            max_value=float(max_h),
            value=3.5,
            step=0.1,
            help="Průměrný počet hran od kořene k izolaci bodu napříč všemi stromy."
        )

    # Výpočet skóre
    calc_score = 2.0 ** (-input_eh / cn)

    k_res1, k_res2, k_res3 = st.columns(3)
    k_res1.metric("Teoretické c(n)", f"{cn:.3f}", help="Očekávaná hloubka pro typické pozorování v BST")
    k_res2.metric("Poměr E(h(x)) / c(n)", f"{input_eh / cn:.3f}")
    k_res3.metric("Skóre s(x, n)", f"{calc_score:.4f}", delta="Podezření na anomálii!" if calc_score >= 0.6 else "Běžný bod")

    if calc_score >= 0.65:
        st.error(f"🔴 **Jednoznačná anomálie!** Bod vyžaduje pouze {input_eh:.1f} řezů oproti očekávaným {cn:.1f}. Skóre {calc_score:.3f} je výrazně nad hranicí 0.6.")
    elif calc_score >= 0.5:
        st.warning(f"🟡 **Hraniční pozorování:** Skóre {calc_score:.3f} odpovídá mírně podprůměrné hloubce. Bod leží na periferii shluku.")
    else:
        st.success(f"🟢 **Typický Inlier:** Hloubka {input_eh:.1f} je větší než očekávaná hodnota {cn:.1f}. Skóre {calc_score:.3f} indikuje bezpečný bod uvnitř shluku.")

# -----------------------------------------------------------------------------
# TAB 4: iForest vs. LOF vs. OC-SVM
# -----------------------------------------------------------------------------
with tab_comp:
    st.subheader("⚖️ Komparativní analýza: Isolation Forest vs. LOF vs. One-Class SVM")
    st.markdown(
        """
        Kdy zvolit Isolation Forest a kdy dát přednost hustotním metodám jako Local Outlier Factor?
        """
    )

    comp_df = pd.DataFrame([
        {
            "Dimenze / Vlastnost": "Základní filozofie",
            "Isolation Forest": "Přímá stromová izolace vzácných bodů",
            "Local Outlier Factor (LOF)": "Poměr lokální dosažitelné hustoty k sousedům",
            "One-Class SVM": "Nalezení optimální oddělující nadroviny v RKHS"
        },
        {
            "Dimenze / Vlastnost": "Časová složitost trénování",
            "Isolation Forest": "O(t · ψ log ψ) — Lineární / Sub-lineární",
            "Local Outlier Factor (LOF)": "O(N² · D) — Kvadratická",
            "One-Class SVM": "O(N² až N³) — Kvadratická až kubická"
        },
        {
            "Dimenze / Vlastnost": "Náročnost na RAM / Paměť",
            "Isolation Forest": "Nízká (pouze uložení t stromů o hloubce ~8)",
            "Local Outlier Factor (LOF)": "Extrémní (matice vzdáleností pro N bodů)",
            "One-Class SVM": "Střední až vysoká (ukládání support vektorů)"
        },
        {
            "Dimenze / Vlastnost": "Citlivost na proměnlivou hustotu",
            "Isolation Forest": "Střední (preferuje globální okraje)",
            "Local Outlier Factor (LOF)": "Vynikající (exceluje při různé hustotě)",
            "One-Class SVM": "Nízká (jednotný kernel parametr gamma)"
        },
        {
            "Dimenze / Vlastnost": "Škálovatelnost pro N > 100 000",
            "Isolation Forest": "✅ Bez problémů zvládá miliony řádků",
            "Local Outlier Factor (LOF)": "❌ Kolaps paměti a výpočetního času",
            "One-Class SVM": "❌ Nepoužitelné bez Nyström aproximace"
        },
        {
            "Dimenze / Vlastnost": "Nutnost škálování (StandardScaler)",
            "Isolation Forest": "❌ Není nutné (invariance stromů)",
            "Local Outlier Factor (LOF)": "✅ Kriticky povinné (eukleidovská metrika)",
            "One-Class SVM": "✅ Kriticky povinné (RBF vzdálenost)"
        }
    ])
    st.dataframe(comp_df, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# TAB 5: Produkční osvědčené postupy
# -----------------------------------------------------------------------------
with tab_prod:
    st.subheader("🛡️ Produkční osvědčené postupy pro nasazení Isolation Forest (2026)")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        with st.container(border=True):
            st.markdown("### 1. Magie Subsamplingu (max_samples = 256)")
            st.markdown(
                """
                Mnoho inženýrů intuitivně nastavuje `max_samples=len(X)` ve víře, že více dat zlepší model. 
                **To je kardinální chyba!** Větší vzorek způsobuje tzv. *Swamping* (běžné body obklopí anomálie) 
                a *Masking* (anomálie vytvoří vlastní husté shluky). 
                Hodnota **ψ = 256** je matematicky optimální pro téměř jakkoliv velký dataset.
                """
            )
        with st.container(border=True):
            st.markdown("### 2. Rozhodovací funkce vs. Pevná kontaminace")
            st.markdown(
                """
                V produkčním API nevolte binární `fit_predict()` s fixní `contamination`. 
                Místo toho ukládejte spojité skóre z `decision_function()` a kalibrujte 
                rozhodovací práh dynamicky podle ekonomické nákladové matice (Cost Matrix).
                """
            )

    with p_col2:
        with st.container(border=True):
            st.markdown("### 3. Extended Isolation Forest (EIF)")
            st.markdown(
                """
                Standardní iForest štěpí prostor pouze rovnoběžně s osami (Axis-Aligned). 
                To vytváří pravoúhlé artefakty a falešné zóny anomálií u šikmých korelací. 
                Pro korelovaná data zvažte **Extended Isolation Forest**, který volí náhodné nadroviny 
                s náhodným směrovým vektorem.
                """
            )
        with st.container(border=True):
            st.markdown("### 4. Ochrana před dimenzionálním šumem")
            st.markdown(
                """
                Ačkoliv je iForest mnohem odolnější vůči dimenzionalitě než k-NN, pokud dataset obsahuje 
                stovky irelevantních šumových příznaků, náhodný výběr os způsobí degradaci stromů. 
                Doporučuje se omezit `max_features` nebo předem provést selekci znaků.
                """
            )

