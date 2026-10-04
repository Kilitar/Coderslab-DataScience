"""
Den 4: NLP – Cvičení TF-IDF: Expertní analýza & Kritika řešení
==============================================================
Téma: Log-Loss vs. Hinge Loss (Logistická regrese vs. SVM) v textovém prostoru
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Log-Loss vs. Hinge Loss (LR vs. SVM v NLP)")
st.caption(
    "Proč jsou lineární klasifikátory v kombinaci s TF-IDF tak efektivní? "
    "Hloubkový rozbor rozdílů mezi Logistickou regresí a Support Vector Machine a stav SOTA v říjnu 2026."
)

tab1, tab2, tab3 = st.tabs([
    "📐 1. Log-Loss vs. Hinge Loss (Matematika)",
    "⚔️ 2. Kdy zvolit LR a kdy SVM pro text",
    "🚀 3. Evoluce reprezentace textu do roku 2026"
])

with tab1:
    st.subheader("1. Matematické srovnání chybových funkcí: Log-Loss vs. Hinge Loss")
    st.markdown(
        r"""
        V obou případech hledáme lineární dělící nadrovinu tvaru $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$. 
        Zásadní rozdíl je však v tom, **jakou chybovou funkci minimalizujeme během trénování**:
        """
    )

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("##### 🔵 Logistická regrese: Log-Loss (Cross-Entropy)")
        st.latex(r"""L_{\text{LR}}(y, \hat{y}) = - \left[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \right]""")
        st.markdown(
            r"""
            * **Pravděpodobnostní pohled:** Optimalizuje věrohodnost (Maximum Likelihood Estimation).
            * Každý datový bod má vliv na gradient vah, i když je správně klasifikován daleko od hranice.
            * Výstupem je přirozeně **kalibrovaná pravděpodobnost** $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$.
            """
        )

    with col_l2:
        st.markdown("##### 🟣 Support Vector Machine: Hinge Loss & Margin")
        st.latex(r"""L_{\text{SVM}}(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x}))""")
        st.markdown(
            r"""
            * **Geometrický pohled:** Hledá dělící nadrovinu s **maximální vzdáleností (margin)** od nejbližších bodů (podpůrných vektorů).
            * Pokud je bod správně klasifikován a leží za hranicí ($y \cdot f(\mathbf{x}) \ge 1$), má **přesně nulovou ztrátu** a model ho ignoruje.
            * Výstupem je geometrická vzdálenost od nadroviny (decision function), nikoliv přímá pravděpodobnost.
            """
        )

    st.markdown("---")
    st.markdown("#### Proč jsou SVM tak populární u textových dat?")
    st.markdown(
        r"""
        Textová data po TF-IDF mají specifické geometrické vlastnosti:
        1. **Vysoká dimenzionalita ($D > 10\,000$):** V takto vysoké dimenzi jsou data téměř vždy **lineárně separabilní**.
        2. **Není potřeba nelineární jádro (RBF):** V textu lineární SVM (`LinearSVC`) funguje lépe než pomalé RBF jádro a trénování trvá pouhé zlomky sekundy ($0.04\text{ s}$).
        3. **Robustnost vůči odlehlým slovům:** Díky Hinge Loss body daleko od rozhodovací hranice neovlivňují sklon nadroviny.
        """
    )

with tab2:
    st.subheader("2. Rozhodovací strom v praxi: Kdy zvolit Logistickou regresi a kdy SVM?")
    st.markdown(
        """
        Při nasazení klasifikátoru textu v reálné byznys produkci platí následující doporučení:
        """
    )

    col_dec1, col_dec2 = st.columns(2)
    with col_dec1:
        st.success("🟢 Kdy zvolit Logistickou regresi:")
        st.markdown(
            r"""
            * **Potřebujete kalibrovanou pravděpodobnost:** Např. v automatickém schvalování nebo SPAM filtru, kde chcete akci spustit až při jistotě nad 98 %.
            * **Požadavek na vysvětlitelnost (Explainability):** Koeficienty $\beta$ odpovídají poměru šancí (Odds Ratios), což je ideální pro manažerské audity.
            * **Snadný online learning:** SGD trénování na streamovaných datech v reálném čase.
            """
        )

    with col_dec2:
        st.info("🟣 Kdy zvolit Support Vector Machine (LinearSVC):")
        st.markdown(
            r"""
            * **Cílem je pouze binární štítek (True/False):** Pokud vás nezajímá pravděpodobnost, ale maximální geometrická stabilita oddělení.
            * **Velmi malý trénovací dataset s mnoha příznaky:** SVM je méně náchylné k přetrénování na malých vzorcích s velkým slovníkem.
            * **Extrémní rychlost inference:** Výpočet skalárního součinu $\mathbf{w}^T \mathbf{x}$ bez nutnosti počítat sigmoidální funkci.
            """
        )

with tab3:
    st.subheader("3. Srovnávací přehled reprezentací textu (1970–2026)")
    st.markdown(
        """
        Jak se vyvíjela přesnost na úlohách klasifikace sentimentu IMDb v průběhu dekád?
        """
    )

    models_data = {
        "Metoda reprezentace": [
            "1. Bag of Words (CountVectorizer)",
            "2. TF-IDF (TfidfVectorizer)",
            "3. TF-IDF + N-gramy (1, 2)",
            "4. Word2Vec / GloVe (Averaged)",
            "5. Sentence-BERT (Bi-Encoder)",
            "6. Moderní LLM (LLaMA 3.3 / GPT-4o)"
        ],
        "Klasifikátor": [
            "Logistic Regression",
            "Logistic Regression / LinearSVC",
            "LinearSVC",
            "MLP / SVM",
            "Cosine Classifier",
            "Zero-shot / Few-shot prompt"
        ],
        "Délka trénování": [
            "~0.3 s",
            "~0.1 s",
            "~1.5 s",
            "~2 min",
            "~5 min",
            "Inference only"
        ],
        "Dosažená přesnost": [
            "85.75 %",
            "87.30 %",
            "89.10 %",
            "88.20 %",
            "93.40 %",
            "> 96.50 %"
        ],
        "Paměťová náročnost": [
            "Nízká (Sparse)",
            "Nízká (Sparse)",
            "Střední (Sparse)",
            "Střední (Dense)",
            "Vysoká (GPU)",
            "Cloud API / High VRAM"
        ]
    }
    st.dataframe(pd.DataFrame(models_data), hide_index=True, width="stretch")

    st.markdown(
        """
        > [!TIP]
        > **Závěr pro datového inženýra:**  
        > V roce 2026 je **TF-IDF + LinearSVC / Logistic Regression** stále absolutně nejlepším nástrojem pro **extrémně levný, okamžitý baseline** (stovky mikrosekund latence), 
        > zatímco transformery a LLM jsou voleny tam, kde je kritický hluboký sémantický kontext a porozumění složitým nuancím.
        """
    )
