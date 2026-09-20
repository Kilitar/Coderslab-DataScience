from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
theory_path = base_dir / "00_Prework" / "theory" / "01_ml_foundations.md"

st.title("🚀 Prework 01: Co je Data Science & Machine Learning")
st.caption("Před-kurzovní teoretický modul: Přechod od Software 1.0 k 2.0, 4 paradigmata strojového učení, anatomie modelu a moderní kontext 2026.")

tab1, tab2, tab3 = st.tabs([
    "📖 Teoretický rozbor",
    "🕹️ Interaktivní přehled: 4 paradigmata ML",
    "⚡ Klasické ML vs. GenAI (2026)"
])

# -----------------------------------------------------------------------------
# TAB 1: TEORETICKÝ TEXT
# -----------------------------------------------------------------------------
with tab1:
    if theory_path.exists():
        with open(theory_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Soubor 01_ml_foundations.md nebyl nalezen.")

# -----------------------------------------------------------------------------
# TAB 2: INTERAKTIVNÍ PŘEHLED PARADIGMAT
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🕹️ Interaktivní vizualizace: Typy úloh ve strojovém učení")
    st.caption("Zvolte kategorii a prozkoumejte, jak algoritmus zpracovává data a jaké otázky řeší.")

    ml_type = st.radio(
        "Vyberte paradigma:",
        options=[
            "1. Učení s učitelem: Regrese (Předpověď spojité hodnoty)",
            "2. Učení s učitelem: Klasifikace (Předpověď diskrétní třídy)",
            "3. Učení bez učitele: Shlukování (Hledání skrytých vzorů)",
            "4. Zpětnovazební učení: Reinforcement Learning (Agent & Odměna)"
        ],
        horizontal=True
    )

    np.random.seed(42)

    if "Regrese" in ml_type:
        st.markdown("#### 📈 Regrese: Odhad ceny nemovitosti podle rozlohy")
        st.markdown(r"""
        - **Cíl:** Předpovědět spojité číslo $y \in \mathbb{R}$.
        - **Model:** Hledá spojitou křivku (přímku, parabolu), která minimalizuje odchylky od skutečných cen.
        """)
        x_pts = np.linspace(30, 150, 60)
        y_pts = 35000 * x_pts + 500000 + np.random.normal(0, 300000, 60)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode="markers", name="Historické prodeje", marker=dict(size=9, color="#38BDF8", line=dict(color="#0284C7", width=1))))
        # Regresní přímka
        m, b = np.polyfit(x_pts, y_pts, 1)
        fig.add_trace(go.Scatter(x=x_pts, y=m*x_pts + b, mode="lines", name="Regresní model f(X)", line=dict(color="#E63946", width=3)))

        fig.update_layout(
            xaxis_title="Podlahová plocha (m²)",
            yaxis_title="Cena nemovitosti (Kč)",
            height=420
        )
        st.plotly_chart(fig, width="stretch")

    elif "Klasifikace" in ml_type:
        st.markdown("#### 🎯 Klasifikace: Schválení úvěru (Příjem vs. Zadlužení)")
        st.markdown(r"""
        - **Cíl:** Zařadit pozorování do jedné z diskrétních tříd $y \in \{0, 1\}$.
        - **Model:** Hledá rozhodovací hranici (Decision Boundary), která oddělí rizikové klienty od spolehlivých.
        """)
        n_pts = 50
        # Třída 0: Schváleno (vyšší příjem, nižší dluh)
        inc_0 = np.random.normal(65000, 12000, n_pts)
        debt_0 = np.random.normal(20000, 8000, n_pts)
        # Třída 1: Zamítnuto (nižší příjem, vyšší dluh)
        inc_1 = np.random.normal(35000, 10000, n_pts)
        debt_1 = np.random.normal(45000, 10000, n_pts)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inc_0, y=debt_0, mode="markers", name="Úvěr schválen", marker=dict(size=9, color="#2A9D8F", symbol="circle")))
        fig.add_trace(go.Scatter(x=inc_1, y=debt_1, mode="markers", name="Úvěr zamítnut", marker=dict(size=9, color="#E63946", symbol="x")))

        # Rozhodovací hranice
        b_x = np.linspace(20000, 85000, 50)
        b_y = 65000 - 0.5 * b_x
        fig.add_trace(go.Scatter(x=b_x, y=b_y, mode="lines", name="Rozhodovací hranice", line=dict(color="gray", dash="dash", width=2)))

        fig.update_layout(
            xaxis_title="Měsíční příjem žadatele (Kč)",
            yaxis_title="Stávající měsíční splátky dluhů (Kč)",
            height=420
        )
        st.plotly_chart(fig, width="stretch")

    elif "Shlukování" in ml_type:
        st.markdown("#### 🧩 Shlukování (Clustering): Segmentace zákazníků e-shopu")
        st.markdown(r"""
        - **Cíl:** Algoritmus **nemá žádné labely $y$**. Sám nachází přirozené shluky podobných zákazníků na základě nákupního chování.
        - **Příklady shluků:** Šetřiví příležitostní kupci, VIP zákazníci, lovci slev.
        """)
        c1_x = np.random.normal(20, 5, 40)
        c1_y = np.random.normal(15, 4, 40)
        c2_x = np.random.normal(70, 8, 40)
        c2_y = np.random.normal(80, 10, 40)
        c3_x = np.random.normal(25, 6, 40)
        c3_y = np.random.normal(75, 8, 40)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c1_x, y=c1_y, mode="markers", name="Shluk 1: Příležitostní kupci", marker=dict(size=9, color="#38BDF8")))
        fig.add_trace(go.Scatter(x=c2_x, y=c2_y, mode="markers", name="Shluk 2: VIP / Prémioví", marker=dict(size=9, color="#E76F51")))
        fig.add_trace(go.Scatter(x=c3_x, y=c3_y, mode="markers", name="Shluk 3: Lovci výprodejů", marker=dict(size=9, color="#F4A261")))

        fig.update_layout(
            xaxis_title="Frekvence nákupů za rok",
            yaxis_title="Průměrná hodnota objednávky (USD)",
            height=420
        )
        st.plotly_chart(fig, width="stretch")

    else:
        st.markdown("#### 🤖 Zpětnovazební učení (Reinforcement Learning)")
        st.markdown(r"""
        V RL se model neučí na statické tabulce dat, ale **v interaktivní smyčce (Trial & Error)**:
        
        $$\text{Stav } s_t \xrightarrow{\text{Agent provede Akci } a_t} \text{Nové prostředí } s_{t+1} + \text{Odměna } r_{t+1}$$

        - **Agent:** Algoritmus dělající rozhodnutí (autonomní vůz, herní bot, ladicí modul LLM).
        - **Prostředí:** Virtuální nebo fyzický svět.
        - **Odměna (Reward):** Číslo, které říká, jak dobrá byla akce (např. dojetí do cíle bez nehody = +100 bodů, náraz = -1000 bodů).
        - **RLHF (Moderní souvislost):** Metoda, kterou OpenAI a Google ladí modely GPT a Gemini, aby odpovídaly lidským preferencím (lidští hodnotitelé dávají odměnu za kvalitní a bezpečnou odpověď).
        """)

# -----------------------------------------------------------------------------
# TAB 3: KLASICKÉ ML VS GENAI (2026)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⚡ Srovnání: Klasické ML vs. Generativní AI & LLMs (2026)")
    st.markdown(r"""
    Proč je i v éře ChatGPT a Gemini naprosto klíčové ovládat **klasické Machine Learning algoritmy** (Scikit-learn, XGBoost, regresi)?
    """)

    comp_df = pd.DataFrame([
        {
            "Dimenze srovnání": "Dominantní typ dat",
            "Klasické ML (Scikit-learn / XGBoost)": "Tabulková data (čísla, kategorie, transakce, senzory)",
            "Generativní AI / LLMs (PyTorch, Transformers)": "Nestrukturovaná data (přirozený jazyk, kód, obrázky, audio)"
        },
        {
            "Dimenze srovnání": "Výpočetní nároky & Cena",
            "Klasické ML (Scikit-learn / XGBoost)": "Extrémně nízké (běží na CPU za milisekundy, centy na milion predikcí)",
            "Generativní AI / LLMs (PyTorch, Transformers)": "Vysoké (vyžaduje GPU/TPU klastry, řádově vyšší cena za dotaz)"
        },
        {
            "Dimenze srovnání": "Deterministická přesnost v číslech",
            "Klasické ML (Scikit-learn / XGBoost)": "100% matematická přesnost, striktní optimalizace ztrátové funkce",
            "Generativní AI / LLMs (PyTorch, Transformers)": "Pravděpodobnostní generování textu, riziko numerických halucinací"
        },
        {
            "Dimenze srovnání": "Vysvětlitelnost (Explainability)",
            "Klasické ML (Scikit-learn / XGBoost)": "Vysoká (koeficienty β, důležitost příznaků, SHAP hodnoty)",
            "Generativní AI / LLMs (PyTorch, Transformers)": "Černá skříňka (miliardy až biliony parametrů)"
        },
        {
            "Dimenze srovnání": "Regulatorní soulad (EU AI Act)",
            "Klasické ML (Scikit-learn / XGBoost)": "Snadno auditovatelné, splňuje přísná kritéria pro finance a zdravotnictví",
            "Generativní AI / LLMs (PyTorch, Transformers)": "Komplexní governance, nutnost ochrany proti jailbreaku a úniku dat"
        }
    ])

    st.dataframe(comp_df, width="stretch", hide_index=True)

    st.info(
        "💡 **Zlaté pravidlo architekta v roce 2026:** "
        "Pokud řešíte tabulku s miliony transakcí, predikci cen, fraud detection nebo churn – sáhněte po **XGBoost, LightGBM nebo Scikit-learn**. "
        "Pokud řešíte sumarizaci textu, chatbota nebo generování kódu – sáhněte po **LLM**. "
        "Nejvýkonnější systémy v praxi obě architektury kombinují!"
    )
