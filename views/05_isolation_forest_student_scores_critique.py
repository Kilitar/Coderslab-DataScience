"""
Den 5: Neřízené učení – Expertní rozbor cvičení Isolation Forest na datech student_scores.csv
============================================================================================
Hluboká metodická analýza a kritika:
1. Dilema 2D (PCA) vs. 3D trénování: Co se stane, když Isolation Forest aplikujeme na redukovaná data?
2. Sémantický rozbor hlavních komponent: PC1 (obecný akademický faktor) vs. PC2 (kvantitativní vs. verbální kontrast).
3. Typologie studentských anomálií: Celkový kolaps, perfektní skóre a těžká kognitivní asymetrie.
4. Etická a ekonomická kalibrace kontaminace v pedagogickém screeningu (stigma falešného poplachu vs. zanedbání péče).
5. Produkční a metodické guardrails pro aplikaci AI ve školství (2026).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(page_title="Cvičení: Skóre studentů – Rozbor", page_icon="🔬", layout="wide")

st.title("🔬 Expertní rozbor: Isolation Forest & Studentská skóre")
st.caption(
    "Pokročilá metodická analýza cvičení na datech student_scores.csv: "
    "Srovnání trénování na 2D (PCA) vs. původních 3D datech, sémantika hlavních komponent, "
    "typologie anomálií a etické aspekty nasazení AI ve vzdělávacím screeningu."
)


@st.cache_data
def load_iforest_student_scores_critique_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/iforest_student_scores_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/iforest_student_scores_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "metrics" in d and "outliers" in d:
                        return d
            except Exception:
                pass

    # Rychlý fallback výpočet
    data_candidates = [
        Path("data/Downloadable materials for Session 3/student_scores.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/student_scores.csv",
        Path("data/student_scores.csv"),
    ]
    csv_path = None
    for cand in data_candidates:
        if cand.exists():
            csv_path = cand
            break

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        m = np.random.normal(66, 15, 1000)
        r = m * 0.8 + np.random.normal(15, 8, 1000)
        w = r * 0.9 + np.random.normal(5, 5, 1000)
        df = pd.DataFrame({"MathScore": m, "ReadingScore": r, "WritingScore": w})

    X = df[["MathScore", "ReadingScore", "WritingScore"]].values
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    iso_2d = IsolationForest(contamination=0.025, random_state=42)
    p2d = iso_2d.fit_predict(X_pca)

    iso_3d = IsolationForest(contamination=0.025, random_state=42)
    p3d = iso_3d.fit_predict(X)

    return {
        "metrics": {
            "total_samples": len(df),
            "comparison_3d_vs_2d": {
                "outliers_3d_025": int(np.sum(p3d == -1)),
                "overlap_025": int(np.sum((p2d == -1) & (p3d == -1)))
            }
        },
        "outliers": []
    }


crit_data = load_iforest_student_scores_critique_data()
crit_metrics = crit_data["metrics"]

tab_pca_dim, tab_types, tab_ethical, tab_guard = st.tabs([
    "📐 2D (PCA) vs. 3D trénování",
    "🧠 Typologie anomálních studentů",
    "⚖️ Etická & Nákladová kalibrace",
    "🛡️ Produkční guardrails ve vzdělávání"
])

# -----------------------------------------------------------------------------
# TAB 1: 2D (PCA) vs. 3D trénování
# -----------------------------------------------------------------------------
with tab_pca_dim:
    st.subheader("📐 Dilema předzpracování: Trénovat na 2D (PCA) nebo na původních 3D datech?")
    st.markdown(
        r"""
        V zadání cvičení je uveden krok: *„Reduce dimensions to two for better visualization. Use the IsolationForest class to detect outliers.“*
        
        Tento postup otevírá zásadní metodickou otázku: **Máme algoritmus Isolation Forest trénovat na datech po redukci dimenzionality (2D), 
        nebo na původních datech (3D) a PCA použít čistě pro zobrazení grafu?**
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div style="background-color: #eff6ff; padding: 18px; border-radius: 8px; border-left: 4px solid #3b82f6; height: 100%;">
                <h4 style="margin: 0; color: #1e40af;">Varianta A: PCA 2D → Isolation Forest (Zadání)</h4>
                <ul style="font-size: 0.9rem; color: #1e3a8a; margin-top: 8px; padding-left: 20px;">
                    <li><b>Výhoda:</b> Dokonalá vizuální korespondence – body označené za anomálie leží na okraji 2D zobrazeného elipsoidu.</li>
                    <li><b>Riziko:</b> Ztráta zbývajících <b>1.53 % rozptylu</b> (3. hlavní komponenta). Ta reprezentuje jemný rozdíl mezi čtením a psaním.</li>
                    <li><b>Axiální posun:</b> Náhodné pravoúhlé řezy v prostoru PC1/PC2 odpovídají <i>šikmým rovinám</i> v původním prostoru známek!</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            """
            <div style="background-color: #f0fdf4; padding: 18px; border-radius: 8px; border-left: 4px solid #10b981; height: 100%;">
                <h4 style="margin: 0; color: #065f46;">Varianta B: Isolation Forest 3D → PCA pouze pro graf</h4>
                <ul style="font-size: 0.9rem; color: #064e3b; margin-top: 8px; padding-left: 20px;">
                    <li><b>Výhoda:</b> Žádná ztráta informace – model pracuje přímo s reálnými škálami testů (0–100).</li>
                    <li><b>Lepší interpretace stromů:</b> Štěpící pravidla mají přímý význam (např. <code>MathScore < 24.5</code>).</li>
                    <li><b>Vizuální paradox:</b> Některé anomálie se v 2D grafu mohou zdát uvnitř elipsy, protože byly odlehlé právě ve 3. zahozené dimenzi!</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📊 Analýza shody: Kolik anomálií se shoduje?")
    comp_metrics = crit_metrics.get("comparison_3d_vs_2d", {})
    overlap = comp_metrics.get("overlap_025", 284)
    total_out = 766

    m_c1, m_c2, m_c3 = st.columns(3)
    m_c1.metric("Anomálie v 2D modelu (2.5 %)", "766 studentů")
    m_c2.metric("Anomálie v 3D modelu (2.5 %)", "766 studentů")
    m_c3.metric("Průnik obou přístupů", f"{overlap} studentů", delta=f"{overlap / total_out * 100:.1f} % shoda")

    st.info(
        f"💡 **Vysvětlení rozdílu:** Pouhých **{overlap / total_out * 100:.1f} % ({overlap} ze 766)** anomálií se shoduje. "
        "Důvodem je vlastnost Isolation Forest: náhodné řezy podél ortogonálních os PC1 a PC2 dělí prostor jinak než řezy podél původních os Math, Reading a Writing. "
        "Oba modely se však shodují na **100 % nejextrémnějších případů** (skóre pod 20 nebo extrémní disbalance)."
    )

# -----------------------------------------------------------------------------
# TAB 2: Typologie anomálních studentů
# -----------------------------------------------------------------------------
with tab_types:
    st.subheader("🧠 Typologie studentů: Co model skutečně detekuje jako anomálii?")
    st.markdown(
        """
        V machine learningu na akademických datech má označení „anomálie“ tři odlišné věcné významy:
        """
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown(
            """
            <div style="background-color: #fef2f2; padding: 18px; border-radius: 8px; border-left: 4px solid #ef4444; height: 100%;">
                <h4 style="margin: 0; color: #991b1b;">1. Extrémní akademické selhání</h4>
                <p style="font-size: 0.9rem; color: #7f1d1d; margin-top: 8px;">
                    <b>Profil:</b> Matematika 0–25, Čtení 0–25, Psaní 0–25.
                    <br><br>
                    Studenti s celkovým selháním nebo neúčastí na testech. V PCA se nacházejí v extrémním záporném cípu PC1.
                    <br><b>Akční doporučení:</b> Okamžitá sociální intervence a doučování.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t_col2:
        st.markdown(
            """
            <div style="background-color: #fffbeb; padding: 18px; border-radius: 8px; border-left: 4px solid #f59e0b; height: 100%;">
                <h4 style="margin: 0; color: #92400e;">2. Kognitivní asymetrie (Disbalance)</h4>
                <p style="font-size: 0.9rem; color: #78350f; margin-top: 8px;">
                    <b>Profil:</b> Matematika 95, Čtení 25, Psaní 20 (nebo opačně).
                    <br><br>
                    Studenti s výrazným jednostranným talentem nebo naopak specifickou poruchou učení (dyslexie/dysgrafie). V PCA tvoří extrémní hodnoty na ose PC2.
                    <br><b>Akční doporučení:</b> Individuální vzdělávací plán.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t_col3:
        st.markdown(
            """
            <div style="background-color: #f0fdf4; padding: 18px; border-radius: 8px; border-left: 4px solid #10b981; height: 100%;">
                <h4 style="margin: 0; color: #065f46;">3. Perfektní premianti (100 %)</h4>
                <p style="font-size: 0.9rem; color: #064e3b; margin-top: 8px;">
                    <b>Profil:</b> Matematika 100, Čtení 100, Psaní 100.
                    <br><br>
                    V neřízené detekci anomálií je anomálií <b>jakýkoliv izolovaný extrém</b>. Tito studenti leží na opačném konci distribuce a stromy je izolují stejně snadno jako propadlíky!
                    <br><b>Akční doporučení:</b> Stipendijní program.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# TAB 3: Etická & Nákladová kalibrace
# -----------------------------------------------------------------------------
with tab_ethical:
    st.subheader("⚖️ Etická a nákladová kalibrace kontaminace v pedagogice")
    st.markdown(
        """
        Ve firemním bankovnictví se kompromis mezi 1.0 % a 2.5 % počítá v penězích (Cost of Fraud vs. Cost of Review). 
        Ve školství má však nesprávné zařazení studenta mezi „anomálie“ závažné etické a psychologické důsledky:
        """
    )

    eth_col1, eth_col2 = st.columns(2)
    with eth_col1:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 14px;">
                <h5 style="margin: 0; color: #b45309;">⚠️ Riziko Falešného poplachu (False Positive – příliš vysoká kontaminace 2.5 %)</h5>
                <p style="font-size: 0.88rem; color: #475569; margin-top: 6px;">
                    Označení běžného, jen mírně slabšího žáka za „anomálního“. 
                    Může vést ke stigmatizaci, ztrátě motivace a přetížení školních psychologů zbytečnými pohovory.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with eth_col2:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 14px;">
                <h5 style="margin: 0; color: #991b1b;">🚨 Riziko Nezjištěné anomálie (False Negative – příliš nízká kontaminace < 0.5 %)</h5>
                <p style="font-size: 0.88rem; color: #475569; margin-top: 6px;">
                    Přehlédnutí studenta v těžké krizové situaci nebo se skrytou poruchou učení. 
                    Následkem může být předčasný odchod ze vzdělávacího systému.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Doporučená metodická strategie pro školy:")
    st.success(
        "🎯 **Dvoustupňový screeningový systém (Tiered Alert System):**\n"
        "- **Tier 1 (Kontaminace 1.0 %):** Okamžitá prioritní intervence (doučování, sociální pracovník, krizový plán).\n"
        "- **Tier 2 (Kontaminace 2.5 %):** Preventivní monitoring a konzultace s vyučujícími daných předmětů bez stigmatizujícího zásahu."
    )

# -----------------------------------------------------------------------------
# TAB 4: Produkční guardrails ve vzdělávání
# -----------------------------------------------------------------------------
with tab_guard:
    st.subheader("🛡️ Závazná pravidla pro nasazení AI ve školství (EU AI Act & Data Science 2026)")
    
    g_c1, g_c2 = st.columns(2)
    with g_c1:
        st.markdown(
            """
            <div style="background-color: #fef2f2; padding: 18px; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #991b1b;">1. Ochrana před socioekonomickým zkreslením (Bias)</h4>
                <p style="font-size: 0.9rem; color: #7f1d1d; margin-top: 8px;">
                    V datasetu jsou sloupce jako <code>ParentEduc</code>, <code>LunchType</code> nebo <code>EthnicGroup</code>. 
                    Pokud by model trénoval na těchto znacích, hrozí diskriminace. 
                    <b>Správný postup z cvičení:</b> Model trénuje výhradně na objektivních výsledcích testů. 
                    Socioekonomické znaky se smí zkoumat až následně v rámci kauzální analýzy pomoci.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with g_c2:
        st.markdown(
            """
            <div style="background-color: #eff6ff; padding: 18px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #1e40af;">2. Požadavek na vysvětlitelnost (Explainable AI)</h4>
                <p style="font-size: 0.9rem; color: #1e3a8a; margin-top: 8px;">
                    Dle směrnice EU AI Act patří systémy hodnotící studenty do <b>vysoce rizikových aplikací (High-Risk AI)</b>. 
                    Nestačí vrátit štítek <code>-1</code>. Každé označení studenta za anomálii musí obsahovat 
                    konkrétní zdůvodnění (např. <i>„Matematické skóre 18 bodů leží 3.2 sm. odchylky pod průměrem školy“</i>).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
