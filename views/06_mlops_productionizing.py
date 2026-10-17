"""
Produkční nasazení modelů strojového učení (MLOps & Deployment)
==============================================================
Interaktivní platforma pro výuku nasazování modelů:
1. Lokace nasazení (On-Premise vs. Cloud vs. Edge kalkulátor)
2. Povaha aplikace (Real-Time API vs. Batch processing vs. Streaming)
3. Simulátor Data Driftu a Concept Driftu v produkci
4. Produkční MLOps checklist & kódová šablona
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(page_title="Produkční nasazení & MLOps", page_icon="🚀", layout="wide")

st.title("🚀 Produkční nasazení modelů & MLOps")
st.caption(
    "Převod modelů z Jupyter sandboxu do živého produkčního provozu: "
    "Strategie nasazení (Cloud vs. Edge vs. On-Premise), Real-time vs. Dávkové predikce, "
    "monitoring Data Driftu a automatizace životního cyklu (MLOps)."
)

tab_arch, tab_nature, tab_drift, tab_checklist = st.tabs([
    "🏢 Kde model poběží? (Architektura)",
    "⚡ Kdy počítat predikce? (Real-time vs. Batch)",
    "📉 Simulátor Data Driftu v produkci",
    "📦 Produkční checklist & Šablony"
])

# -----------------------------------------------------------------------------
# TAB 1: Architektura nasazení (On-Premise vs. Cloud vs. Edge)
# -----------------------------------------------------------------------------
with tab_arch:
    st.subheader("🏢 Výběr optimální lokace pro nasazení (Site-Dependent Deployment)")
    st.markdown(
        """
        Při návrhu infrastruktury pro ML model musíme vyvážit **tři klíčové faktory**: 
        požadavky na bezpečnost citlivých dat, toleranci latence a dostupný rozpočet (CapEx vs. OpEx).
        """
    )

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### ⚙️ Parametry vašeho produkčního systému:")
        data_sensitivity = st.select_slider(
            "Citlivost zpracovávaných dat (Regulace):",
            options=["Veřejná data", "Interní firemní data", "Přísně regulovaná (GDPR / Bankovní tajemství / Zdravotnictví)"],
            value="Interní firemní data"
        )
        traffic_volatility = st.select_slider(
            "Volatilita návštěvnosti (Špičky vs. Klid):",
            options=["Konstantní stabilní zátěž", "Mírné výkyvy během dne", "Extrémní nepředvídatelné špičky (Black Friday)"],
            value="Mírné výkyvy během dne"
        )
        latency_tolerance = st.select_slider(
            "Požadavek na latenci odezvy:",
            options=["Toleruje sekundy až minuty", "Běžná webová odezva (< 200 ms)", "Ultra nízká odezva bez internetu (< 10 ms)"],
            value="Běžná webová odezva (< 200 ms)"
        )

        st.markdown("---")
        st.markdown("#### 🎯 Doporučená lokace nasazení:")
        if latency_tolerance == "Ultra nízká odezva bez internetu (< 10 ms)":
            st.success("📱 **Doporučení: Edge / On-Device Deployment (Mobil, IoT, Browser)**")
            st.write(
                "Model musí běžet přímo na zařízení klienta (přes ONNX Runtime, CoreML, TensorFlow Lite). "
                "Odpadá veškerá síťová latence a data neopouštějí zařízení."
            )
        elif data_sensitivity == "Přísně regulovaná (GDPR / Bankovní tajemství / Zdravotnictví)" and traffic_volatility == "Konstantní stabilní zátěž":
            st.success("🏦 **Doporučení: Lokální server (On-Premise Datacenter)**")
            st.write(
                "Vysoká bezpečnost, plná kontrola nad daty uvnitř privátní sítě. "
                "Při stabilní zátěži se vyplatí fixní investice do vlastních serverů."
            )
        else:
            st.success("☁️ **Doporučení: Cloudový server (AWS / GCP / Azure)**")
            st.write(
                "Maximální elasticita, automatické škálování při špičkách a managed služby "
                "(např. Google Cloud Run, AWS ECS/SageMaker). Ideální pro proměnlivou zátěž."
            )

    with c_right:
        # Radarový graf srovnání
        categories = ["Škálovatelnost", "Bezpečnost dat", "Nízká latence", "Nízké CapEx náklady", "Jednoduchost aktualizace"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[4, 9, 6, 3, 7],
            theta=categories,
            fill="toself",
            name="Lokální On-Premise"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[10, 7, 7, 9, 9],
            theta=categories,
            fill="toself",
            name="Cloudový server"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[8, 9, 10, 8, 3],
            theta=categories,
            fill="toself",
            name="Edge / Koncové zařízení"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Srovnání architektur (1 = nejhorší, 10 = nejlepší)",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar)

    st.markdown("---")
    st.markdown("#### 📋 Přehledná komparativní matice architektur")
    arch_df = pd.DataFrame([
        {
            "Infrastruktura": "On-Premise (Lokální)",
            "Hlavní výhody": "Naprostá kontrola nad daty, bezpečnost, žádné cloudové předplatné.",
            "Rizika & Nevýhody": "Omezená kapacita, drahý hardware, nutný interní DevOps tým.",
            "Příklady": "Bankovní systémy, nemocnice, vládní úřady."
        },
        {
            "Infrastruktura": "Cloud (AWS / GCP / Azure)",
            "Hlavní výhody": "Elasticita (auto-scaling), globální dostupnost, managed MLOps služby.",
            "Rizika & Nevýhody": "Vendor lock-in, průběžné provozní náklady (OpEx) při špatné správě.",
            "Příklady": "E-shopy, SaaS aplikace, webové portály, startupy."
        },
        {
            "Infrastruktura": "Edge / Koncové zařízení",
            "Hlavní výhody": "Funguje 100% offline, nulová latence, nulové náklady na cloudové servery.",
            "Rizika & Nevýhody": "Omezená paměť a baterie, složitá distribuce aktualizací modelu.",
            "Příklady": "FaceID v mobilu, autonomní drony, filtry fotoaparátu, smartwatch."
        }
    ])
    st.dataframe(arch_df, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 2: Povaha aplikace (Real-time vs. Batch)
# -----------------------------------------------------------------------------
with tab_nature:
    st.subheader("⚡ Jak se počítají predikce: Real-time vs. Dávkové zpracování (Batch)")
    st.markdown(
        """
        Zásadní architektonické rozhodnutí: **Potřebuje uživatel výsledek okamžitě v reálném čase, 
        nebo stačí predikce spočítat hromadně v noci a uložit do databáze?**
        """
    )

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown(
            """
            <div style="background-color: #eff6ff; padding: 20px; border-radius: 8px; border-left: 5px solid #3b82f6;">
                <h4 style="color: #1e3a8a; margin: 0;">⚡ 1. Real-Time (Online) Inference</h4>
                <p style="color: #1e40af; font-size: 0.9rem; margin-top: 8px;">
                    Model je zabalen jako webová služba (REST API / gRPC) a čeká na příchozí HTTP požadavky.
                </p>
                <ul style="color: #1e3a8a; font-size: 0.88rem;">
                    <li><b>Princip:</b> Synchronní Request $\rightarrow$ Response smyčka.</li>
                    <li><b>Typická latence:</b> 10 ms – 200 ms.</li>
                    <li><b>Typické technologie:</b> FastAPI, Flask, Triton Inference Server, Docker, Kubernetes.</li>
                    <li><b>Příklady:</b>
                        <ul>
                            <li>Okamžité ocenění bytu po vyplnění formuláře na realitním portálu.</li>
                            <li>Fraud detection při autorizaci platební karty (zamítnutí do 100 ms).</li>
                            <li>Chatbot a generování textu v reálném čase.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_b2:
        st.markdown(
            """
            <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 5px solid #22c55e;">
                <h4 style="color: #14532d; margin: 0;">📦 2. Batch (Dávková) Inference</h4>
                <p style="color: #166534; font-size: 0.9rem; margin-top: 8px;">
                    Model se spouští v naplánovaných intervalech (např. každou noc v 02:00) na obrovských objemech dat.
                </p>
                <ul style="color: #14532d; font-size: 0.88rem;">
                    <li><b>Princip:</b> Asynchronní hromadné zpracování databáze s uložením výsledků do SQL/NoSQL.</li>
                    <li><b>Typická latence:</b> Minuty až hodiny (není kritická).</li>
                    <li><b>Typické technologie:</b> Apache Airflow, Cron, Spark, Ray, Google BigQuery.</li>
                    <li><b>Příklady:</b>
                        <ul>
                            <li>Doporučovací systém nabídek práce: noční přepočet skóre pro 500 000 uživatelů.</li>
                            <li>Měsíční scoring pravděpodobnosti odchodu zákazníka (Churn prediction).</li>
                            <li>Dávková segmentace klientů marketingového oddělení.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("💡 Pravidlo volby: Kdy co zvolit?")
    st.info(
        "Kde funguje real-time zpracování, tam dávkové selhává (uživatel nechce čekat na schválení transakce do půlnoci). "
        "A naopak: Nasadit drahý real-time server na predikci, kterou stačí spočítat jednou za měsíc v noci, "
        "je masivní plýtvání financemi a infrastrukturou."
    )

# -----------------------------------------------------------------------------
# TAB 3: Simulátor Data Driftu v produkci
# -----------------------------------------------------------------------------
with tab_drift:
    st.subheader("📉 Simulátor stárnutí modelu: Detekce Data Driftu")
    st.markdown(
        """
        Po nasazení modelu do produkce se svět nezastaví. Distribuce vstupních proměnných 
        se může v čase posunout (**Data Drift** / Covariate Shift). 
        Vyzkoušejte si, jak statistický test (Kolmogorov-Smirnov test) odhalí posun v datech:
        """
    )

    np.random.seed(42)
    n_samples = 600

    col_drift_left, col_drift_right = st.columns([1, 2])
    with col_drift_left:
        st.markdown("#### 🎛️ Simulace produkčních dat")
        drift_mean = st.slider(
            "Posun střední hodnoty produkčních dat (μ_prod - μ_train):",
            min_value=0.0,
            max_value=3.0,
            value=0.2,
            step=0.1
        )
        drift_std = st.slider(
            "Změna rozptylu (σ_prod / σ_train):",
            min_value=0.5,
            max_value=2.5,
            value=1.0,
            step=0.1
        )
        alpha_level = st.selectbox("Hladina významnosti testu (α):", [0.05, 0.01, 0.001], index=0)

    # Generování dat
    train_dist = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    prod_dist = np.random.normal(loc=drift_mean, scale=drift_std, size=n_samples)

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.ks_2samp(train_dist, prod_dist)

    with col_drift_right:
        df_plot = pd.DataFrame({
            "Hodnota proměnné": np.concatenate([train_dist, prod_dist]),
            "Prostředí": ["Trénovací data (Referenční)"] * n_samples + ["Produkční data (Aktuální)"] * n_samples
        })
        fig_drift = px.histogram(
            df_plot,
            x="Hodnota proměnné",
            color="Prostředí",
            barmode="overlay",
            opacity=0.65,
            nbins=40,
            title=f"Distribuce: Trénink vs. Produkce (KS-stat: {ks_stat:.3f}, p-hodnota: {p_value:.4e})",
            color_discrete_map={
                "Trénovací data (Referenční)": "#2563eb",
                "Produkční data (Aktuální)": "#ef4444"
            }
        )
        st.plotly_chart(fig_drift)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("KS Statistika (vzdálenost distribucí)", f"{ks_stat:.3f}")
    with m2:
        st.metric("p-hodnota", f"{p_value:.4f}")
    with m3:
        if p_value < alpha_level:
            st.error(f"🚨 DETEKOVÁN DATA DRIFT! (p < {alpha_level})")
        else:
            st.success(f"✅ Žádný signifikantní drift (p >= {alpha_level})")

    if p_value < alpha_level:
        st.warning(
            "⚠️ **Akce MLOps pipeline:** Detekován statisticky významný posun vstupní distribuce! "
            "Automatický systém by měl vyvolat poplach (Slack/PagerDuty), zaznamenat nová data do "
            "Feature Store a spustit proces přetrénování (Retraining Pipeline)."
        )

# -----------------------------------------------------------------------------
# TAB 4: Produkční checklist & Šablony
# -----------------------------------------------------------------------------
with tab_checklist:
    st.subheader("📦 MLOps Produkční Checklist & Ukázkový kód")

    c_chk, c_code = st.columns([1, 1])

    with c_chk:
        st.markdown("#### ✅ 10 kroků před schválením modelu do produkce:")
        st.checkbox("1. Model i preprocessory jsou zapouzdřeny v jediné Scikit-learn Pipeline", value=True)
        st.checkbox("2. Byl ověřen nulový Data Leakage při validaci", value=True)
        st.checkbox("3. Model je serializován ve stabilním formátu (joblib / ONNX)", value=True)
        st.checkbox("4. Vstupy do REST API jsou validovány typovým schématem (Pydantic)", value=True)
        st.checkbox("5. Latence odezvy splňuje SLA (např. 95. percentil < 150 ms)", value=True)
        st.checkbox("6. Aplikace běží v izolovaném Docker kontejneru", value=True)
        st.checkbox("7. API má implementován healthcheck endpoint (`/health`)", value=True)
        st.checkbox("8. Všechny predikce a vstupy jsou logovány pro budoucí audit", value=True)
        st.checkbox("9. Je nastaven monitoring Data Driftu a pádů služby", value=True)
        st.checkbox("10. Existuje plán pro okamžitý rollback na předchozí verzi modelu", value=True)

    with c_code:
        st.markdown("#### 💻 Produkční FastAPI šablona pro real-time serving:")
        st.code(
            '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Production Service", version="1.0.0")

# 1. Načtení serializované pipeline do paměti při startu
pipeline = joblib.load("model_pipeline.joblib")

# 2. Striktní typová validace příchozího JSON payloadu
class ClientFeatures(BaseModel):
    age: float
    annual_income: float
    spending_score: float

@app.get("/health")
def healthcheck():
    return {"status": "healthy", "model_version": "1.0.0"}

@app.post("/predict")
def predict_client(payload: ClientFeatures):
    try:
        data = np.array([[payload.age, payload.annual_income, payload.spending_score]])
        pred = pipeline.predict(data)[0]
        prob = pipeline.predict_proba(data)[0].tolist()
        return {
            "prediction": int(pred),
            "probabilities": prob,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
''',
            language="python"
        )
