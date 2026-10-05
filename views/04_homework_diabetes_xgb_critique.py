"""
Domácí úkol (Session 2): Expertní analýza & Kritika modelu XGBoost (Diabetes)
=============================================================================
Dataset: data/diabetes.csv (768 pacientek)
Model: XGBClassifier
Precomputed: 04_Homework/data/diabetes_xgb_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🔬 Expertní analýza & Diagnostika: XGBoost na datech diabetu")
st.caption(
    "Kritické medicínské zhodnocení: Biologicky nemožné nuly v klinických datech, etické a lékařské dilema "
    "mezi Precision a Recall při screeningu chronických onemocnění a rozbor XGBoost Gain metriky."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "diabetes_xgb_precomputed.json"


@st.cache_data
def load_diabetes_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_diabetes_stats()
metrics = stats["test_metrics"] if stats else {}
feat_imp = stats["feature_importances"] if stats else []
meta = stats["metadata"] if stats else {}
baselines = stats["baselines"] if stats else {}

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Klíčový biomarker", "Glukóza nalačno", delta="XGBoost Gain > 45 %")
c2.metric("Chybějící inzulín", "48.7 % biologických nul", delta="XGBoost Sparsity-Aware")
c3.metric("Testovací Precision", f"{metrics.get('precision', 0.7568)*100:.1f} %", delta="Vysoká spolehlivost")
c4.metric("Testovací Recall", f"{metrics.get('recall', 0.4667)*100:.1f} %", delta="⚠️ 53.3 % diabetiček přehlédnuto")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 1. Biologické nuly v lékařských datech",
    "⚖️ 2. Medicínské dilema: Precision vs. Recall",
    "🌲 3. Důležitost příznaků (Gain vs. Weight)",
    "🥊 4. Benchmark modelů & Klinická doporučení"
])

# ==============================================================================
# TAB 1: BIOLOGICKÉ NULY
# ==============================================================================
with tab1:
    st.subheader("Fyziologicky nemožné nuly: Když data lžou bez chybových hlášení")

    st.markdown(
        """
        Pima Indians Diabetes dataset je celosvětově nejznámějším medicínským benchmarkem. 
        Při letmém pohledu přes `df.isnull().sum()` se zdá, že data **nemají žádné chybějící hodnoty**. 
        Skutečnost odhalená pomocí `ydata_profiling` je však zcela odlišná:
        """
    )

    zero_info = meta.get("biological_zeros", {})
    zero_rows = []
    medical_consequences = {
        "Glucose": "Glukóza 0 mg/dl znamená těžkou hypoglykémii a okamžitou smrt. U živého člověka vyloučeno.",
        "BloodPressure": "Tlak 0 mmHg znamená zástavu oběhu a smrt pacienta.",
        "SkinThickness": "Tloušťka kožní řasy na tricepsu nemůže být nulová (kůže má vždy tloušťku v milimetrech).",
        "Insulin": "Hodnota 0 µU/ml znamená, že inzulín nebyl v laboratoři změřen (velmi drahý test).",
        "BMI": "Index tělesné hmotnosti 0 znamená nulovou hmotnost těla."
    }

    for col, d in zero_info.items():
        zero_rows.append({
            "Biomarker (Atribut)": col,
            "Počet nulových záznamů": f"{d.get('zero_count')} pacientek",
            "Podíl z celku (768)": f"{d.get('zero_percentage'):.1f} %",
            "Fyziologická realita": medical_consequences.get(col, "")
        })

    st.dataframe(pd.DataFrame(zero_rows), hide_index=True, width="stretch")

    st.markdown("##### ⚡ Jak s tímto fenoménem nakládá XGBoost?")
    col_x1, col_x2 = st.columns([1, 1])

    with col_x1:
        st.markdown(
            """
            **Sparsity-Aware Split Finding (Algoritmus pro řídká data):**  
            Zatímco lineární modely nebo Random Forest ze Scikit-learn vyžadují imputaci (např. mediánem), 
            XGBoost obsahuje **nativní mechanismus pro chybějící hodnoty**:
            - Při každém dělení uzlu spočítá gradienty pro data jdoucí doleva i doprava.
            - Chybějící hodnoty (včetně `NaN`) pošle do té větve, která přinese **větší zisk (Gain)**.
            - Tím se model automaticky naučí, zda samotný fakt *„lékař neprovedl drahý test na inzulín“* 
              zvyšuje či snižuje pravděpodobnost diabetu!
            """
        )

    with col_x2:
        st.markdown(
            """
            **Porovnání strategií ošetření nul:**  
            - **Ponechání 0 jako čísla:** Strom může interpretovat nulu jako „extrémně nízkou hodnotu“, 
              což je u inzulinu biologický nesmysl, ale může fungovat jako náhodný příznak.
            - **Převod 0 na NaN a nativní XGBoost:** Čisté řešení, které odděluje absenci měření od nízké hladiny.
            - **Mediánová imputace:** Zastírá fakt, že u téměř 50 % pacientek nebyl test proveden, a uměle 
              snižuje rozptyl v datech.
            """
        )


# ==============================================================================
# TAB 2: MEDICÍNSKÉ DILEMA PRECISION VS RECALL
# ==============================================================================
with tab2:
    st.subheader("Etické a medicínské dilema: Proč je optimalizace na Precision v medicíně riskantní?")

    st.markdown(
        """
        Zadání domácího úkolu explicitně instruovalo:  
        `Create a GridSearchCV object... The metric we want to calculate is precision.`
        
        Pojďme podrobit tento požadavek kritické medicínské a datově-vědní analýze.
        """
    )

    p_val = metrics.get("precision", 0.7568)
    r_val = metrics.get("recall", 0.4667)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("##### 🎯 Co znamená Precision = 75.7 %?")
        st.markdown(
            f"""
            - Pokud model prohlásí: *„Tato pacientka má diabetes“*, má pravdu ve **3 ze 4 případů** ({p_val*100:.1f} %).
            - Falešně pozitivních (False Positives) je pouze **9 pacientek** ze 154.
            - **Výhoda:** Lékař zbytečně nestresuje zdravé pacientky a neposílá je na další testy.
            """
        )

    with col_d2:
        st.markdown("##### 🚨 Odvrácená tvář: Recall = 46.7 % (Katastrofa screeningu):")
        st.markdown(
            f"""
            - Ze 60 skutečně nemocných diabetiček v testovací sadě model odhalil **pouze 28**!
            - **32 pacientek s diabetem (53.3 %) model označil za ZDRAVÉ (False Negatives)!**
            - **Důsledek v praxi:** Přehlédnutý diabetes se neléčí, dochází k poškození cév, ledvin, 
              ztrátě zraku (retinopatii) či infarktu.
            """
        )

    st.markdown("---")
    st.markdown("##### ⚖️ Asymetrie nákladů (Cost-Sensitive Evaluation):")
    st.markdown(
        """
        V medicínském screeningu **nejsou obě chyby rovnocenné**:
        - **Cena False Positive:** Provedení kontrolního odběru krve / orálního testu (náklad ~100–300 Kč).
        - **Cena False Negative:** Neléčená cukrovka, dialýza, hospitalizace, zkrácení života (náklad statisíce Kč).

        > [!IMPORTANT]
        > **Závěr pro praxi:** V lékařském screeningu by se model měl optimalizovat na **Recall** 
        > (minimalizace přehlédnutých diagnóz) nebo **$F_2$-skóre**, které dává Recallu dvojnásobnou váhu 
        > oproti Precision!
        """
    )

    # Simulace prahů z předpočtených dat
    thresh_data = stats.get("threshold_tuning", [])
    if thresh_data:
        tdf = pd.DataFrame(thresh_data)
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(x=tdf["threshold"], y=tdf["precision"], mode="lines+markers", name="Precision (Přesnost)", line=dict(color="#2563eb", width=2)))
        fig_th.add_trace(go.Scatter(x=tdf["threshold"], y=tdf["recall"], mode="lines+markers", name="Recall (Záchyt)", line=dict(color="#ef4444", width=2)))
        fig_th.add_trace(go.Scatter(x=tdf["threshold"], y=tdf["f1"], mode="lines+markers", name="F1-Skóre", line=dict(color="#10b981", dash="dash", width=2)))
        fig_th.add_vline(x=0.5, line_dash="dot", line_color="#6b7280", annotation_text="Výchozí práh 0.50")
        fig_th.add_vline(x=0.3, line_dash="dot", line_color="#f59e0b", annotation_text="Doporučený klinický práh 0.30")
        fig_th.update_layout(
            title="Vliv rozhodovacího prahu na Precision vs. Recall (Trade-off)",
            xaxis_title="Rozhodovací práh (Threshold)",
            yaxis_title="Hodnota metriky",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_th, width="stretch")
        st.caption("Při snížení prahu na **0.30** stoupne Recall na **76.7 %** (zachráníme dalších 18 diabetiček) za cenu mírného poklesu Precision na 59 %.")


# ==============================================================================
# TAB 3: GAIN VS WEIGHT
# ==============================================================================
with tab3:
    st.subheader("Důležitost příznaků: XGBoost Gain vs. Weight vs. Cover")

    st.markdown(
        """
        Zatímco Random Forest měří důležitost pomocí Gini indexu, XGBoost poskytuje tři různé pohledy:
        - **Gain (Zisk):** Průměrný pokles logistické ztráty (logloss) přinesený štěpením daného příznaku. **Nejspolehlivější metrika.**
        - **Weight (Frekvence):** Kolikrát byl daný příznak použit jako dělící kritérium ve všech stromech.
        - **Cover (Pokrytí):** Průměrný počet vzorků (druhá derivace / Hessián) ovlivněných štěpením daného příznaku.
        """
    )

    if feat_imp:
        fi_df = pd.DataFrame(feat_imp).sort_values(by="gain", ascending=True)

        fig_gain = px.bar(
            fi_df,
            x="gain",
            y="feature",
            orientation="h",
            labels={"gain": "XGBoost Gain (Pokles logloss)", "feature": "Biomarker"},
            color="gain",
            color_continuous_scale="Magma",
            title="Důležitost příznaků dle XGBoost Gain"
        )
        fig_gain.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gain, width="stretch")

        st.dataframe(
            pd.DataFrame(feat_imp).rename(columns={
                "feature": "Atribut",
                "gain": "XGBoost Gain (Zisk)",
                "weight": "Weight (Počet štěpení)",
                "cover": "Cover (Pokrytí)"
            }),
            hide_index=True,
            width="stretch"
        )

    st.markdown(
        """
        > [!NOTE]
        > **Dominance glukózy:** Hladina glukózy nalačno (`Glucose`) má dramaticky nejvyšší Gain (přes 45 % celkového zisku modelu). 
        > To přesně odpovídá lékařské definici diabetu jako poruchy glukózového metabolismu. 
        > Věk (`Age`) a `BMI` mají vysoký `Weight`, protože slouží stromům k jemnému doladění rizikových skupin 
        > u pacientek na hranici normoglykémie.
        """
    )


# ==============================================================================
# TAB 4: BENCHMARK A DOPORUČENÍ
# ==============================================================================
with tab4:
    st.subheader("Velké srovnání modelů na datech diabetu & Doporučení pro praxi")

    st.markdown(
        """
        Porovnání optimalizovaného XGBoostu s klasickou logistickou regresí a náhodným lesem na stejné testovací sadě:
        """
    )

    lr_b = baselines.get("logistic_regression", {})
    rf_b = baselines.get("random_forest", {})

    b_df = pd.DataFrame([
        {
            "Model": "1. Logistická regrese (Baseline)",
            "Accuracy": f"{lr_b.get('accuracy', 0.7403)*100:.1f} %",
            "Precision (Diabetik)": f"{lr_b.get('precision', 0.7222)*100:.1f} %",
            "Recall (Záchyt)": f"{lr_b.get('recall', 0.5333)*100:.1f} %",
            "F1-skóre": f"{lr_b.get('f1', 0.6133):.4f}",
            "Charakteristika": "Lineární rozhodovací hranice, vysoká interpretovatelnost."
        },
        {
            "Model": "2. Random Forest (Bagging)",
            "Accuracy": f"{rf_b.get('accuracy', 0.7532)*100:.1f} %",
            "Precision (Diabetik)": f"{rf_b.get('precision', 0.7632)*100:.1f} %",
            "Recall (Záchyt)": f"{rf_b.get('recall', 0.5000)*100:.1f} %",
            "F1-skóre": f"{rf_b.get('f1', 0.6042):.4f}",
            "Charakteristika": "Redukce rozptylu průměrováním nezávislých stromů."
        },
        {
            "Model": "3. XGBoost (Tento model – Optimalizováno na Precision)",
            "Accuracy": f"{metrics.get('accuracy', 0.7338)*100:.1f} %",
            "Precision (Diabetik)": f"{metrics.get('precision', 0.7568)*100:.1f} %",
            "Recall (Záchyt)": f"{metrics.get('recall', 0.4667)*100:.1f} %",
            "F1-skóre": f"{metrics.get('f1_score', 0.5773):.4f}",
            "Charakteristika": "Sekvenční korekce reziduí s L1/L2 regularizací a gamma."
        }
    ])
    st.dataframe(b_df, hide_index=True, width="stretch")

    st.markdown("##### 💡 4 klíčová doporučení pro klinické nasazení:")
    st.markdown(
        """
        1. **Přenastavení rozhodovacího prahu na 0.30–0.35:** Pro screeningové použití zvýší záchyt diabetiček 
           na více než 75 % bez zásadního nárůstu falešných poplachů.
        2. **Klinická vysvětlitelnost přes SHAP (SHapley Additive exPlanations):** Místo pouhé diagnózy 
           by systém měl lékaři zobrazit vodopádový graf SHAP hodnot: *„Riziko pacientky je 78 % především kvůli 
           vysoké glykémii (+35 %) a věku (+15 %), zatímco normální krevní tlak riziko snižuje (-8 %).“*
        3. **Kalibrace pravděpodobností:** Výstup `predict_proba` v XGBoostu bývá kvůli regularizaci a boostingu 
           mírně deformovaný (přehnaně sebevědomý). Kalibrace pomocí **Plattova škálování** nebo **izotonické regrese** 
           zajistí, že 70% odhad skutečně odpovídá 70% pravděpodobnosti v populaci.
        4. **Validace na externí populaci:** Pima Indians mají unikátní genetickou predispozici k diabetu 
           (přes 35% prevalence). Model trénovaný na této populaci nelze bez rekalibrace použít na evropskou či asijskou populaci.
        """
    )
