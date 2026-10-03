"""
Homework: Příprava dat pro klasifikační modely – Diabetes Dataset
=================================================================
Interaktivní modul pro analýzu a předzpracování medicínských dat diabetu:
1. Normalizace názvů sloupců a detekce skrytých chybějících hodnot (biologicky nemožných nul).
2. Analýza distribucí (šikmost / skewness) a imputace (průměr pro symetrická rozdělení, medián pro šikmá).
3. Analýza vyváženosti tříd diagnózy (Outcome: 0 vs. 1).
4. Korelační matice a vliv jednotlivých biomarkerů na diabetes.
5. Standardizace příznaků (StandardScaler) a uložení do diabetes_scaled.csv.
6. Interaktivní průzkumník distribucí a klinický kalkulátor standardizovaných skóre (Z-score).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_precomputed_data():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "02_Classification" / "data" / "diabetes_preprocessing_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_raw_and_scaled_dfs():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "02_Classification" / "data" / "diabetes.csv"
    scaled_path = base_dir / "02_Classification" / "data" / "diabetes_scaled.csv"
    
    df_raw = pd.read_csv(raw_path) if raw_path.exists() else None
    df_scaled = pd.read_csv(scaled_path) if scaled_path.exists() else None
    return df_raw, df_scaled


def render_diabetes_preprocessing_view():
    st.title("🩺 DÚ: Příprava dat pro klasifikaci (Diabetes)")
    st.markdown(
        "**Zadání cvičení:** Načtení datové sady `diabetes.csv`, normalizace názvů sloupců do formátu snake_case, "
        "identifikace biologicky nemožných nulových hodnot, imputace dle tvaru distribuce (průměr vs. medián), "
        "korelační analýza a finální standardizace (`StandardScaler`)."
    )

    data = load_precomputed_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_Classification/15_homework_diabetes_preprocessing.py`.")
        return

    df_raw, df_scaled = load_raw_and_scaled_dfs()

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Celkem pacientek", f"{data['raw_shape'][0]}", delta="9 proměnných")
    with c2:
        r1 = data["class_distribution"]["ratio_1"] * 100
        st.metric("Výskyt diabetu (1)", f"{r1:.1f} %", delta=f"{data['class_distribution']['1']} pacientek")
    with c3:
        total_invalid_zeros = sum(data["created_nans"].values())
        st.metric("Neplatné nuly (NaN)", f"{total_invalid_zeros}", delta="5 biomarkerů")
    with c4:
        st.metric("Výstupní soubor", "diabetes_scaled.csv", delta="StandardScaler")

    st.markdown("---")

    # Sekce 1: Kontrola kvality dat a Biologicky nemožné nuly
    st.subheader("1. Detekce biologicky nemožných nul a imputace chybějících hodnot")
    st.markdown(
        "V medicínských datasetech (Pima Indians Diabetes) jsou chybějící měření často kódována jako hodnota `0`. "
        "Z fyziologického hlediska však živý člověk **nemůže mít nulovou glykémii, nulový krevní tlak, "
        "nulovou tloušťku kožní řasy, nulový 2hodinový inzulin ani nulové BMI**."
    )

    # Příprava přehledové tabulky
    zero_rows = []
    for col in data["columns_normalized"]:
        zc = data["zero_counts"].get(col, 0)
        pct = (zc / data["raw_shape"][0]) * 100
        is_invalid = col in data["zero_invalid_cols"]
        skew = data["skewness"].get(col, None)
        strat = data["imputation_strategy"].get(col, "Bez úpravy")
        imp_val = data["imputation_values"].get(col, None)

        zero_rows.append({
            "Proměnná": col,
            "Počet nul v CSV": zc,
            "Podíl nul (%)": f"{pct:.2f} %",
            "Biologický status": "❌ Nemocniční artefakt (Chybějící)" if is_invalid else "✅ Platná hodnota",
            "Šikmost (Skewness)": f"{skew:+.3f}" if skew is not None else "—",
            "Zvolená imputace": f"Průměr ({imp_val:.2f})" if strat == "mean" else (f"Medián ({imp_val:.2f})" if strat == "median" else "Ponecháno"),
            "Odůvodnění": "Symetrické rozdělení (|skew| < 0.5)" if strat == "mean" else ("Silně šikmé rozdělení (|skew| ≥ 0.5)" if strat == "median" else "Nuly jsou legitimní (např. 0 porodů)")
        })

    df_zero_table = pd.DataFrame(zero_rows)
    st.dataframe(df_zero_table, use_container_width=True, hide_index=True)

    # Interaktivní graf počtu nul vs. platných dat
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("##### 📊 Počet a podíl neplatných nulových hodnot")
        invalid_data = [
            {"Proměnná": col, "Počet chybějících (nul)": data["created_nans"][col], "Podíl (%)": (data["created_nans"][col] / 768) * 100}
            for col in data["zero_invalid_cols"]
        ]
        df_invalid = pd.DataFrame(invalid_data).sort_values("Počet chybějících (nul)", ascending=True)

        fig_zeros = px.bar(
            df_invalid,
            x="Počet chybějících (nul)",
            y="Proměnná",
            orientation="h",
            text="Počet chybějících (nul)",
            color="Podíl (%)",
            color_continuous_scale="Reds",
            title="Biologicky nemožné nuly nahrazené za NaN"
        )
        fig_zeros.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        fig_zeros.update_traces(textposition="outside")
        st.plotly_chart(fig_zeros, use_container_width=True)

    with col_chart2:
        st.markdown("##### ⚖️ Poměr cílové třídy `outcome`")
        c_dist = data["class_distribution"]
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=["0: Negativní (Bez diabetu)", "1: Pozitivní (Diabetes)"],
                values=[c_dist["0"], c_dist["1"]],
                hole=0.45,
                marker=dict(colors=["#3b82f6", "#ef4444"]),
                textinfo="label+percent+value"
            )
        ])
        fig_pie.update_layout(
            title="Distribuce diagnózy v populaci pacientek",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Sekce 2: Interaktivní průzkumník distribuce a srovnání před/po imputaci
    st.subheader("2. Interaktivní distribuce biomarkerů a vliv imputace")
    st.markdown(
        "Zvolte libovolnou proměnnou a porovnejte její původní rozdělení (včetně nereálných nul) "
        "s rozdělením po nahrazení za `NaN` a následné imputaci (průměr vs. medián)."
    )

    selected_feature = st.selectbox(
        "Vyberte proměnnou pro detailní analýzu:",
        options=data["columns_normalized"][:-1],
        index=data["columns_normalized"].index("glucose")
    )

    if df_raw is not None:
        # Původní data
        raw_vals = df_raw[
            {"pregnancies": "Pregnancies", "glucose": "Glucose", "blood_pressure": "BloodPressure",
             "skin_thickness": "SkinThickness", "insulin": "Insulin", "bmi": "BMI",
             "diabetes_pedigree_function": "DiabetesPedigreeFunction", "age": "Age"}.get(selected_feature, selected_feature)
        ]

        # Čistá data s NaN
        cleaned_series = raw_vals.replace(0, np.nan) if selected_feature in data["zero_invalid_cols"] else raw_vals
        skew_clean = cleaned_series.dropna().skew()
        mean_clean = cleaned_series.dropna().mean()
        median_clean = cleaned_series.dropna().median()

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.metric("Původní počet nul", f"{(raw_vals == 0).sum()}", delta=f"{(raw_vals == 0).mean() * 100:.1f} % z celku")
        with f_col2:
            st.metric("Šikmost (Skewness)", f"{skew_clean:+.3f}", delta="Symetrické" if abs(skew_clean) < 0.5 else "Šikmé")
        with f_col3:
            st.metric("Průměr (bez nul)", f"{mean_clean:.2f}")
        with f_col4:
            st.metric("Medián (bez nul)", f"{median_clean:.2f}")

        # Interaktivní histogram před vs po imputaci
        fig_dist = go.Figure()
        
        # Původní distribuce (včetně nul)
        fig_dist.add_trace(go.Histogram(
            x=raw_vals,
            name="Původní data (s nulami)",
            opacity=0.55,
            marker_color="#94a3b8",
            nbinsx=35
        ))

        # Data po imputaci
        if selected_feature in data["imputation_values"]:
            fill_v = data["imputation_values"][selected_feature]
            imputed_vals = cleaned_series.fillna(fill_v)
            fig_dist.add_trace(go.Histogram(
                x=imputed_vals,
                name=f"Po imputaci ({data['imputation_strategy'][selected_feature]}: {fill_v:.1f})",
                opacity=0.65,
                marker_color="#2563eb",
                nbinsx=35
            ))
            # Vertikální linka imputované hodnoty
            fig_dist.add_vline(
                x=fill_v,
                line_width=3,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text=f"Imputovaná hodnota = {fill_v:.1f}",
                annotation_position="top right"
            )

        fig_dist.update_layout(
            barmode="overlay",
            title=f"Histogram a porovnání rozdělení proměnné: {selected_feature}",
            xaxis_title=selected_feature,
            yaxis_title="Četnost (Počet pacientek)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # Sekce 3: Korelační analýza
    st.subheader("3. Korelační matice a prediktivní síla biomarkerů")
    st.markdown(
        "Zkoumáme lineární vztah (Pearsonův korelační koeficient $r$) mezi jednotlivými proměnnými po provedení imputace. "
        "Nejsilnějším samostatným prediktorem diabetu je **hladina glukózy**, následovaná **BMI** a **věkem**."
    )

    corr_cols = [c for c in data["columns_normalized"]]
    corr_matrix_data = [[data["correlation_matrix"][c1][c2] for c2 in corr_cols] for c1 in corr_cols]

    tab_corr1, tab_corr2 = st.tabs(["🔥 Plná korelační Heatmapa", "📈 Žebříček korelací s diagnózou (Outcome)"])

    with tab_corr1:
        fig_heat = px.imshow(
            corr_matrix_data,
            x=corr_cols,
            y=corr_cols,
            color_continuous_scale="Blues",
            text_auto=".2f",
            title="Pearsonova korelační matice po imputaci dat"
        )
        fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_corr2:
        outcome_corr_items = [
            {"Příznak": k, "Korelační koeficient r": v}
            for k, v in data["correlation_with_outcome"].items() if k != "outcome"
        ]
        df_corr_outcome = pd.DataFrame(outcome_corr_items).sort_values("Korelační koeficient r", ascending=True)

        fig_bar_corr = px.bar(
            df_corr_outcome,
            x="Korelační koeficient r",
            y="Příznak",
            orientation="h",
            text="Korelační koeficient r",
            color="Korelační koeficient r",
            color_continuous_scale="Teal",
            title="Korelace biomarkerů s výskytem diabetu (Outcome = 1)"
        )
        fig_bar_corr.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
        fig_bar_corr.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar_corr, use_container_width=True)

    st.markdown("---")

    # Sekce 4: Škálování dat (StandardScaler)
    st.subheader("4. Standardizace příznaků (StandardScaler)")
    st.markdown(
        "Protože klasifikační algoritmy (k-NN, Logistická regrese, SVM, Perceptron, Neuronové sítě) "
        "jsou citlivé na měřítko a rozptyl proměnných, provedeme **Standardizaci**: "
        "$$z = \\frac{x - \\mu}{\\sigma}$$ "
        "Výsledné příznaky mají průměr $\\mu = 0$ a směrodatnou odchylku $\\sigma = 1$. "
        "Cílovou binární proměnnou `outcome` ponecháváme beze změny ({0, 1})."
    )

    if df_scaled is not None:
        col_s1, col_s2 = st.columns([1, 1])

        with col_s1:
            st.markdown("##### 📋 Původní data (prvních 5 řádků)")
            st.dataframe(pd.DataFrame(data["sample_head_raw"]), use_container_width=True)

        with col_s2:
            st.markdown("##### 🎯 Škálovaná data v `diabetes_scaled.csv` (prvních 5 řádků)")
            st.dataframe(pd.DataFrame(data["sample_head_scaled"]), use_container_width=True)

        # Plotly boxplot škálovaných příznaků
        features_only = [c for c in df_scaled.columns if c != "outcome"]
        fig_box = px.box(
            df_scaled[features_only],
            title="Rozdělení standardizovaných příznaků (Z-skóre s nulovým průměrem)",
            labels={"value": "Standardizovaná hodnota (Z-skóre)", "variable": "Příznak"}
        )
        fig_box.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Sekce 5: Interaktivní klinický kalkulátor standardizace
    st.subheader("5. Interaktivní klinický kalkulátor standardizace (Z-score)")
    st.markdown(
        "Zadejte fyziologické hodnoty nové pacientky a sledujte její přepočet na standardizované skóre ($z$) "
        "vůči průměru a rozptylu trénovací populace."
    )

    with st.form("calc_form"):
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            user_preg = st.slider("Počet těhotenství", 0, 17, 2)
            user_glu = st.slider("Glukóza (mg/dl)", 50, 220, 120)
        with k2:
            user_bp = st.slider("Tlak krve (mm Hg)", 40, 130, 72)
            user_skin = st.slider("Kožní řasa (mm)", 7, 99, 29)
        with k3:
            user_ins = st.slider("Inzulin (mIU/ml)", 14, 846, 125)
            user_bmi = st.slider("BMI (kg/m²)", 18.0, 67.0, 32.0, step=0.1)
        with k4:
            user_dpf = st.slider("Diabetes pedigree", 0.07, 2.50, 0.47, step=0.01)
            user_age = st.slider("Věk (roky)", 21, 81, 33)

        submitted = st.form_submit_button("Vypočítat standardizovaný profil pacientky", use_container_width=True)

    user_values = {
        "pregnancies": user_preg,
        "glucose": user_glu,
        "blood_pressure": user_bp,
        "skin_thickness": user_skin,
        "insulin": user_ins,
        "bmi": user_bmi,
        "diabetes_pedigree_function": user_dpf,
        "age": user_age
    }

    z_scores = {}
    for feat, val in user_values.items():
        m = data["scaler_means"][feat]
        s = data["scaler_scales"][feat]
        z_scores[feat] = (val - m) / s

    df_user_z = pd.DataFrame([
        {
            "Biomarker": k,
            "Zadaná hodnota": f"{user_values[k]}",
            "Populační průměr (μ)": f"{data['scaler_means'][k]:.2f}",
            "Směrodatná odchylka (σ)": f"{data['scaler_scales'][k]:.2f}",
            "Z-skóre (z)": z_scores[k],
            "Interpretace": "Výrazný nadprůměr (> +1σ)" if z_scores[k] > 1 else ("Výrazný podprůměr (< -1σ)" if z_scores[k] < -1 else "V normálním rozmezí (±1σ)")
        }
        for k in user_values.keys()
    ])

    fig_user_radar = px.bar(
        df_user_z,
        x="Z-skóre (z)",
        y="Biomarker",
        orientation="h",
        color="Z-skóre (z)",
        color_continuous_scale="RdBu_r",
        text="Z-skóre (z)",
        range_color=[-3, 3],
        title="Odchylka pacientky od populačního průměru v jednotkách směrodatné odchylky (Z)"
    )
    fig_user_radar.update_traces(texttemplate="%{text:+.2f} σ", textposition="outside")
    fig_user_radar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_user_radar, use_container_width=True)

    with st.expander("🔍 Zobrazit detailní tabulku přepočtu Z-skóre"):
        st.dataframe(df_user_z, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Příprava dat: Diabetes", layout="wide")
    render_diabetes_preprocessing_view()
