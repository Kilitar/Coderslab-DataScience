import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "lumbar_svm_exercise_1_precomputed.json"
plot_tuning_path = base_dir / "02_Classification" / "plots" / "lumbar_svm_c_gamma_tuning.png"
plot_cm_path = base_dir / "02_Classification" / "plots" / "lumbar_svm_cm_comparison.png"

st.title("🎯 Cvičení 1: Bederní páteř – SVM & Ladění Precision")
st.caption("Splnění všech 8 kroků zadání kurzu: Načtení dat, výchozí SVC(), vyhodnocení Precision a experimenty s laděním jádra, C a gamma.")

if not json_path.exists():
    st.error("Předpočtená data `lumbar_svm_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
base_m = data["baseline_model"]
opt_lin = data["optimal_model_linear"]
opt_rbf = data["optimal_model_rbf_gamma"]
k_comp = pd.DataFrame(data["kernel_comparison"])
c_sweep = pd.DataFrame(data["c_linear_sweep"])
gamma_sweep = pd.DataFrame(data["gamma_rbf_sweep"])

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A KONTROLA PRVNÍCH 10 POZOROVÁNÍ
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pozorování (`head(10)`)")
st.markdown(r"""
Normalizovaný dataset biomechanických měření pánve (`lumbar_normalized_df.csv`) obsahuje **310 pacientů** a **6 anatomických úhlů**:
- `pelvic_incidence`, `pelvic_tilt`, `lumbar_lordosis_angle`, `sacral_slope`, `pelvic_radius`, `degree_spondylolisthesis`.
- Cílová proměnná `class`: `0` = Normal (zdravý pacient), `1` = Abnormal (výhřez ploténky / spondylolistéza).
""")

st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(f"Celkem: {meta['samples_total']} pacientů | Trénovací sada (75 %): {meta['samples_train']} vzorků | Testovací sada (25 %): {meta['samples_test']} vzorků (`random_state=42`).")

st.divider()

# =============================================================================
# KROK 3 AŽ 7: VÝCHOZÍ MODEL (BASELINE BEZ HYPERPARAMETRŮ)
# =============================================================================
st.subheader("3.–7. Výchozí model SVM (Baseline SVC())")
st.markdown(r"""
Model `SVC(random_state=42)` s výchozím RBF jádrem ($C=1{,}0$, $\gamma=\text{'scale'}$) natrénovaný na trénovací sadě (75 % dat):
- Z 232 trénovacích vzorků model vybral **124 podpůrných vektorů** (62 pro třídu 0 a 62 pro třídu 1).
- Výsledné metriky na testovací sadě ($78$ pacientů):
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Výchozí Precision", f"{base_m['precision']*100:.2f} %", help="Klíčová metrika sledovaná zadáním: TP / (TP + FP)")
m2.metric("Výchozí Accuracy", f"{base_m['accuracy']*100:.2f} %")
m3.metric("Výchozí Recall", f"{base_m['recall']*100:.2f} %")
m4.metric("Podpůrné vektory", f"{base_m['n_support_vectors']}", help="Počet hraničních vzorků určujících nadrovinu")

c_cm_base, c_info_base = st.columns([1, 1])

with c_cm_base:
    st.markdown("#### Matice záměn výchozího SVC")
    cm_b = np.array(base_m["confusion_matrix"])
    cm_labels = ["Normal (0)", "Abnormal (1)"]
    fig_cm_base = px.imshow(
        cm_b,
        labels=dict(x="Predikovaná diagnóza", y="Skutečná diagnóza", color="Počet"),
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm_base.update_layout(margin=dict(l=40, r=40, t=30, b=40), height=300)
    st.plotly_chart(fig_cm_base, width="stretch")

with c_info_base:
    st.markdown("#### Diagnostika chyb výchozího modelu")
    st.markdown(r"""
- **5 falešně pozitivních pacientů ($FP = 5$):** Model označil 5 zdravých lidí za abnormální.
- **Důvod:** Výchozí RBF jádro s parametrem $\gamma=\text{'scale'}$ vytváří mírně příliš zakřivenou hranici, která zachytává část zdravých pacientů.
- **Cíl kroku 8:** Experimentovat s volbou hyperparametrů ($C$, `kernel`, $\gamma$) a dosáhnout vyšší hodnoty Precision ($> 90{,}57\,\%$).
    """)

st.divider()

# =============================================================================
# KROK 8: EXPERIMENTY S HYPERPARAMETRY PRO ZVÝŠENÍ PRECISION
# =============================================================================
st.subheader("8. Experimenty s hyperparametry: Překonání výchozí Precision")
st.markdown(r"""
Podle prezentace máme tři hlavní páky pro optimalizaci SVM:
1. **Volba jádra (`kernel`):** Přepnutí z nelineárního RBF na lineární jádro `linear`.
2. **Ladění regularizace ($C$):** Řízení tvrdosti okraje u lineárního jádra.
3. **Ladění dosahu bodů ($\gamma$):** Optimalizace zakřivení u RBF jádra.
""")

# Srovnávací tabulka
st.markdown("#### Přehledové srovnání otestovaných konfigurací SVM")
comp_table = pd.DataFrame({
    "Konfigurace modelu": [
        "Výchozí SVC (RBF, default)",
        "Lineární SVC (C=1.0)",
        "Optimální Lineární SVC (C=2.0)",
        "Optimální RBF SVC (gamma=1.0)"
    ],
    "Jádro": ["RBF", "Linear", "Linear", "RBF"],
    "Hyperparametry": ["C=1.0, gamma='scale'", "C=1.0", "C=2.0", "gamma=1.0, C=1.0"],
    "Test Precision": [
        f"{base_m['precision']*100:.2f} %",
        "93.62 %",
        f"{opt_lin['precision']*100:.2f} %",
        f"{opt_rbf['precision']*100:.2f} %"
    ],
    "Falešně pozitivní (FP)": [5, 3, 2, 2],
    "Test Accuracy": [
        f"{base_m['accuracy']*100:.2f} %",
        "79.49 %",
        f"{opt_lin['accuracy']*100:.2f} %",
        f"{opt_rbf['accuracy']*100:.2f} %"
    ],
    "Zlepšení Precision": [
        "Baseline (0.00 %)",
        "+3.05 %",
        f"+{(opt_lin['precision'] - base_m['precision'])*100:.2f} % 🏆 (Maximum)",
        f"+{(opt_rbf['precision'] - base_m['precision'])*100:.2f} % 🏆 (Maximum)"
    ]
})
st.dataframe(comp_table, width="stretch", hide_index=True)

# Interaktivní simulátor ladění C a gamma
st.markdown("#### Interaktivní simulátor: Ladění parametrů SVM")
tab_lin, tab_rbf = st.tabs(["📏 Lineární jádro (ladění C)", "🌀 RBF jádro (ladění gamma)"])

with tab_lin:
    c_selected = st.select_slider("Zvolte hodnotu parametru C (Lineární jádro):", options=c_sweep["C"].tolist(), value=2.0)
    c_row = c_sweep[c_sweep["C"] == c_selected].iloc[0]

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Test Precision", f"{c_row['test_precision']*100:.2f} %", delta=f"{(c_row['test_precision'] - base_m['precision'])*100:.2f} % vs baseline")
    sc2.metric("Test Accuracy", f"{c_row['test_accuracy']*100:.2f} %")
    sc3.metric("Test Recall", f"{c_row['test_recall']*100:.2f} %")
    sc4.metric("Podpůrné vektory", f"{int(c_row['n_support_vectors'])}")

    fig_c = px.line(
        c_sweep, x="C", y="test_precision",
        markers=True,
        title="Vývoj Precision v závislosti na parametru C (Lineární jádro)",
        labels={"C": "Parametr C (regularizace)", "test_precision": "Test Precision"}
    )
    fig_c.add_hline(y=base_m["precision"], line_dash="dot", line_color="red", annotation_text="Výchozí RBF (90.57 %)")
    fig_c.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_c, width="stretch")

with tab_rbf:
    g_selected = st.select_slider("Zvolte hodnotu parametru gamma (RBF jádro):", options=gamma_sweep["gamma"].tolist(), value=1.0)
    g_row = gamma_sweep[gamma_sweep["gamma"] == g_selected].iloc[0]

    sg1, sg2, sg3, sg4 = st.columns(4)
    sg1.metric("Test Precision", f"{g_row['test_precision']*100:.2f} %", delta=f"{(g_row['test_precision'] - base_m['precision'])*100:.2f} % vs baseline")
    sg2.metric("Test Accuracy", f"{g_row['test_accuracy']*100:.2f} %")
    sg3.metric("Test Recall", f"{g_row['test_recall']*100:.2f} %")
    sg4.metric("Podpůrné vektory", f"{int(g_row['n_support_vectors'])}")

    fig_g = px.line(
        gamma_sweep, x="gamma", y="test_precision",
        markers=True,
        title="Vývoj Precision v závislosti na parametru gamma (RBF jádro)",
        labels={"gamma": "Parametr gamma", "test_precision": "Test Precision"}
    )
    fig_g.add_hline(y=base_m["precision"], line_dash="dot", line_color="red", annotation_text="Výchozí RBF (90.57 %)")
    fig_g.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_g, width="stretch")

st.divider()

# =============================================================================
# VIZUÁLNÍ DIAGNOSTIKA MATIC ZÁMĚN
# =============================================================================
st.subheader("9. Medicínský závěr & Srovnání matic záměn")
st.markdown(r"""
Jak ladění hyperparametrů ovlivnilo klinické rozhodování?
- **Výchozí model:** Způsobil **5 falešně pozitivních diagnóz** (zdravý pacient vystaven stresu a zbytečné invazivní léčbě).
- **Optimalizovaný model (`kernel='linear'`, $C=2.0$ nebo `kernel='rbf'`, $\gamma=1.0$):** Snížil počet falešně pozitivních diagnóz na **pouhé 2**, čímž **Precision stoupla na vynikajících 95,65 %**!
""")

if plot_cm_path.exists():
    st.image(str(plot_cm_path), caption="Porovnání matic záměn: Výchozí SVC (FP=5) vs. Lineární SVC (FP=2) vs. RBF SVC (FP=2)", width="stretch")
