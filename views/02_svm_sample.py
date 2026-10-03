import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "02_Classification" / "data" / "svm_classification_sample_precomputed.json"
plot_regions_path = base_dir / "02_Classification" / "plots" / "svm_sample_decision_regions.png"
plot_bar_path = base_dir / "02_Classification" / "plots" / "svm_sample_kernel_comparison.png"

st.title("🎯 Klasifikační SVM: Ukázka & Laboratoř")
st.caption("Implementace modelu SVC ze slajdů kurzu, srovnání jader (RBF vs. Linear vs. Poly), vizualizace rozhodovacích hranic a ladění hyperparametrů.")

if not json_path.exists():
    st.error("Předpočtená data `svm_classification_sample_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
def_m = data["default_model_rbf"]
kernel_res = data["kernel_comparison"]
best_grid = data["grid_search_best"]

# =============================================================================
# SEKCE 1: ŠKOLNÍ VÝSLEDKY ZE SLAJDŮ KURZU (Slajdy 9-13)
# =============================================================================
st.subheader("1. Školní model ze slajdů kurzu (`make_classification`)")
st.markdown(r"""
V materiálech kurzu (*Support vector machine - an example of implementation, slajdy 9–13*) je model demonstrován na syntetických datech:
- **Konfigurace:** 600 vzorků, 5 příznaků, 2 třídy (`random_state=42`)
- **Dělení dat:** 70 % trénovací sada ($420$ vzorků), 30 % testovací sada ($180$ vzorků)
- **Výchozí model:** `SVC()` bez zadání hyperparametrů (používá jádro **RBF**, $C=1{,}0$, $\gamma=\text{'scale'}$).
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (slajd 12)", f"{def_m['accuracy']*100:.1f} %", help="Přesně odpovídá 92.2 % uvedeným ve slajdu 12")
col2.metric("Precision (slajd 12)", f"{def_m['precision']*100:.1f} %", help="Přesně odpovídá 92.4 % uvedeným ve slajdu 12")
col3.metric("Recall", f"{def_m['recall']*100:.1f} %")
col4.metric("Podpůrné vektory", f"{def_m['n_support_vectors']}", help=f"Ze 420 trénovacích vzorků model vybral {def_m['n_support_vectors']} hraničních bodů")

c_cm, c_info = st.columns([1, 1])

with c_cm:
    st.markdown("#### Matice záměn výchozího SVC (RBF)")
    cm = np.array(def_m["confusion_matrix"])
    cm_labels = ["Třída 0", "Třída 1"]
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
        x=cm_labels,
        y=cm_labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(margin=dict(l=40, r=40, t=30, b=40), height=300)
    st.plotly_chart(fig_cm, width="stretch")

with c_info:
    st.markdown("#### Konstruktor třídy `SVC` v Scikit-learn (slajdy 5–7)")
    st.markdown(r"""
- **`kernel`:** Typ jádrové funkce (`'rbf'`, `'linear'`, `'poly'`, `'sigmoid'`).
- **`C`:** Regularizační parametr. Řídí šířku okraje a trest za chybné zařazení vzorků.
- **`gamma`:** Dosah vlivu jednotlivých trénovacích bodů u nelineárních jader.
- **`degree`:** Stupeň polynomu (pouze pro `kernel='poly'`).
- **`shrinking=True`:** Heuristické urychlení výpočtu vynecháním bodů ležících daleko od okraje.
- **`decision_function_shape='ovr'`:** Způsob vícedenní klasifikace (`ovr` vs `ovo`).
    """)

st.divider()

# =============================================================================
# SEKCE 2: SROVNÁNÍ JÁDER ZE SLAJDŮ KURZU (Slajd 12 a 13)
# =============================================================================
st.subheader("2. Srovnání jádrových funkcí (`rbf` vs. `linear` vs. `poly` vs. `sigmoid`)")
st.markdown(r"""
Podle slajdů 12–13 dosahuje manipulace s jádrem výrazného posunu v přesnosti:
- **Polynomiální jádro (`poly`):** Dosáhlo nejvyšší **Accuracy = 94,4 %** a **Precision = 95,6 %**! To naznačuje, že v datech existují mírně nelineární polynomiální závislosti.
- **Lineární jádro (`linear`):** Rovněž skvělý výsledek: **Accuracy = 93,3 %**, **Precision = 94,4 %**.
""")

# Tabulka srovnání
k_df = pd.DataFrame([
    {
        "Jádrová funkce": k.upper(),
        "Accuracy (%)": f"{res['accuracy']*100:.1f} %",
        "Precision (%)": f"{res['precision']*100:.1f} %",
        "Recall (%)": f"{res['recall']*100:.1f} %",
        "F1-score": f"{res['f1_score']:.4f}",
        "Počet podpůrných vektorů": res["n_support_vectors"],
        "Hodnocení ze slajdů": "Výchozí spolehlivá volba (slajd 12)" if k=="rbf" else (
            "Vynikající lineární separace (slajd 13)" if k=="linear" else (
                "🏆 Nejlepší výsledek v kurzu (slajd 13)" if k=="poly" else "Nejnižší metrika"
            )
        )
    }
    for k, res in kernel_res.items()
])
st.dataframe(k_df, width="stretch", hide_index=True)

if plot_bar_path.exists():
    st.image(str(plot_bar_path), caption="Grafické srovnání metrik jednotlivých jader SVM (Accuracy a Precision)", width="stretch")

st.divider()

# =============================================================================
# SEKCE 3: VIZUALIZACE ROZHODOVACÍCH HRANIC (Slajdy 13-14)
# =============================================================================
st.subheader("3. Vizualizace dělících oblastí a podpůrných vektorů (slajdy 13–14)")
st.markdown(r"""
V reálném 5D prostoru nelze nadrovinu přímo nakreslit. Zde je 2D projekce na první dva příznaky demonstrující, jak jednotlivá jádra tvarují rozhodovací oblasti:
- Černé kroužky označují **skutečné podpůrné vektory**, které drží okraj nadroviny.
""")

if plot_regions_path.exists():
    st.image(str(plot_regions_path), caption="Rozhodovací oblasti a podpůrné vektory pro Linear, RBF a Polynomial jádro", width="stretch")

st.divider()

# =============================================================================
# SEKCE 4: OPTIMALIZACE HYPERPARAMETRŮ (GridSearchCV)
# =============================================================================
st.subheader("4. Optimalizace hyperparametrů pomocí `GridSearchCV` (slajd 14)")
st.markdown(r"""
Systematickým prohledáním mřížky parametrů ($C$, $\gamma$, stupeň polynomu $d$, posun `coef0`) nacházíme optimální konfiguraci:
""")

bg_c1, bg_c2, bg_c3, bg_c4 = st.columns(4)
bg_c1.metric("Optimální jádro", f"{best_grid['best_params'].get('kernel', '').upper()}")
bg_c2.metric("Hodnota C", f"{best_grid['best_params'].get('C', '')}")
bg_c3.metric("Stupeň polynomu (degree)", f"{best_grid['best_params'].get('degree', 'N/A')}")
bg_c4.metric("Posun (coef0)", f"{best_grid['best_params'].get('coef0', 'N/A')}")

st.info(f"💡 Nejlepší parametry: `{best_grid['best_params']}` $\\implies$ **Precision = {best_grid['precision']*100:.2f} %**, **Accuracy = {best_grid['accuracy']*100:.2f} %** při {best_grid['n_support_vectors']} podpůrných vektorech.")
