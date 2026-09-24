import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("🐧 Cvičení 2: Multiclass metriky klasifikace (Druhy tučňáků)")
st.caption("Komplexní vyhodnocení kvality modelu k-NN na vícetřídních datech `penguins_df_normalized.csv`: Matice záměn 3x3, vážené metriky (Weighted F1, Precision, Recall) a hledání optima bez přetrénování.")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
norm_csv_path = data_dir / "penguins_df_normalized.csv"
if not norm_csv_path.exists():
    norm_csv_path = base_dir / "02_Classification" / "penguins_df_normalized.csv"
json_path = data_dir / "penguins_metrics_exercise_2_precomputed.json"


@st.cache_data
def load_penguins_data_and_split():
    df = pd.read_csv(norm_csv_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


@st.cache_data
def get_penguins_baseline_metrics(json_file_str: str, _X_train, _y_train, _X_test, _y_test):
    p = Path(json_file_str)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "baseline_k5" in data and isinstance(data["baseline_k5"], dict):
                    return data["baseline_k5"]
        except Exception:
            pass

    clf = KNeighborsClassifier(n_neighbors=5).fit(_X_train, _y_train)
    pred = clf.predict(_X_test)
    return {
        "accuracy": float(accuracy_score(_y_test, pred)),
        "precision_weighted": float(precision_score(_y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(_y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(_y_test, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(_y_test, pred, average="macro", zero_division=0)),
    }


@st.cache_data
def get_penguins_sweep_df(json_file_str: str, _X_train, _y_train, _X_test, _y_test):
    p = Path(json_file_str)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "sweep_data" in data and len(data["sweep_data"]) > 0:
                    return pd.DataFrame(data["sweep_data"])
        except Exception:
            pass

    results = []
    for k in range(1, 36):
        clf = KNeighborsClassifier(n_neighbors=k).fit(_X_train, _y_train)
        tr = clf.score(_X_train, _y_train)
        te = clf.score(_X_test, _y_test)
        pred = clf.predict(_X_test)
        results.append({
            "k": k,
            "train_accuracy": float(tr),
            "test_accuracy": float(te),
            "test_precision_weighted": float(precision_score(_y_test, pred, average="weighted", zero_division=0)),
            "test_recall_weighted": float(recall_score(_y_test, pred, average="weighted", zero_division=0)),
            "test_f1_weighted": float(f1_score(_y_test, pred, average="weighted", zero_division=0)),
            "test_f1_macro": float(f1_score(_y_test, pred, average="macro", zero_division=0)),
            "acc_gap": float(abs(tr - te)),
        })
    return pd.DataFrame(results)


df, X_train, X_test, y_train, y_test = load_penguins_data_and_split()
b5 = get_penguins_baseline_metrics(str(json_path), X_train, y_train, X_test, y_test)
sweep_df = get_penguins_sweep_df(str(json_path), X_train, y_train, X_test, y_test)
species_labels = ["Adelie", "Chinstrap", "Gentoo"]

# =============================================================================
# 1. METRICKÉ KARTY (VÝCHOZÍ MODEL k=5 DLE ZADÁNÍ)
# =============================================================================
st.subheader("🎯 Výchozí výsledky modelu ze zadání ($k = 5$, testovací sada $N=101$)")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Accuracy (Přesnost)", f"{b5.get('accuracy', 0.9406)*100:.2f} %", "95 ze 101 správně")
with c2:
    st.metric("Weighted Precision", f"{b5.get('precision_weighted', 0.9466)*100:.2f} %", "Vážená preciznost")
with c3:
    st.metric("Weighted Recall", f"{b5.get('recall_weighted', 0.9406)*100:.2f} %", "Vážená senzitivita")
with c4:
    st.metric("Weighted F1-score", f"{b5.get('f1_weighted', 0.9400)*100:.2f} %", "Harmonický průměr")
with c5:
    st.metric("Macro F1-score", f"{b5.get('f1_macro', 0.9401)*100:.2f} %", "Nevážený průměr")

st.markdown("---")

# =============================================================================
# 2. INTERAKTIVNÍ SIMULÁTOR HYPERPARAMETRU k
# =============================================================================
st.subheader("🎛️ Interaktivní simulátor: Výběr počtu sousedů ($k$)")

col_k, col_btns = st.columns([2, 1])
with col_k:
    selected_k = st.slider("Počet sousedů ($k$):", min_value=1, max_value=35, value=5, step=1)
with col_btns:
    st.write("**Rychlé volby:**")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("Výchozí k=5"):
            selected_k = 5
    with b_col2:
        if st.button("Optimum k=2"):
            selected_k = 2

# Trénování modelu pro vybrané k
knn_dyn = KNeighborsClassifier(n_neighbors=selected_k)
knn_dyn.fit(X_train, y_train)

train_acc_dyn = knn_dyn.score(X_train, y_train)
preds_dyn = knn_dyn.predict(X_test)
test_acc_dyn = accuracy_score(y_test, preds_dyn)
prec_w_dyn = precision_score(y_test, preds_dyn, average="weighted", zero_division=0)
rec_w_dyn = recall_score(y_test, preds_dyn, average="weighted", zero_division=0)
f1_w_dyn = f1_score(y_test, preds_dyn, average="weighted", zero_division=0)

cm_dyn = confusion_matrix(y_test, preds_dyn, labels=species_labels)

# Metrické zobrazení pro dynamické k
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    gap = abs(train_acc_dyn - test_acc_dyn)
    st.metric("Train Accuracy", f"{train_acc_dyn*100:.2f} %", delta=f"{gap*100:.1f} % gap" if selected_k == 1 else "OK")
with m2:
    st.metric("Test Accuracy", f"{test_acc_dyn*100:.2f} %", delta=f"{(test_acc_dyn - b5['accuracy'])*100:+.2f} % vs k=5")
with m3:
    st.metric("Test Precision (W)", f"{prec_w_dyn*100:.2f} %")
with m4:
    st.metric("Test Recall (W)", f"{rec_w_dyn*100:.2f} %")
with m5:
    st.metric("Test F1-score (W)", f"{f1_w_dyn*100:.2f} %")

# Diagnostická hlášení
if selected_k == 1:
    st.error("⚠️ **Příznak přetrénování (Overfitting):** Při $k=1$ je Train Accuracy 100 %, ale Test Accuracy padá na 94.06 % (gap téměř 6 %). Model se učil jednotlivé body včetně náhodného rozptylu.")
elif selected_k == 2:
    st.success(r"🏆 **Globální optimum modelu ($k=2$)!** Test Accuracy dosahuje **97.03 %** (98 ze 101 správně, pouze 3 chyby!) a Weighted F1 je **97.01 %**. Rozdíl mezi Train (98.71 %) a Test (97.03 %) je pouhých **1.68 %** $\implies$ **Žádné známky přetrénování!**")
elif selected_k == 5:
    st.info("📌 **Nejlepší liché $k=5$:** Školní výchozí nastavení. Zabraňuje remízám a dosahuje velmi dobré přesnosti 94.06 %.")
elif selected_k >= 25:
    st.warning("⚠️ **Počínající podtrénování (Underfitting):** Při příliš vysokém $k$ se rozhodovací hranice příliš vyhlazuje a model začíná chybovat (přesnost padá pod 80 %).")

# =============================================================================
# 3. GRAFICKÁ VIZUALIZACE: MATICE ZÁMĚN A KŘIVKY METRIK
# =============================================================================
col_p1, col_p2 = st.columns([1, 1.2])

with col_p1:
    # 3x3 Heatmap v Plotly
    text_matrix = []
    for i, actual in enumerate(species_labels):
        row = []
        for j, pred in enumerate(species_labels):
            val = cm_dyn[i, j]
            if i == j:
                row.append(f"<b>Správně: {val}</b>")
            else:
                row.append(f"Záměna: {val}" if val > 0 else "0")
        text_matrix.append(row)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_dyn,
        x=[f"Pred: {s}" for s in species_labels],
        y=[f"Aktuální: {s}" for s in species_labels],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale="Blues",
        showscale=False
    ))
    fig_cm.update_layout(
        title=f"<b>Matice záměn 3x3 pro k = {selected_k}</b>",
        xaxis_title="Predikovaný druh",
        yaxis_title="Skutečný druh",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_cm, width="stretch")

with col_p2:
    if sweep_df is not None and not sweep_df.empty:
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["k"], y=sweep_df["train_accuracy"]*100,
            mode="lines+markers", name="Train Accuracy",
            line=dict(color="#e74c3c", dash="dash", width=1.5), opacity=0.7
        ))
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["k"], y=sweep_df["test_accuracy"]*100,
            mode="lines+markers", name="Test Accuracy",
            line=dict(color="#3498db", width=2)
        ))
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["k"], y=sweep_df["test_f1_weighted"]*100,
            mode="lines+markers", name="Weighted F1-score (Optimum)",
            line=dict(color="#2ecc71", width=2.5)
        ))
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["k"], y=sweep_df["test_precision_weighted"]*100,
            mode="lines", name="Weighted Precision",
            line=dict(color="#9b59b6", width=1.5, dash="dot"), opacity=0.7
        ))
        fig_sweep.add_vline(x=selected_k, line_width=2, line_dash="dot", line_color="#f39c12",
                            annotation_text=f"Vybráno k={selected_k}", annotation_position="top left")
        fig_sweep.update_layout(
            title="<b>Vývoj metrik napříč k (1 až 35) & Detekce přetrénování</b>",
            xaxis_title="Počet sousedů (k)",
            yaxis_title="Hodnota metriky (%)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_sweep, width="stretch")

st.markdown("---")

# =============================================================================
# 4. TEORETICKÁ DISKUSE A ODPOVĚĎ NA ZADÁNÍ
# =============================================================================
st.subheader("💡 Odpověď na zadání a detailní analýza")

t1, t2, t3 = st.tabs(["🎯 Volba metriky: Weighted F1", "📉 Analýza přetrénování & Optimum", "📋 Tabulka sweepu"])

with t1:
    st.markdown("""
    ### Proč volíme Weighted F1-score jako klíčovou metriku?
    
    1. **Vícetřídní problém s nevyváženými třídami:**
       - Testovací vzorek obsahuje 44 tučňáků *Adelie*, 36 *Gentoo*, ale jen 21 *Chinstrap*.
       - Prostá Accuracy může být zavádějící, pokud by model ignoroval menšinovou třídu.
    2. **Rozdíl mezi způsoby průměrování (Averaging):**
       - **Macro Average**: Počítá aritmetický průměr metrik jednotlivých tříd bez ohledu na počet vzorků.
       - **Weighted Average**: Váží metriku každé třídy jejím skutečným zastoupením (*support*). Poskytuje nejvěrnější celkový obraz kvality klasifikátoru.
    3. **Kombinace Precision a Recall:**
       - V biologickém monitoringu je stejně důležité nezařazovat tučňáka do špatného druhu (Precision), jako žádný existující exemplář nepřehlédnout (Recall). Harmonický průměr $F_1$ penalizuje extrémní nerovnováhu mezi těmito aspekty.
    """)

with t2:
    st.markdown("""
    ### Hledání optimálního $k$ a testování přetrénování:
    
    | Parametr $k$ | Train Accuracy | Test Accuracy | Weighted F1 | Gap (Train - Test) | Stav |
    | :---: | :---: | :---: | :---: | :---: | :--- |
    | **$k = 1$** | **100.00 %** | 94.06 % | 94.05 % | **5.94 %** | 🚨 **Přetrénování (Overfitting)** |
    | **$k = 2$** | **98.71 %** | **97.03 %** | **97.01 %** | **1.68 %** | 🏆 **Globální optimum (Bez přetrénování)** |
    | **$k = 5$** | 96.99 % | 94.06 % | 94.00 % | 2.94 % | 📌 **Nejlepší liché $k$** |
    | **$k = 30$**| 79.40 % | 68.32 % | 66.37 % | 11.08 % | 📉 **Podtrénování (Underfitting)** |
    
    - **Proč je $k=2$ nejlepší a není přetrénovaný?**
      - Rozdíl mezi trénovací a testovací přesností činí pouhých **1.68 %**.
      - Z celkových 101 testovacích vzorků model spletl pouze 3 tučňáky (1 Adelie zařazen jako Chinstrap a 2 Chinstrap jako Adelie). Druh *Gentoo* byl rozpoznán se **100% precizností i senzitivitou**!
    """)

with t3:
    if sweep_df is not None and not sweep_df.empty:
        st.dataframe(
            sweep_df.rename(columns={
                "k": "k (sousedé)",
                "train_accuracy": "Train Acc",
                "test_accuracy": "Test Acc",
                "test_precision_weighted": "Weighted Precision",
                "test_recall_weighted": "Weighted Recall",
                "test_f1_weighted": "Weighted F1",
                "test_f1_macro": "Macro F1",
                "acc_gap": "Gap (|Train-Test|)"
            }),
            width="stretch"
        )
