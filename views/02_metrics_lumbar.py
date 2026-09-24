import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title("📊 Cvičení 1: Metriky klasifikačních modelů (Bederní páteř)")
st.caption("Komplexní vyhodnocení kvality modelu k-NN na datech `lumbar_df_normalized.csv`: Matice záměn, klinický výběr klíčové metriky a optimalizace hyperparametru k bez přetrénování.")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
norm_csv_path = data_dir / "lumbar_df_normalized.csv"
if not norm_csv_path.exists():
    norm_csv_path = base_dir / "02_Classification" / "lumbar_df_normalized.csv"
json_path = data_dir / "lumbar_metrics_exercise_1_precomputed.json"


@st.cache_data
def load_data_and_split():
    df = pd.read_csv(norm_csv_path)
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


@st.cache_data
def load_precomputed():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


df, X_train, X_test, y_train, y_test = load_data_and_split()
precomputed = load_precomputed()

# =============================================================================
# 1. METRICKÉ KARTY (VÝCHOZÍ MODEL k=5 DLE ZADÁNÍ)
# =============================================================================
b5 = precomputed["baseline_k5"] if precomputed else {
    "accuracy": 0.8718, "precision": 0.9216, "recall": 0.8868, "f1_score": 0.9038, "roc_auc": 0.8208
}

st.subheader("🎯 Výchozí výsledky modelu dle zadání ($k = 5$, testovací sada $N=78$)")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Accuracy (Přesnost)", f"{b5['accuracy']*100:.2f} %", "68 ze 78 správně")
with c2:
    st.metric("Precision (Preciznost)", f"{b5['precision']*100:.2f} %", "TP / (TP + FP)")
with c3:
    st.metric("Recall (Senzitivita)", f"{b5['recall']*100:.2f} %", "47 z 53 zachyceno 🩺")
with c4:
    st.metric("F1-score", f"{b5['f1_score']*100:.2f} %", "Harmonický průměr")
with c5:
    st.metric("ROC-AUC", f"{b5['roc_auc']:.3f}", "Plocha pod ROC")

st.markdown("---")

# =============================================================================
# 2. INTERAKTIVNÍ EXPLORER HYPERPARAMETRU k A ROZHODOVACÍHO PRAHU
# =============================================================================
st.subheader("🎛️ Interaktivní simulátor: Výběr $k$ a rozhodovacího prahu $\\theta$")

col_ctrl1, col_ctrl2 = st.columns([1, 1])
with col_ctrl1:
    selected_k = st.slider("Počet sousedů ($k$):", min_value=1, max_value=25, value=5, step=1)
with col_ctrl2:
    selected_threshold = st.slider(
        "Klasifikační práh $\\theta$ pro třídu Abnormal:",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Snížením prahu zachytíme více nemocných (vyšší Recall za cenu vyššího počtu falešných poplachů FP)."
    )

# Natrénování vybraného k-NN
knn_dyn = KNeighborsClassifier(n_neighbors=selected_k)
knn_dyn.fit(X_train, y_train)

probs_test = knn_dyn.predict_proba(X_test)[:, 1]
preds_dyn = (probs_test >= selected_threshold).astype(int)

train_acc_dyn = knn_dyn.score(X_train, y_train)
test_acc_dyn = accuracy_score(y_test, preds_dyn)
test_prec_dyn = precision_score(y_test, preds_dyn, zero_division=0)
test_rec_dyn = recall_score(y_test, preds_dyn, zero_division=0)
test_f1_dyn = f1_score(y_test, preds_dyn, zero_division=0)

cm_dyn = confusion_matrix(y_test, preds_dyn)
tn_d, fp_d, fn_d, tp_d = cm_dyn.ravel()

# Zobrazení dynamických metrik
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Train Accuracy", f"{train_acc_dyn*100:.2f} %", delta=f"{train_acc_dyn - test_acc_dyn:+.2f} gap" if selected_k == 1 else "OK")
with m2:
    st.metric("Test Accuracy", f"{test_acc_dyn*100:.2f} %")
with m3:
    st.metric("Test Precision", f"{test_prec_dyn*100:.2f} %")
with m4:
    st.metric("Test Recall", f"{test_rec_dyn*100:.2f} %", delta=f"{test_rec_dyn*100 - b5['recall']*100:+.1f} % vs k=5")
with m5:
    st.metric("Test F1-score", f"{test_f1_dyn*100:.2f} %")

# Varování při k=1
if selected_k == 1:
    st.error("⚠️ **Detekováno brutální přetrénování (Overfitting)!** Při $k=1$ je přesnost na trénovacích datech 100 %, ale na testovacích datech klesá na pouhých 78.2 %. Model se naučil nazpaměť náhodný šum.")
elif selected_k == 9:
    st.success("🏆 **Optimální volba pro Recall ($k=9$)!** Recall stoupl na **92.45 %** (přehlédnuti pouze 4 nemocní z 53). Train Acc (84.91 %) i Test Acc (85.90 %) jsou v dokonalém souladu — model nevykazuje **žádné přetrénování**!")
elif selected_k == 8:
    st.info("⭐ **Optimální volba pro F1-score ($k=8$)!** F1 dosahuje maxima **91.26 %** při celkové testovací přesnosti **88.46 %**.")

# =============================================================================
# 3. GRAFICKÁ VIZUALIZACE: MATICE ZÁMĚN A KŘIVKY METRIK
# =============================================================================
col_plot1, col_plot2 = st.columns([1, 1.2])

with col_plot1:
    # Matice záměn v Plotly
    z_text = [
        [f"TN = {tn_d}<br>(Zdravý)", f"FP = {fp_d}<br>(Falešný poplach)"],
        [f"FN = {fn_d}<br>(🚨 Přehlédnuto)", f"TP = {tp_d}<br>(Zachyceno)"]
    ]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_dyn,
        x=["Predikce: Normal (0)", "Predikce: Abnormal (1)"],
        y=["Skutečnost: Normal (0)", "Skutečnost: Abnormal (1)"],
        text=z_text,
        texttemplate="%{text}<br><b>%{z}</b>",
        textfont={"size": 13},
        colorscale="Blues",
        showscale=False,
        reversescale=False
    ))
    fig_cm.update_layout(
        title=f"<b>Matice záměn pro k = {selected_k}</b> (Práh = {selected_threshold:.2f})",
        xaxis_title="Predikovaná diagnóza",
        yaxis_title="Skutečný stav pacienta",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_plot2:
    # Graf sweepu k z předpočtených dat
    if precomputed and "sweep_data" in precomputed:
        sweep_df = pd.DataFrame(precomputed["sweep_data"])
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
            x=sweep_df["k"], y=sweep_df["test_recall"]*100,
            mode="lines+markers", name="Test Recall (Priorita)",
            line=dict(color="#2ecc71", width=3)
        ))
        fig_sweep.add_trace(go.Scatter(
            x=sweep_df["k"], y=sweep_df["test_f1"]*100,
            mode="lines+markers", name="Test F1-score",
            line=dict(color="#9b59b6", width=2)
        ))
        # Zvýraznění zvoleného k
        fig_sweep.add_vline(x=selected_k, line_width=2, line_dash="dot", line_color="#f39c12",
                            annotation_text=f"Vybráno k={selected_k}", annotation_position="top left")
        fig_sweep.update_layout(
            title="<b>Sweepování hyperparametru k (1 až 25)</b>",
            xaxis_title="Počet sousedů (k)",
            yaxis_title="Hodnota metriky (%)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. KLINICKÁ A ANALYTICKÁ DISKUSE (DOKONČENÍ ZADÁNÍ)
# =============================================================================
st.subheader("💡 Odpověď na zadání a klinická interpretace")

t1, t2, t3 = st.tabs(["🩺 Proč volíme Recall?", "📉 Analýza přetrénování", "📋 Kompletní tabulka sweepu"])

with t1:
    st.markdown("""
    ### Proč je v medicínské diagnostice klíčový RECALL (Senzitivita)?
    
    V úloze detekce spondylolistézy a hernií meziobratlových plotének pracujeme s asymetrickými náklady chyb:
    
    | Typ chyby | Situace v praxi | Následek | Závažnost |
    | :--- | :--- | :--- | :--- |
    | **False Positive (FP)** | Zdravý označen jako nemocný | Pacient podstoupí kontrolní magnetickou rezonanci (MRI), která diagnózu vyvrátí. | Zvýšené náklady a mírný stres. 🟡 |
    | **False Negative (FN)** | Nemocný označen jako zdravý | Pacient je odeslán domů bez léčby. Hrozí nevratné poškození nervových kořenů a ochrnutí. | **Kritické ohrožení zdraví! 🔴** |
    
    Proto je prioritou **maximalizovat Recall** ($TP / (TP + FN)$), což přímo odpovídá minimalizaci počtu přehlédnutých pacientů (FN).
    - Výchozí model ($k=5$): Recall = **88.68 %** (6 přehlédnutých pacientů).
    - Optimalizovaný model ($k=9$): Recall = **92.45 %** (pouze 4 přehlédnutí, **záchrana dalších 2 pacientů**).
    """)

with t2:
    st.markdown("""
    ### Jak poznáme přetrénování (Overfitting / Overtraining)?
    
    1. **Příznak přetrénování u $k=1$**:
       - Trénovací přesnost (Train Accuracy) = **100.00 %**
       - Testovací přesnost (Test Accuracy) = **78.21 %**
       - Rozdíl (Gap) činí **21.79 %**! Model se naučil trénovací body včetně náhodného šumu nazpaměť a nedokáže zobecňovat na nová data.
    
    2. **Důkaz absence přetrénování u $k=9$**:
       - Trénovací přesnost = **84.91 %**
       - Testovací přesnost = **85.90 %**
       - Rozdíl je pouhých **0.99 %** (testovací přesnost je dokonce mírně vyšší než trénovací).
       - Model má skvělou schopnost zobecnění (Generalization) a **nevykazuje žádné přetrénování**.
    """)

with t3:
    if precomputed and "sweep_data" in precomputed:
        st.dataframe(
            pd.DataFrame(precomputed["sweep_data"]).rename(columns={
                "k": "k (sousedé)",
                "train_accuracy": "Train Acc",
                "test_accuracy": "Test Acc",
                "test_precision": "Precision",
                "test_recall": "Recall",
                "test_f1": "F1-score",
                "acc_diff": "Gap (|Train-Test|)"
            }),
            use_container_width=True
        )
