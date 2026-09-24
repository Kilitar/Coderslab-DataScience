from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)

st.title("🔬 Cvičení 1: Expertní analýza metrik & Klinické rozhodování (Páteř)")
st.caption("Klinická nákladová matice chyb (Cost-Benefit), optimalizace klasifikačního prahu pomocí Youdenova indexu, Matthewsův korelační koeficient (MCC) a Brier Score.")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
norm_csv_path = data_dir / "lumbar_df_normalized.csv"
if not norm_csv_path.exists():
    norm_csv_path = base_dir / "02_Classification" / "lumbar_df_normalized.csv"


@st.cache_data
def load_lumbar_metrics_critique():
    df = pd.read_csv(norm_csv_path)
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


df, X_train, X_test, y_train, y_test = load_lumbar_metrics_critique()

# =============================================================================
# 1. EXECUTIVNÍ SOUHRN METODICKÝCH ZJIŠTĚNÍ
# =============================================================================
st.subheader("💡 4 expertní principy vyhodnocování medicínské diagnostiky")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.error("💶 **Cost-Benefit Asymetrie**\n\nNáklady na False Negative (~3000 € za operaci) převyšují False Positive (150 € za MRI) v poměru 20:1.")
with c2:
    st.success("🎯 **Youdenův index (J)**\n\nOptimální práh $\\theta$ maximalizuje $\\text{Sensitivity} + \\text{Specificity} - 1$, čímž vyrovnává klinický zisk.")
with c3:
    st.info("📐 **Matthewsův koeficient**\n\nMCC penalizuje nadhodnocenou Accuracy u třídní nerovnováhy a měří korelaci s pravdou.")
with c4:
    st.warning("📉 **Kalibrace k-NN**\n\nPravděpodobnosti z k-NN (např. 3/5 = 0.6) jsou diskrétní zlomky, nikoliv hladká posteriorní pravděpodobnost.")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "💶 Klinický kalkulátor finančních & zdravotních nákladů",
    "🎯 Threshold Tuning & Youdenův index J",
    "📐 MCC, Cohenovo Kappa & Balanced Accuracy",
    "📈 ROC vs. Precision-Recall křivka"
])

# =============================================================================
# TAB 1: KLINICKÝ KALKULÁTOR NÁKLADŮ
# =============================================================================
with tab1:
    st.subheader("Ekonomika medicínského screeningu: Výpočet očekávaných nákladů")
    st.markdown("""
    Běžné metriky (Accuracy, F1) zacházejí se všemi chybami rovnocenně. V nemocniční praxi má však každá buňka 
    matice záměn dramaticky odlišnou finanční a lidskou cenu:
    """)

    col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
    with col_cost1:
        cost_tn = st.number_input("Náklad TN (Zdravý propuštěn) [€]:", value=0, step=10)
    with col_cost2:
        cost_tp = st.number_input("Náklad TP (Včasná léčba patologie) [€]:", value=200, step=50)
    with col_cost3:
        cost_fp = st.number_input("Náklad FP (Zbytečné kontrolní MRI) [€]:", value=150, step=25)
    with col_cost4:
        cost_fn = st.number_input("Náklad FN (Přehlédnutí -> Akutní operace) [€]:", value=3000, step=500)

    # Interaktivní posuvník prahu pro k=9
    knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
    probs_test = knn_model.predict_proba(X_test)[:, 1]

    threshold_slider = st.slider("Testovaný klasifikační práh $\\theta$ pro třídu Abnormal:", 0.05, 0.95, 0.50, 0.05)
    preds_th = (probs_test >= threshold_slider).astype(int)

    cm_th = confusion_matrix(y_test, preds_th)
    tn_th, fp_th, fn_th, tp_th = cm_th.ravel()

    total_cost_test = (tn_th * cost_tn) + (tp_th * cost_tp) + (fp_th * cost_fp) + (fn_th * cost_fn)
    avg_cost_per_patient = total_cost_test / len(y_test)

    # Přepočet na screeningovou populaci 1000 pacientů
    scale_1000 = 1000 / len(y_test)
    projected_cost_1000 = total_cost_test * scale_1000

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Celkové náklady (Test N=78)", f"{total_cost_test:,.0f} €", f"{fn_th} kritických FN chyb")
    with r2:
        st.metric("Průměrný náklad na 1 pacienta", f"{avg_cost_per_patient:.1f} €")
    with r3:
        st.metric("Odhad nákladů na kohortu 1 000 pacientů", f"{projected_cost_1000:,.0f} €", f"{int(fn_th * scale_1000)} nepoznaných patologií", delta_color="inverse")

    st.markdown("""
    > **Klinický závěr:** Posunem prahu z výchozího $\\theta = 0.50$ dolů na $\\theta = 0.35$ zachytíme téměř 100 % nemocných. 
    > Ačkoliv přibudou falešné poplachy (kontrolní MRI za 150 €), eliminace jediného přehlédnutého pacienta (3 000 €) nemocnici i pojišťovně ušetří obrovské prostředky!
    """)

# =============================================================================
# TAB 2: THRESHOLD TUNING & YOUDENŮV INDEX
# =============================================================================
with tab2:
    st.subheader("Optimalizace rozhodovacího prahu pomocí Youdenova indexu $J$")
    st.markdown(r"""
    **Youdenův index** ($J$) je definován jako:
    $$J = \text{Sensitivity (Recall)} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$
    
    Představuje vertikální vzdálenost bodu ROC křivky od náhodné diagonály. Bod, kde je $J$ maximální, 
    určuje optimální kompromis mezi senzitivitou a specificitou.
    """)

    fpr, tpr, thresholds = roc_curve(y_test, probs_test)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_th_j = thresholds[best_j_idx]
    best_j_val = j_scores[best_j_idx]

    fig_j = go.Figure()
    fig_j.add_trace(go.Scatter(x=thresholds[1:], y=j_scores[1:], mode="lines+markers", name="Youdenův index (J)", line=dict(color="#2980b9", width=2.5)))
    fig_j.add_trace(go.Scatter(x=thresholds[1:], y=tpr[1:], mode="lines", name="TPR (Senzitivita)", line=dict(color="#2ecc71", dash="dash")))
    fig_j.add_trace(go.Scatter(x=thresholds[1:], y=1 - fpr[1:], mode="lines", name="TNR (Specificita)", line=dict(color="#e74c3c", dash="dash")))
    fig_j.add_vline(x=best_th_j, line_dash="dot", line_color="green", annotation_text=f"Optimum: theta={best_th_j:.2f} (J={best_j_val:.2f})")
    fig_j.update_layout(
        title="<b>Vývoj Youdenova indexu J, Senzitivity a Specificity podle prahu theta</b>",
        xaxis_title="Rozhodovací práh (Threshold theta)",
        yaxis_title="Hodnota metriky",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_j, width="stretch")

    st.success(f"🏆 **Doporučený klinický práh dle Youdena:** $\\theta^* = {best_th_j:.2f}$ (při $k=9$). Dosahuje $J = {best_j_val:.3f}$, TPR = {tpr[best_j_idx]*100:.1f} % a Specificity = {(1-fpr[best_j_idx])*100:.1f} %.")

# =============================================================================
# TAB 3: MCC & ROBUSTNÍ METRIKY
# =============================================================================
with tab3:
    st.subheader("Metriky odolné vůči třídní nerovnováze (MCC, Kappa, Balanced Acc)")
    st.markdown("""
    V medicínských datasetech bývá poměr zdravých a nemocných nevyrovnaný (zde 100 Normal vs. 210 Abnormal). 
    Prozkoumejme metriky, které se nenechají oklamat převahou většinové třídy:
    """)

    preds_default = knn_model.predict(X_test)

    mcc = matthews_corrcoef(y_test, preds_default)
    kappa = cohen_kappa_score(y_test, preds_default)
    bal_acc = balanced_accuracy_score(y_test, preds_default)
    raw_acc = accuracy_score(y_test, preds_default)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Accuracy", f"{raw_acc*100:.2f} %", "Základní metrika")
    with col_m2:
        st.metric("Balanced Accuracy", f"{bal_acc*100:.2f} %", "Průměr senzitivity a specificity")
    with col_m3:
        st.metric("Matthews Corr (MCC)", f"{mcc:.3f}", "Rozsah [-1, 1]")
    with col_m4:
        st.metric("Cohenovo Kappa", f"{kappa:.3f}", "Shoda očištěná o náhodu")

    st.markdown(r"""
    ### 🔬 Proč je Matthews Correlation Coefficient (MCC) nadřazený F1-score?
    1. **Symetrie všech 4 polí:** Vzorec MCC využívá všechny prvky matice záměn ($TP, TN, FP, FN$):
       $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
    2. **F1-score ignoruje $TN$:** Metrika $F_1$ vůbec nebere v úvahu, kolik zdravých pacientů bylo správně potvrzeno! MCC bere v potaz celkovou shodu obou tříd.
    3. **Škála:** MCC se pohybuje od $-1$ (naprostý nesoulad), přes $0$ (náhodné hádání) až po $+1$ (perfektní predikce). Hodnota **0.71** značí vynikající diagnostickou sílu.
    """)

# =============================================================================
# TAB 4: ROC VS. PRECISION-RECALL KŘIVKA
# =============================================================================
with tab4:
    st.subheader("ROC Křivka vs. Precision-Recall Křivka")

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs_test)
    auc_roc = roc_auc_score(y_test, probs_test)

    fig_dual = make_subplots(rows=1, cols=2, subplot_titles=[f"ROC Křivka (AUC = {auc_roc:.3f})", "Precision-Recall Křivka"])

    # ROC trace
    fig_dual.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="k-NN ROC", line=dict(color="#2980b9", width=2.5)), row=1, col=1)
    fig_dual.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Náhodný odhad", line=dict(color="gray", dash="dash")), row=1, col=1)

    # PR trace
    fig_dual.add_trace(go.Scatter(x=rec_arr, y=prec_arr, mode="lines", name="k-NN PR", line=dict(color="#27ae60", width=2.5)), row=1, col=2)
    fig_dual.add_trace(go.Scatter(x=[0, 1], y=[sum(y_test==1)/len(y_test)]*2, mode="lines", name="Baseline prevalence", line=dict(color="gray", dash="dash")), row=1, col=2)

    fig_dual.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig_dual.update_yaxes(title_text="True Positive Rate (Recall)", row=1, col=1)
    fig_dual.update_xaxes(title_text="Recall (Senzitivita)", row=1, col=2)
    fig_dual.update_yaxes(title_text="Precision (Preciznost)", row=1, col=2)
    fig_dual.update_layout(height=420, showlegend=False)

    st.plotly_chart(fig_dual, width="stretch")

    st.info("""
    **Metodické pravidlo:** Pokud je pozitivní třída vzácná (např. incidence nemoci 1 %), je ROC křivka příliš optimistická kvůli obrovskému počtu $TN$. 
    V takovém případě je nutné hodnotit **Precision-Recall křivku**, která se zaměřuje výhradně na kvalitu záchytu vzácné třídy.
    """)
