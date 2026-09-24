import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

st.title("🔬 Expertní analýza: Multiclass metriky a úskalí evaluace tučňáků")
st.caption(
    "Pokročilá dekonstrukce vyhodnocení více-třídní klasifikace (Adelie, Chinstrap, Gentoo): "
    "One-vs-Rest (OvR) ROC & PR křivky, důkaz rovnosti Micro F1 = Accuracy, diskretizační patologie k-NN Log-Lossu "
    "a ochranářská matice asymetrických nákladů."
)

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
norm_csv_path = data_dir / "penguins_df_normalized.csv"
if not norm_csv_path.exists():
    norm_csv_path = base_dir / "02_Classification" / "penguins_df_normalized.csv"


@st.cache_data
def load_penguins_critique_data(csv_str: str):
    df = pd.read_csv(csv_str)
    X = df.drop("species", axis=1)
    y = df["species"]
    species_names = ["Adelie", "Chinstrap", "Gentoo"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return df, X, y, species_names, X_train, X_test, y_train, y_test


df, X, y, species_names, X_train, X_test, y_train, y_test = load_penguins_critique_data(str(norm_csv_path))

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 1. Multiclass OvR ROC & PR křivky",
    "⚖️ 2. Macro vs. Micro vs. Weighted F1",
    "📉 3. Log-Loss & Kalibrace k-NN",
    "🐧 4. Ochranářská matice nákladů",
])

# ==============================================================================
# TAB 1: ONE-VS-REST (OvR) ROC & PR KŘIVKY
# ==============================================================================
with tab1:
    st.subheader("One-vs-Rest (OvR) ROC křivky pro jednotlivé druhy")
    st.markdown(
        """
        V binární klasifikaci máme jednu ROC křivku a jednu metriku AUC. Vícetřídní klasifikace ($C = 3$) 
        vyžaduje dekompozici problému na **3 nezávislé binární úlohy** metodou **One-vs-Rest (OvR)**:
        * **Třída 0 (Adelie):** Adelie vs. (Chinstrap + Gentoo)
        * **Třída 1 (Chinstrap):** Chinstrap vs. (Adelie + Gentoo)
        * **Třída 2 (Gentoo):** Gentoo vs. (Adelie + Chinstrap)
        """
    )

    k_sel = st.slider("Zvolte hyperparametr k pro ROC analýzu:", min_value=1, max_value=25, value=5, step=2, key="ovr_k")
    weights_sel = st.radio("Váhová funkce pro sousedy:", ["uniform", "distance"], horizontal=True, key="ovr_weights")

    clf_ovr = KNeighborsClassifier(n_neighbors=k_sel, weights=weights_sel)
    clf_ovr.fit(X_train, y_train)

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = clf_ovr.predict_proba(X_test)

    # Calculate ROC and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, sp in enumerate(species_names):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3.0
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("AUC: Adelie", f"{roc_auc[0]:.4f}")
    col_m2.metric("AUC: Chinstrap", f"{roc_auc[1]:.4f}")
    col_m3.metric("AUC: Gentoo", f"{roc_auc[2]:.4f}")
    col_m4.metric("AUC: Macro Průměr", f"{roc_auc['macro']:.4f}")

    fig_roc = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, sp in enumerate(species_names):
        fig_roc.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            mode="lines+markers",
            name=f"{sp} (AUC = {roc_auc[i]:.3f})",
            line=dict(color=colors[i], width=2.5),
        ))

    fig_roc.add_trace(go.Scatter(
        x=fpr["macro"], y=tpr["macro"],
        mode="lines",
        name=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Náhodný klasifikátor",
        line=dict(color="gray", width=1, dash="dot"),
    ))

    fig_roc.update_layout(
        title=f"One-vs-Rest ROC Křivky (k={k_sel}, váhy={weights_sel})",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR / Senzitivita)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
        legend=dict(x=0.55, y=0.15),
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_roc, width="stretch")

    st.info(
        """
        💡 **Klíčové expertní zjištění z OvR ROC:**
        1. **Gentoo je téměř perfektně oddělitelný ($AUC \approx 1.0$):** V morfologickém prostoru tvoří samostatný izolovaný ostrov. 
           Jeho tělesná hmotnost a délka ploutví jsou natolik signifikantní, že k-NN u něj prakticky nechybuje.
        2. **Hraniční záměny Adelie vs. Chinstrap:** Křivky pro Adelie a Chinstrap vykazují drobný pokles AUC ($0.97 - 0.99$). 
           Oba druhy sdílejí podobnou tělesnou konstituci a bez specifických příznaků (poměr zobáku) se v normovaném L2 prostoru částečně prolínají.
        """
    )

# ==============================================================================
# TAB 2: MACRO VS. MICRO VS. WEIGHTED F1
# ==============================================================================
with tab2:
    st.subheader("Matematická dekonstrukce průměrování metrik")
    st.markdown(
        """
        V reálném data science reportingu je fatální chybou napsat: *"Náš model dosáhl F1-skóre 97 %."*
        V multiclass klasifikaci **neexistuje jedno F1-skóre** — existují minimálně 3 fundamentálně odlišné způsoby agregace:
        """
    )

    st.latex(
        r"\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}, \quad "
        r"\text{Weighted } F_1 = \sum_{c=1}^C \left(\frac{N_c}{N}\right) F_{1, c}, \quad "
        r"\text{Micro } F_1 = \frac{2 \cdot \sum TP_c}{2 \cdot \sum TP_c + \sum FP_c + \sum FN_c}"
    )

    st.markdown(
        r"""
        ### 🔍 Teoretický důkaz: Proč Micro $F_1 \equiv \text{Accuracy}$?
        V uzavřené jednolabelové multiclass klasifikaci (každý vzorek má přesně 1 predikovanou a 1 pravou třídu) platí:
        $$\sum_{c=1}^C FP_c = \sum_{c=1}^C FN_c = \text{Celkový počet chybných predikcí}$$
        $$\sum_{c=1}^C TP_c = \text{Celkový počet správných predikcí}$$
        
        Dosadíme-li do vzorce pro Micro Precision a Micro Recall:
        $$\text{Micro Precision} = \frac{\sum TP_c}{\sum TP_c + \sum FP_c} = \frac{\text{Správné}}{N} = \text{Accuracy}$$
        $$\text{Micro Recall} = \frac{\sum TP_c}{\sum TP_c + \sum FN_c} = \frac{\text{Správné}}{N} = \text{Accuracy}$$
        
        Protože $\text{Micro Precision} = \text{Micro Recall} = \text{Accuracy}$, harmonický průměr dvou identických čísel je rovněž roven tomuto číslu:
        $$\mathbf{\text{Micro } F_1 \equiv \text{Accuracy}}$$
        *Metrika Micro $F_1$ tedy nepřináší žádnou novou informaci oproti prosté přesnosti!*
        """
    )

    st.divider()

    st.subheader("Interaktivní simulátor: Zranitelnost Weighted F1 vůči menšinové třídě")
    st.markdown(
        r"""
        V našem datasetu je zastoupení tříd asymetrické:
        * **Adelie:** 152 jedinců (~44 %)
        * **Gentoo:** 124 jedinců (~36 %)
        * **Chinstrap:** pouhých 68 jedinců (~20 %) $\leftarrow$ *menšinová třída*
        
        Co se stane, pokud model zkolabuje na menšinové třídě Chinstrap?
        """
    )

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        f1_ade = st.slider("F1 pro Adelie:", 0.0, 1.0, 0.98, 0.01, key="sim_f1_ade")
        f1_gen = st.slider("F1 pro Gentoo:", 0.0, 1.0, 0.99, 0.01, key="sim_f1_gen")
    with sim_col2:
        f1_chin = st.slider("F1 pro Chinstrap (menšina):", 0.0, 1.0, 0.50, 0.01, key="sim_f1_chin")

    n_ade, n_gen, n_chin = 152, 124, 68
    n_tot = n_ade + n_gen + n_chin

    macro_res = (f1_ade + f1_gen + f1_chin) / 3.0
    weighted_res = (n_ade * f1_ade + n_gen * f1_gen + n_chin * f1_chin) / n_tot

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Macro F1", f"{macro_res:.3%}", delta=f"{macro_res - 0.97:.1%}")
    res_c2.metric("Weighted F1", f"{weighted_res:.3%}", delta=f"{weighted_res - 0.97:.1%}")
    res_c3.metric("Zkreslení (Weighted - Macro)", f"{(weighted_res - macro_res):.2%}")

    fig_sim = go.Figure(data=[
        go.Bar(name="Skóre", x=["Macro F1", "Weighted F1"], y=[macro_res, weighted_res],
               marker_color=["#ef553b", "#00cc96"], text=[f"{macro_res:.1%}", f"{weighted_res:.1%}"], textposition="auto")
    ])
    fig_sim.update_layout(yaxis=dict(range=[0, 1.05]), height=320, template="plotly_white", title="Srovnání zkreslení při selhání na Chinstrapovi")
    st.plotly_chart(fig_sim, width="stretch")

    st.warning(
        """
        ⚠️ **Doporučení pro produkci:** Pokud optimalizujete model na nebalancovaných datech, 
        **nikdy nespoléhejte na Weighted F1**. Weighted F1 dává manažerům falešný pocit bezpečí, 
        protože vysoká přesnost na majoritě (Adelie) zamaskuje kritická selhání na vzácné třídě (Chinstrap).
        Vždy sledujte **Macro F1** nebo jednotlivá dílčí skóre pro každou třídu.
        """
    )

# ==============================================================================
# TAB 3: LOG-LOSS & KALIBRACE k-NN
# ==============================================================================
with tab3:
    st.subheader("Multiclass Log-Loss (Křížová entropie) a patologie k-NN")
    st.markdown(
        r"""
        Zatímco diskrétní metriky (Accuracy, F1) se dívají pouze na to, zda model trefil správný label $\hat{y} = \arg\max p$, 
        **Log-Loss** penalizuje model na základě **jistoty (kalibrace pravděpodobností)**:
        """
    )

    st.latex(
        r"L_{\log}(Y, P) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i, c} \ln(p_{i, c})"
    )

    st.markdown(
        r"""
        ### 🧨 Patologie k-NN s uniformními vahami:
        Standardní k-NN počítá pravděpodobnost třídy $c$ jako prostý podíl sousedů: $p_c = \frac{k_c}{k}$.
        1. **Hrubá diskretizace:** Pro $k=5$ může $p_c$ nabývat pouze hodnot z množiny $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.
        2. **Fatální divergence:** Pokud všech 5 sousedů patří do třídy Adelie ($p_{\text{Adelie}} = 1.0, p_{\text{Chinstrap}} = 0.0$), 
           ale testovaný vzorek je ve skutečnosti Chinstrap, pak:
           $$L_{\log} \propto -\ln(p_{\text{Chinstrap}}) = -\ln(0.0) \to +\infty$$
           V knihovně `scikit-learn` je pravděpodobnost ořezána na $\varepsilon = 10^{-15}$, což dává penalizaci $-\ln(10^{-15}) \approx 34.5$ pro jediný vzorek!
        """
    )

    st.write("#### Porovnání Log-Loss v závislosti na $k$ a váhové funkci:")

    k_vals = list(range(1, 31, 2))
    log_losses_uniform = []
    log_losses_distance = []

    for k_i in k_vals:
        clf_u = KNeighborsClassifier(n_neighbors=k_i, weights="uniform").fit(X_train, y_train)
        clf_d = KNeighborsClassifier(n_neighbors=k_i, weights="distance").fit(X_train, y_train)

        prob_u = clf_u.predict_proba(X_test)
        prob_d = clf_d.predict_proba(X_test)

        log_losses_uniform.append(log_loss(y_test, prob_u, labels=[0, 1, 2]))
        log_losses_distance.append(log_loss(y_test, prob_d, labels=[0, 1, 2]))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=k_vals, y=log_losses_uniform,
        mode="lines+markers",
        name="Uniformní váhy (diskrétní p)",
        line=dict(color="#ef553b", width=2.5),
    ))
    fig_loss.add_trace(go.Scatter(
        x=k_vals, y=log_losses_distance,
        mode="lines+markers",
        name="Vážené vzdáleností (spojité p)",
        line=dict(color="#00cc96", width=2.5),
    ))
    fig_loss.update_layout(
        title="Multiclass Log-Loss na testovací sadě vs. k",
        xaxis_title="Hyperparametr k",
        yaxis_title="Log-Loss (menší = lepší)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_loss, width="stretch")

    st.info(
        """
        🎯 **Pozorování:** 
        * Při malém $k$ (např. $k=1$ nebo $k=3$) je Log-Loss obrovský, protože model je přehnaně sebevědomý (overconfident) 
          a jakákoli chyba v predikci vystřelí křížovou entropii do stropu.
        * S rostoucím $k$ dochází k přirozenému hlazení pravděpodobností (tzv. *softening*), čímž se Log-Loss dramaticky stabilizuje.
        * `weights='distance'` poskytuje jemnější a lépe kalibrované rozdělení pravděpodobností.
        """
    )

# ==============================================================================
# TAB 4: OCHRANÁŘSKÁ MATICE NÁKLADŮ
# ==============================================================================
with tab4:
    st.subheader("Asymetrické ekologické náklady a ochrana biodiverzity")
    st.markdown(
        """
        V reálné biologické praxi nemají všechny chyby stejnou váhu:
        * **Chinstrap (Tučňák uzdičkový)** je v mnoha koloniích ohrožen úbytkem krilu a klimatickou změnou. 
          Pokud Chinstrapa chybně klasifikujeme jako běžného Adelie, **nedostane satelitní GPS vysílač** 
          a terénní tým ztratí monitoring jeho hnízdiště (kritická ekologická ztráta).
        * Naopak, pokud Adelie označíme za Chinstrapa, tým pouze zbytečně nasadí vysílač běžnému tučňákovi (malá finanční ztráta materiálu).
        """
    )

    st.write("#### Nastavte relativní sankce za záměnu párů:")

    col_cost1, col_cost2, col_cost3 = st.columns(3)
    with col_cost1:
        cost_chin_as_ade = st.number_input("Chinstrap $\\to$ Adelie (Ztráta monitoringu)", value=1000, step=100)
        cost_gen_as_ade = st.number_input("Gentoo $\\to$ Adelie", value=200, step=50)
    with col_cost2:
        cost_ade_as_chin = st.number_input("Adelie $\\to$ Chinstrap (Zbytečný vysílač)", value=150, step=50)
        cost_gen_as_chin = st.number_input("Gentoo $\\to$ Chinstrap", value=250, step=50)
    with col_cost3:
        cost_ade_as_gen = st.number_input("Adelie $\\to$ Gentoo", value=100, step=50)
        cost_chin_as_gen = st.number_input("Chinstrap $\\to$ Gentoo", value=800, step=100)

    # Cost Matrix: rows = True, cols = Pred
    # [ [0, C(0->1), C(0->2)],
    #   [C(1->0), 0, C(1->2)],
    #   [C(2->0), C(2->1), 0] ]
    cost_matrix = np.array([
        [0, cost_ade_as_chin, cost_ade_as_gen],
        [cost_chin_as_ade, 0, cost_chin_as_gen],
        [cost_gen_as_ade, cost_gen_as_chin, 0],
    ])

    clf_curr = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_pred_curr = clf_curr.predict(X_test)
    cm_curr = confusion_matrix(y_test, y_pred_curr, labels=[0, 1, 2])

    total_cost = np.sum(cm_curr * cost_matrix)

    st.write("#### Výpočet celkových nákladů na testovacím vzorku:")
    res_cost_col1, res_cost_col2, res_cost_col3 = st.columns(3)
    res_cost_col1.metric("Celková přesnost", f"{accuracy_score(y_test, y_pred_curr):.2%}")
    res_cost_col2.metric("Chybně určení Chinstrapi", f"{cm_curr[1, 0] + cm_curr[1, 2]} ks")
    res_cost_col3.metric("Celková finančně-ekologická škoda", f"{total_cost:,.0f} €")

    # Interactive simulation across k values
    costs_across_k = []
    ks_test = list(range(1, 25))
    for k_val in ks_test:
        c_k = KNeighborsClassifier(n_neighbors=k_val).fit(X_train, y_train)
        pred_k = c_k.predict(X_test)
        cm_k = confusion_matrix(y_test, pred_k, labels=[0, 1, 2])
        costs_across_k.append(np.sum(cm_k * cost_matrix))

    best_k_idx = int(np.argmin(costs_across_k))
    best_k_cost = ks_test[best_k_idx]

    fig_cost_k = go.Figure()
    fig_cost_k.add_trace(go.Scatter(
        x=ks_test, y=costs_across_k,
        mode="lines+markers",
        name="Celkové náklady",
        line=dict(color="#d62728", width=2.5),
    ))
    fig_cost_k.add_vline(
        x=best_k_cost,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"Optimum k={best_k_cost} ({costs_across_k[best_k_idx]:,.0f} €)",
    )
    fig_cost_k.update_layout(
        title="Křivka celkových nákladů (Cost Curve) v závislosti na k",
        xaxis_title="Hyperparametr k",
        yaxis_title="Celková škoda (€)",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_cost_k, width="stretch")

    st.success(
        f"🎯 **Závěr Cost-Benefit analýzy:** Při současně zadaných ekologických nákladech je optimální "
        f"zvolit **$k = {best_k_cost}$**, které minimalizuje celkovou penalizaci na **{costs_across_k[best_k_idx]:,.0f} €**. "
        f"Klasický přístup sledující pouze prostou Accuracy by tuto asymetrii zcela ignoroval!"
    )
