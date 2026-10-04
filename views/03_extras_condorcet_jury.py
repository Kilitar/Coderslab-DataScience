"""
Den 3 Extras: 🗳️ Condorcetova porota (Condorcet's Jury Theorem) & Ansámblové hlasování
=====================================================================================
- Matematický základ hlasovacích ansámblů (Marquis de Condorcet, 1785).
- Proč většina nezávislých modelů s p > 0.5 konverguje k 100% přesnosti.
- Proč ansámbl selhává a padá k 0 %, pokud p < 0.5.
- Vliv korelace mezi modely (ρ): reálný limit v machine learningu.
- Interaktivní simulátor a analytické křivky binomického rozdělení.
"""

from math import comb

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Condorcetova porota – Ansámbly", page_icon="🗳️", layout="wide")

st.markdown("""
# 🗳️ Condorcetova porota (Condorcet's Jury Theorem)
### Matematický důkaz, proč je kolektivní inteligence chytřejší než jednotlivec
""")

st.info("""
V roce 1785 francouzský matematik **Marquis de Condorcet** dokázal teorém, který dnes tvoří 
**matematický pilíř všech hlasovacích ansámblů v Machine Learningu** (Voting Classifier, Random Forest, Bagging). 
Tato interaktivní laboratoř ukazuje přesné podmínky, za kterých ansámbl zázračně poráží jednotlivce – a kdy naopak vede ke katastrofě.
""")

def p_majority_independent(n: int, p: float) -> float:
    """Spočítá analytickou pravděpodobnost většinového hlasování pro liché N."""
    k_min = (n // 2) + 1
    prob = sum(comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k)) for k in range(k_min, n + 1))
    return float(prob)

# --- Postranní panel: Parametry ---
with st.sidebar:
    st.header("⚙️ Parametry hlasování")
    p_single = st.slider("Přesnost 1 modelu / porotce (p):", min_value=0.30, max_value=0.95, value=0.60, step=0.01,
                         help="Pravděpodobnost, že jeden samostatný klasifikátor odpoví správně.")
    n_voters = st.slider("Počet modelů v ansámblu (N - liché číslo):", min_value=1, max_value=101, value=25, step=2)
    rho_corr = st.slider("Korelace mezi modely (ρ):", min_value=0.0, max_value=0.70, value=0.10, step=0.05,
                         help="V praxi modely nejsou zcela nezávislé – trénují se na podobných datech.")

    st.divider()
    n_sims = st.select_slider("Počet Monte Carlo simulací:", options=[1000, 5000, 10000, 20000], value=5000)

# Výpočet analytických hodnot
p_indep = p_majority_independent(n_voters, p_single)

# Monte Carlo simulace s korelací
# Generujeme společný latentní signál Z a individuální šumy E
rng = np.random.RandomState(42)
if rho_corr > 0.0:
    # Latentní model: spojitá proměnná s korelací rho
    # Y_i = sqrt(rho)*Z + sqrt(1-rho)*E_i
    z_latent = rng.normal(0, 1, size=(n_sims, 1))
    e_indiv = rng.normal(0, 1, size=(n_sims, n_voters))
    score = np.sqrt(rho_corr) * z_latent + np.sqrt(1.0 - rho_corr) * e_indiv
    # Prahování pro dosažení marginální pravděpodobnosti p_single
    from scipy.stats import norm
    threshold = norm.ppf(1.0 - p_single)
    votes = (score > threshold).astype(int)
else:
    votes = (rng.uniform(0, 1, size=(n_sims, n_voters)) < p_single).astype(int)

majority_votes = (votes.sum(axis=1) > (n_voters // 2))
p_mc_corr = float(majority_votes.mean())

# --- Horní lišta metrik ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Přesnost 1 modelu (p)", f"{p_single * 100:.1f} %")
c2.metric(f"Přesnost ansámblu ({n_voters} nezávislých)", f"{p_indep * 100:.1f} %", delta=f"{(p_indep - p_single) * 100:+.1f} %")
c3.metric(f"Přesnost při korelaci ρ = {rho_corr:.2f}", f"{p_mc_corr * 100:.1f} %", delta=f"{(p_mc_corr - p_indep) * 100:.1f} % vs nezávislé")
c4.metric("Potřebná většina hlasů", f"{(n_voters // 2) + 1} z {n_voters}")

st.divider()

# --- GRAFY ---
tab_curve, tab_mc, tab_cond = st.tabs([
    "📈 1. Analytická křivka (N vs. Úspěšnost většiny)",
    "🎲 2. Monte Carlo rozdělení hlasů",
    "📜 3. Condorcetův teorém a 3 zlatá pravidla ML",
])

with tab_curve:
    st.subheader(f"Jak roste přesnost ansámblu s velikostí N pro různé kvality modelu p")
    
    n_axis = np.arange(1, 103, 2)
    p_scenarios = [0.45, 0.51, 0.55, 0.60, 0.70]
    
    fig_curve = go.Figure()
    colors = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"]
    
    for p_val, col in zip(p_scenarios, colors):
        probs = [p_majority_independent(int(n), p_val) * 100 for n in n_axis]
        fig_curve.add_trace(go.Scatter(
            x=n_axis, y=probs, mode="lines",
            name=f"p = {p_val:.2f} ({'Podprůměr' if p_val < 0.5 else 'Lepší než náhoda'})",
            line=dict(color=col, width=2.5 if p_val != 0.45 else 3, dash="dash" if p_val < 0.5 else "solid"),
        ))
    
    # Hranice náhody 50 %
    fig_curve.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Náhoda (50 %)")
    
    fig_curve.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Počet modelů v ansámblu (N)"),
        yaxis=dict(title="Pravděpodobnost správného většinového rozhodnutí (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_curve, use_container_width=True)

with tab_mc:
    st.subheader(f"Rozdělení počtu správných hlasů v {n_sims:,} simulovaných případech")
    st.caption(f"Aktuální nastavení: N = {n_voters}, p = {p_single:.2f}, ρ = {rho_corr:.2f}")
    
    correct_counts = votes.sum(axis=1)
    df_votes = pd.DataFrame({"Spravne_hlasy": correct_counts})
    
    fig_hist = px.histogram(
        df_votes, x="Spravne_hlasy", nbins=n_voters + 1,
        color_discrete_sequence=["#6366f1"],
    )
    # Zvýraznění prahu většiny
    majority_threshold = (n_voters // 2) + 0.5
    fig_hist.add_vline(x=majority_threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Práh většiny (≥ {(n_voters // 2) + 1})")
    
    fig_hist.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=f"Počet modelů hlasujících správně (z celkem {n_voters})", dtick=max(1, n_voters // 10)),
        yaxis=dict(title="Četnost v simulaci"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_cond:
    st.markdown(r"""
    ### 📜 Formulace Condorcetova teorému poroty (1785)
    
    Mějme skupinu $N$ nezávislých voličů (nebo ML modelů), kteří rozhodují o binární otázce (Správně / Špatně). 
    Každý volič má pravděpodobnost $p$ zvolit správnou odpověď.
    
    1. **Pravidlo 1 ($p > 0.5$):** Pokud je každý model alespoň nepatrně lepší než hod mincí ($p > 0.5$), 
       pravděpodobnost, že většina rozhodne správně, je **vždy vyšší než $p$** a pro $N \to \infty$ **limitně konverguje k $1.0$ (100 %)**.
    2. **Pravidlo 2 ($p < 0.5$ – Katastrofa davu):** Pokud je model horší než náhoda ($p < 0.5$), 
       přidávání dalších takových modelů situaci **nezachrání, ale zhorší**: přesnost většiny padá limitně k **0.0 (0 %)**!
    3. **Pravidlo 3 (Předpoklad nezávislosti $\rho = 0$):** Teorém platí striktně za předpokladu nezávislosti chyb. 
       Pokud dělají modely stejné systematické chyby (korelace $\rho > 0$), ansámbl se zasekne na asymptotickém stropu.
    
    > **💡 Přímá vazba na Machine Learning:** 
    > Proto v Random Forestu náhodně subsamplujeme příznaky (`max_features`) a v Baggingu tvoříme bootstrap vzorky – **bojujeme za de-korelaci ($\rho \to 0$)**, 
    > abychom naplno využili Condorcetovu magii!
    """)
