import streamlit as st

st.title("🧭 Expertní analýza: Interaktivní průvodce výběrem modelu")
st.caption("Jak přemýšlet jako seniorní Machine Learning inženýr: Rozhodovací strom pro volbu správného regresního algoritmu podle povahy vašich dat a byznys cílů.")

st.markdown(r"""
Častá otázka studentů: *„Mám nová data. Podle čeho se mám rozhodnout, zda použít OLS, Ridge, Lasso, Polynom nebo Rozhodovací strom?“*  
Odpovězte na několik otázek o vašem projektu a algoritmus vám doporučí optimální postup:
""")

with st.container(border=True):
    q_data_type = st.radio(
        "1. Jaký charakter mají vaše prediktory (nezávislé proměnné)?",
        [
            "Převážně spojité fyzikální veličiny s plynulými závislostmi (např. teplota, koncentrace, rozměry).",
            "Směs kategorií, skokových tarifů, administrativních zón a tabulkových čísel.",
            "Obrovské množství sloupců (desítky až stovky), z nichž mnohé jsou silně korelované nebo zbytečné."
        ]
    )

    q_goal = st.radio(
        "2. Co je hlavním byznys cílem vašeho modelu?",
        [
            "100% čistá interpretovatelnost koeficientů pro management nebo regulátora (např. bankovní scoring, ekonometrie).",
            "Maximální možná predikční přesnost bez ohledu na složitost vnitřních pravidel.",
            "Automatický výběr nejdůležitějších faktorů (Feature Selection) z obřího množství proměnných."
        ]
    )

    q_extrapol = st.radio(
        "3. Budete předpovídat data ležící mimo rozsah dosud naměřených hodnot (extrapolace)?",
        [
            "Ano (např. předpověď budoucího trendu růstu, inflace nebo extrémních hodnot).",
            "Ne (nová data budou v podobném rozsahu jako dosavadní trénovací data)."
        ]
    )

st.markdown("### 🎯 Doporučená strategie pro váš projekt:")

# Vyhodnocení
if "extrapolace" in q_extrapol and "Ano" in q_extrapol:
    st.error(
        "🚫 **STOP: Zákaz použití Rozhodovacích stromů!**  \n"
        "Rozhodovací stromy (CART) ani lesy (Random Forest) **neumí extrapolovat**. Mimo rozsah trénovacích dat narazí na konstantní strop.  \n"
        "👉 **Doporučení:** Použijte **Lineární regresi s L2 regularizací (Ridge)** nebo polynomiální trend nízkého stupně."
    )
elif "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal:
    st.success(
        "🏆 **Doporučený model: Lasso regrese (L1) nebo Elastic Net**  \n"
        "Díky L1 penalizaci model automaticky vynuluje šumové a redundantní příznaky a zanechá pouze klíčové tahouny.  \n"
        "💡 **Pozor:** Nezapomeňte před trénováním data normalizovat pomocí `StandardScaler`!"
    )
elif "Směs kategorií" in q_data_type or "Maximální možná" in q_goal:
    st.success(
        "🏆 **Doporučený model: Rozhodovací strom (CART) s prořezáním (nebo Random Forest v Dni 2)**  \n"
        "Stromy excelují na tabulkových datech, nevyžadují škálování a perfektně zvládají pravoúhlé ohraničení zón a skokové tarify.  \n"
        "💡 **Pozor:** Vždy ladit `max_depth` a `min_samples_leaf`, aby nedošlo k gigantickému přeučení!"
    )
elif "interpretovatelnost" in q_goal:
    st.success(
        "🏆 **Doporučený model: OLS Lineární regrese nebo Ridge regrese**  \n"
        "Každý koeficient beta má přímou ekonomickou interpretaci (*'při zvýšení plochy o 1 m² vzroste cena o X Kč'*).  \n"
        "Pokud jsou prediktory korelované, přidejte lehkou L2 penalizaci (Ridge)."
    )
else:
    st.info(
        "🏆 **Doporučený postup: Začněte OLS přímkou jako baselinou a porovnejte s prořezaným stromem.**"
    )

st.markdown("---")
st.markdown("### 🗺️ Interaktivní mapa rozhodovacího stromu výběru:")
st.markdown("Níže je přehledný vizuální diagram celého rozhodovacího procesu. Aktuální cesta podle vaší volby je **zvýrazněna zeleně**:")

# Dynamické určení aktivního uzlu pro vizualizaci
active_choice = "ols"
if "extrapolace" in q_extrapol and "Ano" in q_extrapol:
    active_choice = "ridge"
elif "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal:
    active_choice = "lasso"
elif "Směs kategorií" in q_data_type or "Maximální možná" in q_goal:
    active_choice = "tree"
elif "interpretovatelnost" in q_goal:
    active_choice = "ols"

import plotly.graph_objects as go

# Souřadnice uzlů
# Úroveň 0: Kořen
# Úroveň 1: Extrapolace vs No
# Úroveň 2: Nelinearita vs Mnoho sloupců
# Úroveň 3: Výsledné modely
nodes = {
    "root": {"x": 0.5, "y": 1.0, "label": "<b>Tabulková data?</b><br>Máme tabulku s čísly/kategoriemi", "color": "#334155", "border": "#64748B"},
    "extrapol": {"x": 0.35, "y": 0.72, "label": "<b>Extrapolace do budoucna?</b><br>Bude se předpovídat mimo rozsah?", "color": "#334155", "border": "#64748B"},
    "deep": {"x": 0.85, "y": 0.72, "label": "<b>Nestrukturovaná data</b><br>Text / Obrázky -> Deep Learning", "color": "#1E293B", "border": "#475569"},
    "ridge": {"x": 0.15, "y": 0.40, "label": "🏆 <b>Ridge regrese (L2)</b><br>nebo lineární trend<br><i>(Stromy zakázány!)</i>", "color": "#065F46" if active_choice == "ridge" else "#1E293B", "border": "#10B981" if active_choice == "ridge" else "#334155"},
    "nonlin": {"x": 0.55, "y": 0.42, "label": "<b>Nelinearity & Skoky?</b><br>Tarify, zóny, zlomy v datech?", "color": "#334155", "border": "#64748B"},
    "tree": {"x": 0.38, "y": 0.12, "label": "🏆 <b>Rozhodovací strom (CART)</b><br>nebo Random Forest<br><i>(Nevyžaduje škálování)</i>", "color": "#065F46" if active_choice == "tree" else "#1E293B", "border": "#10B981" if active_choice == "tree" else "#334155"},
    "cols": {"x": 0.72, "y": 0.22, "label": "<b>Mnoho sloupců / Šum?</b><br>Desítky až stovky proměnných?", "color": "#334155", "border": "#64748B"},
    "lasso": {"x": 0.62, "y": 0.02, "label": "🏆 <b>Lasso (L1) / Elastic Net</b><br>Automatická selekce příznaků", "color": "#065F46" if active_choice == "lasso" else "#1E293B", "border": "#10B981" if active_choice == "lasso" else "#334155"},
    "ols": {"x": 0.88, "y": 0.02, "label": "🏆 <b>OLS Lineární regrese</b><br>Maximální interpretovatelnost", "color": "#065F46" if active_choice == "ols" else "#1E293B", "border": "#10B981" if active_choice == "ols" else "#334155"}
}

# Hrany (spojnice)
edges = [
    ("root", "extrapol", "ANO (Tabulka)"),
    ("root", "deep", "NE (Text/Foto)"),
    ("extrapol", "ridge", "ANO (Trend mimo data)"),
    ("extrapol", "nonlin", "NE (V rámci dat)"),
    ("nonlin", "tree", "ANO (Skoky/Zóny)"),
    ("nonlin", "cols", "NE (Hladké/Lineární)"),
    ("cols", "lasso", "ANO (Mnoho šumu)"),
    ("cols", "ols", "NE (Čisté málo sloupců)")
]

fig_flow = go.Figure()

# Kreslení hran
for src, dst, txt in edges:
    x0, y0 = nodes[src]["x"], nodes[src]["y"]
    x1, y1 = nodes[dst]["x"], nodes[dst]["y"]
    fig_flow.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode="lines",
        line=dict(color="#475569", width=2),
        hoverinfo="none",
        showlegend=False
    ))
    # Popisek hrany
    fig_flow.add_annotation(
        x=(x0 + x1) / 2, y=(y0 + y1) / 2,
        text=txt,
        showarrow=False,
        font=dict(size=10, color="#94A3B8"),
        bgcolor="#0F172A",
        bordercolor="#334155",
        borderwidth=1,
        borderpad=3
    )

# Kreslení uzlů jako boxů s anotacemi
for k, n in nodes.items():
    fig_flow.add_annotation(
        x=n["x"], y=n["y"],
        text=n["label"],
        showarrow=False,
        font=dict(size=11, color="#FFFFFF"),
        bgcolor=n["color"],
        bordercolor=n["border"],
        borderwidth=2.5 if "🏆" in n["label"] and ("#10B981" in n["border"] or "#065F46" in n["color"]) else 1.5,
        borderpad=8,
        align="center"
    )

fig_flow.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.02, 1.02]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.08, 1.08]),
    height=580,
    margin=dict(l=10, r=10, t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_flow, width="stretch")

