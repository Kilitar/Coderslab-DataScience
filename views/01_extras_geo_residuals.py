from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("🗺️ Expertní analýza: Geografická mapa reziduí Seattlu")
st.caption("Proč OLS selhává a Rozhodovací strom exceluje: Vizuální inspekce chyb predikce v reálných GPS souřadnicích King County.")

@st.cache_data
def load_geo_data():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "01_Regression" / "data" / "day1_extras_precomputed.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["geo_sample"])

df_geo = load_geo_data()

st.markdown(r"""
V Cvičení 1 (OLS) dosáhla lineární regrese $R^2 \approx 0.70$.  
V Cvičení 8 (Rozhodovací strom) stouplo skóre na $R^2 \approx 0.795$ a chyba RMSE klesla o **35 000 USD**.  
Tato interaktivní mapa odhaluje **přesnou příčinu**: Kde konkrétně v oblasti Seattlu a Lake Washington modely chybují?
""")

sel_model = st.radio(
    "Zvolte model pro zobrazení reziduí:",
    ["Lineární regrese (OLS)", "Optimální rozhodovací strom (CART)"],
    horizontal=True
)

res_col = "pct_ols" if "OLS" in sel_model else "pct_tree"
abs_col = "res_ols" if "OLS" in sel_model else "res_tree"

# Omezení škály pro čitelnost barevné mapy (-50 % až +50 %)
df_geo["clipped_err"] = df_geo[res_col].clip(-50, 50)

fig_map = px.scatter(
    df_geo,
    x="long",
    y="lat",
    color="clipped_err",
    color_continuous_scale="RdBu_r", # Červená = model podhodnotil (drahý dům), Modrá = nadhodnotil
    size="sqft_living",
    size_max=12,
    hover_data={
        "price": ":,.0f",
        "sqft_living": ":,.0f",
        "grade": True,
        abs_col: ":,.0f",
        res_col: ":.1f",
        "clipped_err": False,
        "lat": False,
        "long": False
    },
    labels={
        "clipped_err": "Chyba (%)",
        "long": "Zeměpisná délka",
        "lat": "Zeměpisná šířka"
    },
    title=f"Geografické rozložení relativních chyb reziduí: {sel_model}"
)

fig_map.update_layout(
    height=600,
    coloraxis_colorbar=dict(
        title="Chyba predikce",
        ticksuffix=" %",
        tickvals=[-50, -25, 0, 25, 50],
        ticktext=["<-50% (Podhodnoceno)", "-25%", "0% (Přesná)", "+25%", ">+50% (Nadhodnoceno)"]
    )
)
st.plotly_chart(fig_map, width="stretch")

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("""
    #### 🔴 Červené shluky (Model hrubě podhodnotil cenu)
    - Koncentrují se v **severní části Seattlu, na poloostrově Bellevue a podél pobřeží jezera**.
    - **U Lineární regrese:** OLS předpokládá globální přímku a nedokáže pochopit, že stejný dům o 150 m² má v centru Seattlu trojnásobnou hodnotu oproti jihu.
    """)
with col_info2:
    st.markdown("""
    #### 🟢 / ⚪ Vyrovnané oblasti u Rozhodovacího stromu
    - Přepněte na **Rozhodovací strom**: červená i modrá barva dramaticky blednou!
    - **Důvod:** Strom provedl pravoúhlé řezy souřadnic (`lat <= 47.53`, `long <= -122.21`) a izoloval prestižní enklávy do samostatných listů s vysokým cenovým průměrem.
    """)
