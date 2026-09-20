from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("🗺️ Mapa Seattlu: Kde přesně modely dělají chyby?")
st.caption("Porovnání Lineární regrese vs. Rozhodovacího stromu na skutečné mapě King County: Proč strom ušetřil 35 000 USD na každém domě?")

@st.cache_data
def load_geo_data():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "01_Regression" / "data" / "day1_extras_precomputed.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["geo_sample"])

df_geo = load_geo_data()

# =============================================================================
# VYSVĚTLENÍ LIDSKOU ŘEČÍ (PŘED GRAFEM)
# =============================================================================
with st.container(border=True):
    st.markdown("### 💡 O co na této mapě jde a jak ji číst?")
    st.markdown("""
    Při trénování jsme zjistili, že **Rozhodovací strom je o 35 000 USD přesnější než Lineární regrese**.  
    Tato mapa ukazuje **1 500 skutečných domů v Seattlu a okolí** a barvou znázorňuje, **jak moc se model spletl**:

    - 🔴 **Červená tečka = Model dům těžce PODHODNOTIL** (dům je ve skutečnosti mnohem dražší, než model tipoval – např. luxusní čtvrť u jezera).
    - ⚪ **Bílá / Světlá tečka = Model se TREFIL PŘESNĚ** (chyba do ±10 %).
    - 🔵 **Modrá tečka = Model dům těžce NADHODNOTIL** (model tipoval vysokou cenu, ale lokalita je levnější).
    """)

# Výběr modelu a typu zobrazení
col_m1, col_m2 = st.columns([2, 2])
with col_m1:
    sel_model = st.radio(
        "👉 Vyberte model k zobrazení:",
        ["1. Lineární regrese (OLS) – Chybující baseline", "2. Rozhodovací strom (CART) – Vítězný model"],
        horizontal=False
    )
with col_m2:
    map_style = st.selectbox(
        "Podklad mapy:",
        ["carto-positron (Světlé ulice)", "open-street-map (Klasická mapa)", "carto-darkmatter (Tmavá mapa)"]
    )

is_tree = "strom" in sel_model.lower()
res_col = "pct_tree" if is_tree else "pct_ols"
abs_col = "res_tree" if is_tree else "res_ols"
pred_col = "pred_tree" if is_tree else "pred_ols"

# Připravíme čitelné popisky pro hover
df_plot = df_geo.copy()
df_plot["err_pct"] = df_plot[res_col].clip(-50, 50)
df_plot["skutecna_cena"] = df_plot["price"].apply(lambda x: f"{x:,.0f} USD")
df_plot["predikce_modelu"] = df_plot[pred_col].apply(lambda x: f"{x:,.0f} USD")
df_plot["chyba_v_usd"] = df_plot[abs_col].apply(lambda x: f"{x:,.0f} USD")
df_plot["chyba_procenta"] = df_plot[res_col].apply(lambda x: f"{x:+.1f} %")

# Výběr mapového podkladu
style_key = "carto-positron" if "positron" in map_style else ("open-street-map" if "open-street" in map_style else "carto-darkmatter")

# Plotly 6.0+ přejmenovalo scatter_mapbox na scatter_map a parametr mapbox_style na map_style
if hasattr(px, "scatter_map"):
    fig_map = px.scatter_map(
        df_plot,
        lat="lat",
        lon="long",
        color="err_pct",
        color_continuous_scale="RdBu_r",
        range_color=[-50, 50],
        size_max=9,
        zoom=9.5,
        center={"lat": 47.56, "lon": -122.25},
        map_style=style_key,
        hover_name="skutecna_cena",
        hover_data={
            "lat": False,
            "long": False,
            "err_pct": False,
            "predikce_modelu": True,
            "chyba_v_usd": True,
            "chyba_procenta": True,
            "sqft_living": True,
            "grade": True
        },
        title=f"Skutečná mapa chyb: {sel_model}"
    )
else:
    fig_map = px.scatter_mapbox(
        df_plot,
        lat="lat",
        lon="long",
        color="err_pct",
        color_continuous_scale="RdBu_r",
        range_color=[-50, 50],
        size_max=9,
        zoom=9.5,
        center={"lat": 47.56, "lon": -122.25},
        mapbox_style=style_key,
        hover_name="skutecna_cena",
        hover_data={
            "lat": False,
            "long": False,
            "err_pct": False,
            "predikce_modelu": True,
            "chyba_v_usd": True,
            "chyba_procenta": True,
            "sqft_living": True,
            "grade": True
        },
        title=f"Skutečná mapa chyb: {sel_model}"
    )

fig_map.update_layout(
    height=620,
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_colorbar=dict(
        title="Odchylka predikce",
        ticksuffix=" %",
        tickvals=[-50, -25, 0, 25, 50],
        ticktext=["<-50% (Podhodnoceno)", "-25%", "Trefa (0 %)", "+25%", ">+50% (Nadhodnoceno)"]
    )
)

st.plotly_chart(fig_map, width="stretch")

# Srovnávací shrnutí
col_a, col_b = st.columns(2)
with col_a:
    with st.container(border=True):
        st.markdown("#### ❌ Co vidíme u Lineární regrese (OLS):")
        st.markdown("""
        - Sever Seattlu a břehy jezera (Bellevue, Mercer Island) svítí **sytě červeně**!
        - **Proč?** Lineární rovnice zná jen celkovou plochu a stavbu. Nechápe pojem *„prestižní adresa u vody“*.
        - Obyčejný dům o 150 m² na severu stojí 1.5 milionu USD, ale OLS mu předpoví jen 500 tisíc USD.
        """)
with col_b:
    with st.container(border=True):
        st.markdown("#### ✅ Co se stane po přepnutí na Rozhodovací strom:")
        st.markdown("""
        - Červené i tmavě modré skvrny **výrazně vyblednou na neutrální barvy**.
        - **Jak to strom dokázal?** Vytvořil podmínky:  
          `lat >= 47.62` a `long <= -122.20` $\\to$ *„Tohle je drahý sever Seattlu, přičti 500 000 USD k základu!“*
        - Právě díky schopnosti ohraničit lokality pravoúhlými řezy dosáhl strom o tolik lepších výsledků.
        """)
