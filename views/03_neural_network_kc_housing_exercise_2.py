"""
Den 3: Neuronové sítě – Cvičení 2 (Predikce cen nemovitostí KC Housing) – Výsledky zadání
======================================================================================
Zpracování oficiálního zadání kurzu v Keras:
1. Načtení dat kc_house_data_preprocessed.csv (21 613 domů, 18 nezávislých proměnných).
2. Předzpracování: Normalizace vstupů i výstupu (StandardScaler) a split 80:20 (random_state=42).
3. Experimentování s počtem skrytých vrstev s podmínkou <= 18 neuronů v každé vrstvě.
4. Výstupní vrstva: 1 neuron s lineární aktivací (Dense(1, activation='linear')).
5. Kompilace s optimalizátorem Adam a ztrátovou funkcí MSE (mean_squared_error).
6. Trénování sítě (40 epoch, batch_size=128, validation_split=0.15).
7. Predikce na testovací sadě X_test a výpočet testovacího MAE a RMSE.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_kc_house_mlp_data():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "kc_house_mlp_exercise_2_precomputed.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_neural_network_kc_housing_exercise_2_view():
    st.title("🎯 Cvičení 2: Predikce cen domů (Keras Regresní MLP)")
    st.markdown(
        r"""
        V tomto cvičení stavíme regresní neuronovou síť v knihovně **TensorFlow / Keras** pro predikci 
        cen nemovitostí v King County (`kc_house_data_preprocessed.csv`). 
        Zkoumáme vliv počtu skrytých vrstev na hodnotu ztrátové funkce **MSE** při striktním omezení 
        **$\le 18$ neuronů ve vrstvě**.
        """
    )

    data = load_kc_house_mlp_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `03_Advanced_ML_Neural_Networks/09_neural_network_kc_housing_exercise_2.py`.")
        return

    meta = data["metadata"]
    exps = data["experiments"]
    baselines = data["baselines"]
    best_exp = exps["exp3_3layers"]

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Testovací MAE (Vítězný model)",
            value=f"{best_exp['test_mae']:,.0f} USD",
            delta=f"-{baselines['linear_regression']['mae'] - best_exp['test_mae']:,.0f} USD vs. OLS",
            delta_color="normal"
        )
    with k2:
        st.metric(
            label="Testovací RMSE",
            value=f"{best_exp['test_rmse']:,.0f} USD",
            delta=f"R² skóre: {best_exp['test_r2']*100:.2f} %"
        )
    with k3:
        st.metric(
            label="Architektura šampióna",
            value="18 -> 12 -> 6 -> 1",
            delta="655 trénovatelných vah"
        )
    with k4:
        st.metric(
            label="Trénovací vzorky",
            value=f"{meta['n_train']:,}",
            delta=f"Test: {meta['n_test']:,} (18 příznaků)"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Experimenty s architekturou",
        "⚖️ Srovnání s baselines (OLS, Strom, RF)",
        "🔍 Rezidua & Predikce",
        "💻 Zdrojový kód zadání"
    ])

    # =========================================================================
    # TAB 1: EXPERIMENTY S ARCHITEKTUROU
    # =========================================================================
    with tab1:
        st.subheader("1. Hledání optimálního počtu vrstev podle hodnoty MSE loss")
        st.markdown(
            r"""
            Zadání požaduje:
            - **Vstupní vrstva:** 18 neuronů (počet nezávislých proměnných).
            - **Skryté vrstvy:** Počet neuronů $\le 18$. Počet vrstev zvolit na základě experimentů a sledování hodnoty ztrátové funkce.
            - **Výstupní vrstva:** 1 neuron s lineární aktivací.
            """
        )

        exp_table = []
        for eid, einfo in exps.items():
            exp_table.append({
                "Experiment": einfo["name"],
                "Architektura vrstev": einfo["layers_desc"],
                "Počet vah": einfo["n_params"],
                "Doba trénování": f"{einfo['duration_s']:.2f} s",
                "Test MAE (USD)": f"{einfo['test_mae']:,.2f} USD",
                "Test RMSE (USD)": f"{einfo['test_rmse']:,.2f} USD",
                "R² skóre": f"{einfo['test_r2']*100:.2f} %"
            })
        st.table(pd.DataFrame(exp_table))

        st.markdown("---")
        st.subheader("Porovnání křivek ztrátové funkce (Validační MSE)")

        fig_loss = go.Figure()
        colors = {"exp1_1layer": "#f59e0b", "exp2_2layers": "#3b82f6", "exp3_3layers": "#10b981", "exp4_narrow": "#ef4444"}
        for eid, einfo in exps.items():
            hist = einfo["history"]
            fig_loss.add_trace(go.Scatter(
                x=hist["epoch"],
                y=hist["val_loss"],
                mode="lines",
                name=f"{einfo['name']} (Val MSE)",
                line=dict(color=colors.get(eid, "#94a3b8"), width=2)
            ))

        fig_loss.update_layout(
            title="Průběh validační MSE ztráty napříč 40 epochami",
            xaxis_title="Epocha",
            yaxis_title="Validační MSE Loss (škálovaný)",
            height=430,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_loss)

        st.success(
            r"""
            **Závěr experimentu:**  
            - **1 vrstva (18 neuronů):** Dokáže zachytit jen základní nelinearity, MAE = 83 046 USD.
            - **2 vrstvy (18 -> 10 neuronů):** Výrazné zlepšení na MAE = 78 565 USD.
            - **3 vrstvy (18 -> 12 -> 6 neuronů):** **Vítězná konfigurace!** Plynulé trychtýřovité zužování sítě dosahuje nejnižší MAE = **78 157 USD** při pouhých 655 vahách.
            - **Příliš úzká síť (4 neurony - Bottleneck):** Ztrácí důležité informace o prostorových vazbách a MAE degraduje na 96 951 USD.
            """
        )

    # =========================================================================
    # TAB 2: SROVNÁNÍ MODELŮ
    # =========================================================================
    with tab2:
        st.subheader("2. Přímé srovnání: OLS vs. Rozhodovací strom vs. Random Forest vs. Keras MLP")
        st.markdown(
            r"""
            Srovnání přesnosti modelů na testovací sadě domů v King County:
            """
        )

        cmp_data = [
            {"Model": "📈 Lineární regrese (OLS)", "Test MAE (USD)": f"{baselines['linear_regression']['mae']:,.2f} USD", "Test RMSE (USD)": f"{baselines['linear_regression']['rmse']:,.2f} USD", "R² skóre": f"{baselines['linear_regression']['r2']*100:.2f} %", "Poznámka": "Lineární omezení na prostorové nelinearity"},
            {"Model": "🌲 Rozhodovací strom (DecisionTree)", "Test MAE (USD)": f"{baselines['decision_tree']['mae']:,.2f} USD", "Test RMSE (USD)": f"{baselines['decision_tree']['rmse']:,.2f} USD", "R² skóre": f"{baselines['decision_tree']['r2']*100:.2f} %", "Poznámka": "Schodovité predikce, vysoký rozptyl"},
            {"Model": "🧠 Keras MLP (3 skryté vrstvy)", "Test MAE (USD)": f"{best_exp['test_mae']:,.2f} USD", "Test RMSE (USD)": f"{best_exp['test_rmse']:,.2f} USD", "R² skóre": f"{best_exp['test_r2']*100:.2f} %", "Poznámka": "Hladké nelineární aproximace (655 vah)"},
            {"Model": "🌳 Náhodný les (Random Forest)", "Test MAE (USD)": f"{baselines['random_forest']['mae']:,.2f} USD", "Test RMSE (USD)": f"{baselines['random_forest']['rmse']:,.2f} USD", "R² skóre": f"{baselines['random_forest']['r2']*100:.2f} %", "Poznámka": "100 stromů – robustní na geografické shluky"}
        ]
        st.table(pd.DataFrame(cmp_data))

        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Lineární regrese", "Rozhodovací strom", "Keras MLP (Sítě)", "Random Forest"],
                y=[baselines['linear_regression']['mae'], baselines['decision_tree']['mae'], best_exp['test_mae'], baselines['random_forest']['mae']],
                marker_color=["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]
            )
        ])
        fig_bar.update_layout(
            title="Srovnání průměrné chyby MAE (USD) – menší hodnota je lepší",
            yaxis_title="MAE v USD",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar)

    # =========================================================================
    # TAB 3: REZIDUA & PREDIKCE
    # =========================================================================
    with tab3:
        st.subheader("3. Analýza predikcí a reziduí vítězného modelu MLP")
        res_sample = pd.DataFrame(data["sample_residuals"])

        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            fig_sc = px.scatter(
                res_sample,
                x="actual",
                y="pred_mlp",
                color="grade",
                title="Skutečná vs. Predikovaná cena (Keras MLP)",
                labels={"actual": "Skutečná cena (USD)", "pred_mlp": "Predikce sítě (USD)", "grade": "Třída stavby (Grade)"},
                color_continuous_scale="Plasma"
            )
            max_v = max(res_sample["actual"].max(), res_sample["pred_mlp"].max())
            fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", name="Ideální shoda (y=x)", line=dict(color="gray", dash="dash")))
            fig_sc.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sc)

        with c_sc2:
            fig_res = px.scatter(
                res_sample,
                x="pred_mlp",
                y="res_mlp",
                color="sqft_living",
                title="Rezidua sítě (Skutečnost - Predikce)",
                labels={"pred_mlp": "Predikovaná cena (USD)", "res_mlp": "Reziduum (USD)", "sqft_living": "Obytná plocha (sqft)"},
                color_continuous_scale="Viridis"
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_res)

        st.markdown("#### Ukázka testovacích domů a predikcí")
        st.dataframe(
            res_sample[["bedrooms", "bathrooms", "sqft_living", "grade", "actual", "pred_mlp", "res_mlp"]].rename(
                columns={
                    "actual": "Skutečná cena (USD)",
                    "pred_mlp": "Predikce sítě (USD)",
                    "res_mlp": "Chyba / Reziduum (USD)"
                }
            ).head(10)
        )

    # =========================================================================
    # TAB 4: ZDROJOVÝ KÓD
    # =========================================================================
    with tab4:
        st.subheader("4. Kompletní kód řešení dle zadání")
        st.code(
            r'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Načtení předzpracovaného datasetu
df = pd.read_csv("data/kc_house_data_preprocessed.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# 2. Oddělení nezávislých proměnných a cíle
X = df.drop("price", axis=1)
y = df["price"]
n_features = X.shape[1] # 18 nezávislých proměnných

# 3. Rozdělení na trénovací a testovací sadu (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Normalizace dat (StandardScaler)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# 5. Sestavení architektury neuronové sítě (skryté vrstvy <= 18 neuronů)
model = Sequential([
    Dense(18, activation="relu", input_shape=(n_features,)),
    Dense(12, activation="relu"),
    Dense(6, activation="relu"),
    Dense(1, activation="linear") # Výstupní vrstva pro regresi
])

# 6. Kompilace s Adam a MSE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# 7. Trénování sítě
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=40,
    batch_size=128,
    validation_split=0.15,
    verbose=1
)

# 8. Predikce a zpětná transformace na původní měřítko v USD
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# 9. Výpočet testovacích metrik MAE a RMSE
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Testovací MAE:  {test_mae:,.2f} USD")
print(f"Testovací RMSE: {test_rmse:,.2f} USD")
            ''',
            language="python"
        )


if __name__ == "__main__":
    render_neural_network_kc_housing_exercise_2_view()
