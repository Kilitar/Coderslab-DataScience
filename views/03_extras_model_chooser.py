"""
Den 3 Extras: 🧭 Průvodce výběrem modelu (Model Selection Wizard)
================================================================
- Expertní interaktivní dotazník pro volbu optimálního ML algoritmu v praxi.
- Vyhodnocuje typ dat (tabulková vs. obrazová vs. textová), velikost vzorku, 
  požadavky na interpretovatelnost, chybějící hodnoty a rychlost inference.
- Doporučuje algoritmy: OLS/LogReg, Decision Tree, Random Forest, XGBoost, MLP, CNN.
- Generuje okamžitě spustitelný Python kód se správnými hyperparametry a odkazuje na lekce kurzu.
"""

from typing import Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Průvodce výběrem modelu", page_icon="🧭", layout="wide")

st.markdown("""
# 🧭 Průvodce výběrem modelu (Model Selection Wizard)
### Který algoritmus zvolit pro tvůj projekt? Interaktivní rozhodovací strom pro Data Scientisty
""")

st.info("""
V kurzu jsme prošli celou evoluci modelů: od **Lineární regrese a OLS**, přes **Rozhodovací stromy**, 
**Random Forest**, **XGBoost** až po **Hluboké neuronové sítě**. V praxi však neplatí, že nejsložitější model 
je vždy ten nejlepší. Tento průvodce ti pomůže na základě 6 klíčových parametrů tvého projektu vybrat vítěze.
""")

col_q1, col_q2 = st.columns(2)

with col_q1:
    st.subheader("1. Povaha dat a úlohy")
    task_type = st.radio("Typ predikční úlohy:", ["Regrese (spojitá hodnota)", "Klasifikace (kategorie / třídy)"], index=0)
    
    data_nature = st.selectbox(
        "Struktura a typ dat:",
        [
            "Tabulková strukturovaná data (čísla, kategorie, tabulky z DB)",
            "Obrazová data / Grid (fotografie, rentgenové snímky, MNIST pixely)",
            "Textová data nebo sekvence (NLP, časové řady)",
        ],
        index=0,
    )
    
    data_size = st.select_slider(
        "Velikost trénovací množiny (počet řádků N):",
        options=["Malá (< 1 000 řádků)", "Střední (1 000 – 100 000 řádků)", "Velká (> 100 000 řádků)"],
        value="Střední (1 000 – 100 000 řádků)",
    )

with col_q2:
    st.subheader("2. Provozní a byznysové požadavky")
    interpretability = st.radio(
        "Požadavek na interpretovatelnost (Explainability):",
        [
            "Kritická (nutno exaktně obhájit koeficienty/pravidla před auditem či lékařem)",
            "Střední (stačí Feature Importances / SHAP hodnoty)",
            "Nízká (zajímá nás čistě predikční výkon – Black Box je v pořádku)",
        ],
        index=1,
    )
    
    missing_data = st.radio(
        "Kvalita dat & Chybějící hodnoty (NaN):",
        [
            "Data obsahují NaN a nechceme je ručně imputovat",
            "Data jsou kompletní nebo máme robustní preprocessing pipeline",
        ],
        index=1,
    )
    
    latency_req = st.radio(
        "Omezení latence při nasazení do produkce (Inference speed):",
        [
            "Běžná serverová/dávková inference (desítky až stovky ms jsou v pořádku)",
            "Ultra-nízká latence na mikroprocesoru / IoT / real-time (< 1 ms)",
        ],
        index=0,
    )

# --- VYHODNOCOVACÍ ALGORITMUS ---
scores: Dict[str, float] = {
    "Lineární model (OLS / Ridge / Lasso / Logistic Regression)": 0.0,
    "Jednoduchý rozhodovací strom (Decision Tree)": 0.0,
    "Náhodný les (Random Forest)": 0.0,
    "Gradient Boosting (XGBoost / LightGBM)": 0.0,
    "Hluboký vícevrstvý perceptron (MLP Neural Network)": 0.0,
    "Konvoluční neuronová síť (CNN)": 0.0,
}

# 1. Obrazová vs tabulková data
if "Obrazová" in data_nature:
    scores["Konvoluční neuronová síť (CNN)"] += 50.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] += 15.0
elif "Textová" in data_nature:
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] += 20.0
    scores["Gradient Boosting (XGBoost / LightGBM)"] += 15.0
else:
    # Tabulková data: království stromů a XGBoostu
    scores["Gradient Boosting (XGBoost / LightGBM)"] += 35.0
    scores["Náhodný les (Random Forest)"] += 30.0
    scores["Lineární model (OLS / Ridge / Lasso / Logistic Regression)"] += 15.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] += 5.0

# 2. Interpretovatelnost
if "Kritická" in interpretability:
    scores["Lineární model (OLS / Ridge / Lasso / Logistic Regression)"] += 40.0
    scores["Jednoduchý rozhodovací strom (Decision Tree)"] += 35.0
    scores["Gradient Boosting (XGBoost / LightGBM)"] -= 20.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] -= 30.0
    scores["Konvoluční neuronová síť (CNN)"] -= 30.0
elif "Střední" in interpretability:
    scores["Náhodný les (Random Forest)"] += 20.0
    scores["Gradient Boosting (XGBoost / LightGBM)"] += 20.0

# 3. Velikost dat
if "Malá" in data_size:
    scores["Lineární model (OLS / Ridge / Lasso / Logistic Regression)"] += 25.0
    scores["Jednoduchý rozhodovací strom (Decision Tree)"] += 20.0
    scores["Náhodný les (Random Forest)"] += 15.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] -= 30.0
    scores["Konvoluční neuronová síť (CNN)"] -= 20.0
elif "Velká" in data_size:
    scores["Gradient Boosting (XGBoost / LightGBM)"] += 25.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] += 20.0
    scores["Náhodný les (Random Forest)"] += 10.0

# 4. Chybějící hodnoty
if "obsahují NaN" in missing_data:
    scores["Gradient Boosting (XGBoost / LightGBM)"] += 30.0  # Nativně podporuje sparse & NaN
    scores["Lineární model (OLS / Ridge / Lasso / Logistic Regression)"] -= 25.0
    scores["Hluboký vícevrstvý perceptron (MLP Neural Network)"] -= 25.0

# 5. Latence
if "Ultra-nízká" in latency_req:
    scores["Lineární model (OLS / Ridge / Lasso / Logistic Regression)"] += 30.0
    scores["Jednoduchý rozhodovací strom (Decision Tree)"] += 25.0
    scores["Náhodný les (Random Forest)"] -= 10.0

# Seřazení výsledků
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
best_model, best_score = ranked[0]
second_model, _ = ranked[1]
third_model, _ = ranked[2]

st.divider()

# --- VÝSLEDNÉ DOPORUČENÍ ---
st.subheader("🏆 Výsledek expertního posouzení")

c_res1, c_res2, c_res3 = st.columns([1.5, 1.2, 1.2])

with c_res1:
    st.success(f"### 🥇 Doporučený model č. 1:\n**{best_model}**")
    st.markdown(f"**Proč právě tento model?** Nejlépe balancuje tvoje požadavky na data (*{data_nature.split('(')[0].strip()}*), velikost vzorku a interpretovatelnost.")

with c_res2:
    st.info(f"### 🥈 Alternativa č. 2:\n**{second_model}**")
    st.caption("Výborná volba pro srovnání baseline nebo pokud selže ladění hyperparametrů vítěze.")

with c_res3:
    st.write(f"### 🥉 Alternativa č. 3:\n**{third_model}**")

st.divider()

# --- UKÁZKA STARTOVACÍHO KÓDU ---
st.subheader(f"💻 Startovací kód v Pythonu pro: {best_model}")

if "XGBoost" in best_model:
    code_snippet = """import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doporučené moderní hyperparametry pro tabulková data
model = xgb.XGBClassifier(  # nebo xgb.XGBRegressor pro regresi
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

# Trénování s early stoppingem
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=20
)
"""
elif "Random Forest" in best_model:
    code_snippet = """from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest s de-korelací a omezením přeučení
model = RandomForestClassifier(  # nebo RandomForestRegressor
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=3,
    max_features="sqrt",  # klíčové pro de-korelaci stromů
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)
"""
elif "Konvoluční" in best_model:
    code_snippet = """from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
"""
elif "Lineární" in best_model:
    code_snippet = """from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Lineární modely striktně vyžadují normalizaci!
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=500, random_state=42)  # nebo Ridge() pro regresi
)
pipeline.fit(X_train, y_train)
"""
elif "perceptron" in best_model:
    code_snippet = """from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  # nebo layers.Dense(n_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
"""
else:
    code_snippet = """from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

model = DecisionTreeClassifier(  # nebo DecisionTreeRegressor
    max_depth=4,  # mělčí strom pro zachování interpretovatelnosti
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
"""

st.code(code_snippet, language="python")
