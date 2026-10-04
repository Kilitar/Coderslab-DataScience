"""
Den 3 Extras: 📚 Hyperparameter Cheat Sheet & Diagnostika problémů (Symptom-to-Fix)
===================================================================================
- Kompletní tahák do kapsy pro ladění Decision Trees, Random Forest, XGBoost a Keras NN.
- Vliv jednotlivých parametrů na Bias vs. Variance.
- Interaktivní diagnostická matice "Symptom -> Řešení" pro řešení problémů v praxi.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hyperparameter Cheat Sheet & Diagnostika", page_icon="📚", layout="wide")

st.markdown("""
# 📚 Hyperparameter Cheat Sheet & Diagnostika problémů
### Rychlý referenční průvodce laděním modelů a řešením praktických krizí (Symptom-to-Fix)
""")

tab_trees, tab_rf, tab_xgb, tab_nn, tab_diag = st.tabs([
    "🌳 Rozhodovací stromy",
    "🌲 Random Forest",
    "⚡ XGBoost",
    "🧠 Neuronové sítě (Keras)",
    "🩺 Diagnostika problémů (Symptom-to-Fix)",
])

# --- TAB 1: DECISION TREE ---
with tab_trees:
    st.subheader("Rozhodovací stromy (DecisionTreeClassifier / Regressor)")
    df_dt = pd.DataFrame([
        {"Parametr": "max_depth", "Výchozí hodnota": "None", "Typický rozsah": "3 – 12", "Vliv na Bias/Variance": "Vyšší hloubka = nižší bias, vysoká variance (overfitting)"},
        {"Parametr": "min_samples_split", "Výchozí hodnota": "2", "Typický rozsah": "5 – 50", "Vliv na Bias/Variance": "Vyšší hodnota tlumí větvení a regularizuje model"},
        {"Parametr": "min_samples_leaf", "Výchozí hodnota": "1", "Typický rozsah": "2 – 30", "Vliv na Bias/Variance": "Klíčový pro zamezení izolovaných listů s 1 vzorkem"},
        {"Parametr": "max_features", "Výchozí hodnota": "None", "Typický rozsah": "'sqrt', 'log2', 0.5 – 0.8", "Vliv na Bias/Variance": "Omezení výběru příznaků zavádí náhodnost"},
        {"Parametr": "ccp_alpha", "Výchozí hodnota": "0.0", "Typický rozsah": "0.0001 – 0.05", "Vliv na Bias/Variance": "Cost-Complexity Pruning (dodatečné ořezání přerostlého stromu)"},
    ])
    st.dataframe(df_dt, hide_index=True, width="stretch")

# --- TAB 2: RANDOM FOREST ---
with tab_rf:
    st.subheader("Náhodný les (RandomForestClassifier / Regressor)")
    df_rf = pd.DataFrame([
        {"Parametr": "n_estimators", "Výchozí hodnota": "100", "Typický rozsah": "100 – 500", "Vliv na Bias/Variance": "Více stromů NIKDY nezpůsobí overfitting (pouze snižuje varianci)"},
        {"Parametr": "max_features", "Výchozí hodnota": "'sqrt' (klas.) / 1.0 (regr.)", "Typický rozsah": "'sqrt', 0.3 – 0.5", "Vliv na Bias/Variance": "Klíčové pro de-korelaci stromů (snižuje asymptotickou mez ρ·σ²)"},
        {"Parametr": "bootstrap", "Výchozí hodnota": "True", "Typický rozsah": "True", "Vliv na Bias/Variance": "Trénování na náhodných výběrech s opakováním (cca 63.2 % dat)"},
        {"Parametr": "oob_score", "Výchozí hodnota": "False", "Typický rozsah": "True", "Vliv na Bias/Variance": "Validace 'zdarma' na netrénovaných out-of-bag vzorcích"},
        {"Parametr": "n_jobs", "Výchozí hodnota": "None", "Typický rozsah": "-1", "Vliv na Bias/Variance": "Paralelizace přes všechna jádra procesoru (nemá vliv na matematiku)"},
    ])
    st.dataframe(df_rf, hide_index=True, width="stretch")

# --- TAB 3: XGBOOST ---
with tab_xgb:
    st.subheader("XGBoost (XGBClassifier / XGBRegressor)")
    df_xgb = pd.DataFrame([
        {"Parametr": "n_estimators", "Výchozí hodnota": "100", "Typický rozsah": "100 – 1000+", "Vliv na Bias/Variance": "Příliš mnoho stromů MŮŽE přeučit! Používej early stopping."},
        {"Parametr": "learning_rate (eta)", "Výchozí hodnota": "0.3", "Typický rozsah": "0.01 – 0.1", "Vliv na Bias/Variance": "Nižší η vyžaduje více stromů, ale výrazně zlepšuje generalizaci"},
        {"Parametr": "max_depth", "Výchozí hodnota": "6", "Typický rozsah": "3 – 7", "Vliv na Bias/Variance": "V boostingu se používají MĚLKÉ stromy (často stačí hloubka 3-5)"},
        {"Parametr": "subsample", "Výchozí hodnota": "1.0", "Typický rozsah": "0.6 – 0.9", "Vliv na Bias/Variance": "Řádkové vzorkování (Stochastic Gradient Boosting)"},
        {"Parametr": "colsample_bytree", "Výchozí hodnota": "1.0", "Typický rozsah": "0.6 – 0.9", "Vliv na Bias/Variance": "Sloupcové vzorkování (analogie max_features v RF)"},
        {"Parametr": "gamma (min_split_loss)", "Výchozí hodnota": "0.0", "Typický rozsah": "0.1 – 5.0", "Vliv na Bias/Variance": "Minimální redukce ztráty nutná pro další rozdělení uzlu"},
        {"Parametr": "reg_alpha / reg_lambda", "Výchozí hodnota": "0 / 1", "Typický rozsah": "L1 (0.1-10) / L2 (1-100)", "Vliv na Bias/Variance": "L1 a L2 regularizace vah na listech stromu"},
    ])
    st.dataframe(df_xgb, hide_index=True, width="stretch")

# --- TAB 4: KERAS NN ---
with tab_nn:
    st.subheader("Neuronové sítě (TensorFlow / Keras)")
    df_nn = pd.DataFrame([
        {"Komponenta": "Aktivace skrytých vrstev", "Standard": "ReLU", "Alternativy": "LeakyReLU, GELU, Tanh", "Doporučení": "Vyhni se Sigmoidu ve skrytých vrstvách (způsobuje vanishing gradient)."},
        {"Komponenta": "Aktivace výstupu", "Standard": "Sigmoid (binární) / Softmax (multi) / Linear (regr.)", "Alternativy": "Softplus", "Doporučení": "Výstup musí odpovídat zvolené loss funkci!"},
        {"Komponenta": "Optimalizátor", "Standard": "Adam (lr=1e-3)", "Alternativy": "AdamW, RMSprop, SGD+Momentum", "Doporučení": "Adam je nejrobustnější výchozí volba."},
        {"Komponenta": "Regularizace Dropout", "Standard": "0.2 – 0.5", "Alternativy": "L2 kernel_regularizer", "Doporučení": "Vkládej za Dense vrstvy při overfittingu."},
        {"Komponenta": "Batch Size", "Standard": "32 nebo 64", "Alternativy": "16, 128, 256", "Doporučení": "Menší dávky zavádí regularizační šum, větší dávky lépe vytěžují GPU."},
        {"Komponenta": "Škálování vstupů", "Standard": "StandardScaler / MinMaxScaler", "Alternativy": "Normalizace na [0, 1]", "Doporučení": "STRIKTNÍ NUTNOST! Bez škálování neuronové sítě nekonvergují."},
    ])
    st.dataframe(df_nn, hide_index=True, width="stretch")

# --- TAB 5: DIAGNOSTIKA ---
with tab_diag:
    st.subheader("🩺 Diagnostická matice krizových situací (Symptom-to-Fix)")
    
    symptom = st.selectbox(
        "Vyber symptom / problém tvého modelu:",
        [
            "1. Model má 99 % na trénovacích datech, ale na testovacích selhává (Extrémní Overfitting)",
            "2. Model má nízkou přesnost jak na trénovacích, tak na testovacích datech (Underfitting / Vysoký bias)",
            "3. Trénování trvá neúnosně dlouho a přetěžuje hardware",
            "4. Ztráta (Loss) neuronové sítě vůbec neklesá nebo osciluje (Mizející / Explodující gradient)",
            "5. Datová sada je silně třídně nevyvážená (např. 98 % zdravých vs. 2 % nemocných)",
        ],
    )
    
    if "1. Model má 99 %" in symptom:
        st.error("🚨 **Diagnóza:** Vysoká variance / Přeučení (Overfitting). Model si zapamatoval trénovací šum.")
        st.markdown("""
        * **Random Forest:** Sniž `max_depth` (např. na 6–8), zvyš `min_samples_leaf` (na 5–10), sniž `max_features` pro větší de-korelaci.
        * **XGBoost:** Sniž `max_depth` (na 3–4), sniž `learning_rate` (na 0.03), nastav `subsample=0.8` a `colsample_bytree=0.8`, zvyš `gamma` a `reg_lambda`.
        * **Neuronové sítě:** Přidej vrstvy `Dropout(0.3 - 0.5)`, zaveď L2 regularizaci `kernel_regularizer=l2(1e-4)`, použij `EarlyStopping(patience=5)` a zmenši počet neuronů.
        """)
    elif "2. Model má nízkou" in symptom:
        st.warning("⚠️ **Diagnóza:** Vysoký bias / Podučení (Underfitting). Model je příliš jednoduchý a nechápe závislosti.")
        st.markdown("""
        * **Stromy & Random Forest:** Zvyš `max_depth`, povol hlubší větvení, přidej do dat nelineární příznaky a interakce (feature engineering).
        * **XGBoost:** Zvyš `max_depth` (na 6–8), zvyš počet stromů `n_estimators`, zkontroluj, zda není `learning_rate` příliš malý.
        * **Neuronové sítě:** Přidej skryté vrstvy nebo zvyš počet neuronů, zkontroluj, zda nepoužíváš nevhodnou aktivaci (např. lineární místo ReLU), ověř normalizaci dat.
        """)
    elif "3. Trénování trvá" in symptom:
        st.info("⏱️ **Diagnóza:** Výpočetní neefektivita a neoptimální konfigurace jader.")
        st.markdown("""
        * **Scikit-learn (RF/DT):** Vždy nastav `n_jobs=-1` (využití všech CPU jader).
        * **XGBoost:** Nastav `tree_method='hist'` (histogramový algoritmus – zrychlení 10x až 20x při zachování přesnosti!).
        * **Neuronové sítě:** Zvyš `batch_size` (např. z 32 na 128 nebo 256), zapni GPU akceleraci, zredukuj počet vstupních dimenzí.
        """)
    elif "4. Ztráta (Loss)" in symptom:
        st.error("💥 **Diagnóza:** Numerická nestabilita gradientního sestupu.")
        st.markdown("""
        * **Mizející gradient (Loss se nepohne):** Okamžitě nahraď Sigmoid/Tanh ve skrytých vrstvách za **ReLU**. Zkontroluj inicializaci vah (`he_normal`).
        * **Explodující gradient (Loss = NaN):** Sniž `learning_rate` o řád (např. z 0.1 na 0.001), zaveď Gradient Clipping (`clipnorm=1.0`).
        * **Normalizace:** Zkontroluj, zda jsi data normalizoval (`StandardScaler`). Neškálované proměnné (např. cena 500 000 vs počet pokojů 3) spolehlivě zabijí trénink!
        """)
    else:
        st.warning("⚖️ **Diagnóza:** Třídní nerovnováha (Class Imbalance). Model se naučil predikovat pouze majoritní třídu.")
        st.markdown("""
        * **Metrika:** Přestaň sledovat Accuracy! Vyhodnocuj **Precision, Recall, F1-Score a PR-AUC**.
        * **Random Forest:** Nastav `class_weight='balanced'` nebo `class_weight='balanced_subsample'`.
        * **XGBoost:** Nastav parametr `scale_pos_weight = (počet záporných) / (počet kladných)`.
        * **Práh:** Použij naši **Cost-Sensitive Threshold kalkulačku** a sniž rozhodovací práh z 0.50 dolů!
        """)
