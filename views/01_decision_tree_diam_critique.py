import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("🔬 Cvičení 9: Expertní analýza – Strom vs. Polynom na diamantech")
st.caption("Detailní metodologická komparace: Proč rozhodovací strom drtí polynomiální regresi na cenách diamantů, fenomén 'magických karátů' a diagnostika reziduí.")

tab_vs, tab_carat, tab_resid, tab_wrapper = st.tabs([
    "⚔️ Strom vs. Polynomiální regrese",
    "💎 Fenomén 'Magických karátových hranic'",
    "📉 Diagnostika reziduí & Rozptyl",
    "⚙️ Architektura 'train_and_evaluate_tree' wrapperu"
])

# =============================================================================
# TAB 1: STROM VS POLYNOM
# =============================================================================
with tab_vs:
    st.markdown("### ⚔️ Proč rozhodovací strom poráží i stupeň 3 polynomiální regrese?")
    st.markdown(r"""
    V Cvičení 7 jsme diamantům nasadili polynomiální regresi stupně 2 a 3 s Lasso regularizací.  
    Podívejme se na přímé srovnání s DecisionTreeRegressor:
    """)

    comp_data = [
        {"Model": "OLS Lineární regrese (Cvičení 2)", "Test R²": "0.88 - 0.92", "Test RMSE": "1 150 - 1 400 USD", "Interpretovatelnost": "Koeficienty beta", "Extrapolace": "Lineární trend"},
        {"Model": "Polynomiální regrese 3. stupně (Cvičení 7)", "Test R²": "0.950 - 0.958", "Test RMSE": "850 - 920 USD", "Interpretovatelnost": "Velmi těžká (mnoho členů)", "Extrapolace": "Explodující křivky"},
        {"Model": "Výchozí strom (Baseline, Cvičení 9)", "Test R²": "0.9681", "Test RMSE": "715.5 USD", "Interpretovatelnost": "Nulová (36k listů)", "Extrapolace": "Konstantní strop"},
        {"Model": "Optimální prořezaný strom (GridSearch, Cvičení 9)", "Test R²": "0.9787", "Test RMSE": "584.0 USD", "Interpretovatelnost": "Pravidla If-Else (1.8k listů)", "Extrapolace": "Konstantní strop"}
    ]
    st.dataframe(pd.DataFrame(comp_data), width="stretch")

    st.markdown(r"""
    #### Proč má strom tak drtivou převahu?
    1. **Nulový předpoklad globálního tvaru:** Polynom se snaží proložit data hladkou křivkou $y = w_1 x + w_2 x^2 + w_3 x^3$. Reálný trh diamantů ale není hladký!
    2. **Orgonální řezy v multidimenzionálním prostoru:** Strom dokáže říct: *„Pokud má diamant carat $\ge 1.0$, ALE barva je špatná ($color \le 2$), aplikuj slevu 40 %“*. To je pro polynom složitá interakce více stupňů, zatímco pro strom triviální 2 podmínky If-Else.
    """)

# =============================================================================
# TAB 2: MAGICKÉ KARÁTY
# =============================================================================
with tab_carat:
    st.markdown("### 💎 Fenomén 'Magických karátových hranic'")
    st.markdown(r"""
    Na trhu s diamanty existuje psychologický skok v ceně:
    - Diamant o váze **0.99 karátu** stojí např. 4 500 USD.
    - Diamant o váze **1.00 karátu** stojí okamžitě 6 000 USD! (Skok o 30 % za rozdíl pouhých 0.01 ct).
    
    Obdobné skoky nastávají na hranicích **0.50 ct, 0.70 ct, 1.00 ct, 1.50 ct, 2.00 ct**.
    """)

    st.markdown("""
    ```
    Skutečný trh:                  Hladký polynom:                 Rozhodovací strom:
    Cena                           Cena                            Cena
      ^       |                      ^        /                      ^       ┌──────
      |       |  ┌─────              |       /                       |       │
      |       |  │                   |      /                        |   ┌───┘
      |  ─────┘  │                   |     /                         |   │
      +--------1.00---> Karáty       +--------1.00---> Karáty        +---┴---1.00---> Karáty
         (Skoková nespojitost)           (Vyhlazuje a podstřeluje)         (Přesný skokový řez!)
    ```
    """)

    st.success(
        "💡 **Proč je strom ideální:** "
        "Rozhodovací strom je ze své podstaty **po částech konstantní funkce (step-function)**. "
        "Dělicí podmínka typu `carat <= 0.995` dokonale zachycuje tuto tržní diskontinuitu, "
        "kterou žádný hladký polynom nedokáže přesně vystihnout bez nežádoucího vlnění (Rungeho jev)."
    )

# =============================================================================
# TAB 3: REZIDUA & ROZPTYL
# =============================================================================
with tab_resid:
    st.markdown("### 📉 Diagnostika reziduí & Konstantní strop")
    st.markdown(r"""
    Ačkoliv strom dosahuje $R^2 = 97.9 \%$, nese s sebou fundamentální vlastnosti stromových modelů:
    """)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        #### 1. Heteroskedasticita (Rostoucí rozptyl chyb)
        - U malých diamantů do 1 500 USD je MAE stromu pod **90 USD**.
        - U velkých investičních diamantů nad 15 000 USD je rozptyl v datech obrovský a chyba dosahuje stovek až tisíců USD.
        - Důvod: Vzácných velkých diamantů je v datasetu málo, takže v listech stromu chybí dostatek vzorků pro přesný průměr.
        """)
    with col_r2:
        st.markdown("""
        #### 2. Neschopnost extrapolace (Strop)
        - Pokud na trh přijde unikátní diamant o váze **5.5 karátu** (v trénovacích datech bylo maximum 5.01 ct), 
          strom **nikdy nepředpoví cenu vyšší**, než jakou má nejdražší list v trénovacích datech!
        - Výstupem stromu je vždy pouze průměr trénovacích bodů v daném listu.
        """)

# =============================================================================
# TAB 4: WRAPPER FUNKCE
# =============================================================================
with tab_wrapper:
    st.markdown("### ⚙️ Architektura 'train_and_evaluate_tree' wrapperu")
    st.markdown(r"""
    Zadání cvičení výslovně požadovalo vytvořit pomocnou funkci pro zrychlení procesu modelování:
    """)

    st.code('''def train_and_evaluate_tree(X_train, y_train, X_test, y_test, **tree_params):
    """
    1. Inicializuje DecisionTreeRegressor s volitelnými hyperparametry
    2. Natrénuje model na X_train, y_train
    3. Spočte predikce pro Train i Test sadu
    4. Vypočte metriky: R2, MAE, MSE, RMSE a Overfitting Gap
    5. Volitelně vygeneruje a uloží graf Predikce vs. Skutečnost
    6. Vrátí natrénovaný model a slovník metrik
    """
    model = DecisionTreeRegressor(random_state=42, **tree_params)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "mse_test": mean_squared_error(y_test, y_pred_test),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "n_leaves": model.get_n_leaves(),
        "depth": model.get_depth()
    }
    return model, metrics''', language="python")

    st.info("💡 Tento přístup umožnil v několika řádcích kódu systematicky porovnat desítky kombinací hyperparametrů a napojit GridSearchCV.")
