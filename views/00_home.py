import streamlit as st

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🔬 Data Science & Machine Learning Portfolio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interaktivní webová platforma pro řešení praktických úloh, simulace modelů a teoretický rozbor kurzu (Coderslab 2026).</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Aktuální blok", value="01. Regrese", delta="Aktivní")
with col2:
    st.metric(label="Dokončená cvičení", value="2 / 2", delta="100 %")
with col3:
    st.metric(label="Natrénované modely", value="4", delta="Scikit-learn + HGB")
with col4:
    st.metric(label="Stav platformy", value="Streamlit Cloud", delta="Online")

st.markdown("---")

st.header("📌 Rozvrh a struktura kurzu")

course_data = [
    {
        "Blok": "Blok 0: Prework",
        "Téma": "Python, Pandas, Matplotlib, Plotly, Statistika, Čištění dat",
        "Stav": "Dokončeno (13 sešitů)",
        "Umístění": "00_Prework/",
    },
    {
        "Blok": "Blok 1: Lineární regrese",
        "Téma": "OLS, Gradient Descent, King County nemovitosti, Klenotník a diamanty, 4C parametry",
        "Stav": "Aktivní v levém menu",
        "Umístění": "01_Regression/",
    },
    {
        "Blok": "Blok 1: Klasifikace",
        "Téma": "Logistická regrese, k-NN, Rozhodovací stromy, SVM, Metriky",
        "Stav": "Připraveno",
        "Umístění": "02_Classification/",
    },
    {
        "Blok": "Blok 2: Pokročilé ML modely",
        "Téma": "Random Forest, XGBoost, Boosting & Bagging, Neuronové sítě",
        "Stav": "Plánováno",
        "Umístění": "03_Advanced_ML_Neural_Networks/",
    },
    {
        "Blok": "Blok 3: NLP & Neřízené učení",
        "Téma": "TF-IDF, Word2Vec, BERT, PCA, Shlukování k-Means",
        "Stav": "Plánováno",
        "Umístění": "04_NLP/ a 05_Unsupervised_Learning/",
    },
]

st.dataframe(
    course_data,
    width="stretch",
    column_config={
        "Blok": st.column_config.TextColumn("Blok / Modul", width="medium"),
        "Téma": st.column_config.TextColumn("Klíčová témata", width="large"),
        "Stav": st.column_config.TextColumn("Status", width="medium"),
        "Umístění": st.column_config.TextColumn("Umístění v repozitáři", width="medium"),
    },
)

st.markdown("---")

st.subheader("🌲 Navigace v postranním panelu:")
st.info(
    """
    👈 **V levém menu jsou témata organizována do tematických modulů (Teorie + Cvičení pohromadě):**
    - **01. Lineární regrese (OLS)**: Teorie OLS & Gradient Descent + Cvičení 1 (Reality) a Cvičení 2 (Diamanty)
    - **02. Metriky regresních modelů**: Teorie $R^2$, MAE, RMSE + Cvičení 3 (Metriky reality) a Cvičení 4 (Metriky diamantů & $R^2$ paradox)
    - **03. Regularizace (Lasso, Ridge, Elastic Net)**: Teorie L1/L2 + Teorie Elastic Net + Cvičení 5 (Reality) a Cvičení 6 (Diamanty)
    - **04. Polynomiální regrese**: Teorie nelineárního mapování, Rungeho fenomén a moderní spliny
    """
)
