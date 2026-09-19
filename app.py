import streamlit as st

st.set_page_config(
    page_title="Data Science & ML Portfolio | Coderslab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stylizace a moderní vzhled
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-status {
        background-color: #10B981;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
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
    st.metric(label="Aktuální modul", value="01. Regrese", delta="Aktivní")
with col2:
    st.metric(label="Dokončená cvičení", value="2 / 2", delta="100 %")
with col3:
    st.metric(label="Dostupné modely", value="4", delta="Scikit-learn + HGB")
with col4:
    st.metric(label="Platforma", value="Streamlit Cloud", delta="Online")

st.markdown("---")

st.header("📌 Rozvrh a struktura kurzu")

course_data = [
    {
        "Blok": "Blok 0: Prework",
        "Téma": "Python, Pandas, Matplotlib, Plotly, Statistika, Čištění dat",
        "Stav": "Dokončeno (13 sešitů)",
        "Odkaz": "00_Prework/",
    },
    {
        "Blok": "Blok 1: Lineární regrese",
        "Téma": "OLS, Gradient Descent, King County nemovitosti, Klenotník a diamanty, 4C parametry",
        "Stav": "Aktivní na webu",
        "Odkaz": "pages/01_📈_Linear_Regression.py",
    },
    {
        "Blok": "Blok 1: Klasifikace",
        "Téma": "Logistická regrese, k-NN, Rozhodovací stromy, SVM, Metriky",
        "Stav": "Připraveno",
        "Odkaz": "02_Classification/",
    },
    {
        "Blok": "Blok 2: Pokročilé ML modely",
        "Téma": "Random Forest, XGBoost, Boosting & Bagging, Neuronové sítě",
        "Stav": "Plánováno",
        "Odkaz": "03_Advanced_ML_Neural_Networks/",
    },
    {
        "Blok": "Blok 3: NLP & Neřízené učení",
        "Téma": "TF-IDF, Word2Vec, BERT, PCA, Shlukování k-Means",
        "Stav": "Plánováno",
        "Odkaz": "04_NLP/ a 05_Unsupervised_Learning/",
    },
]

st.dataframe(
    course_data,
    use_container_width=True,
    column_config={
        "Blok": st.column_config.TextColumn("Blok / Modul", width="medium"),
        "Téma": st.column_config.TextColumn("Klíčová témata", width="large"),
        "Stav": st.column_config.TextColumn("Status", width="medium"),
        "Odkaz": st.column_config.TextColumn("Umístění v repozitáři", width="medium"),
    },
)

st.markdown("---")

st.subheader("🚀 Začněte zkoumat:")
st.info(
    """
    👈 **V levém postranním panelu vyberte stránku `01_📈_Linear_Regression`**  
    Najdete tam:
    - **Interaktivního odhadce cen nemovitostí** (King County dataset).
    - **Automatickou oceňovací kalkulačku diamantů pro klenotníka** se 4C parametry.
    - **Kompletní teoretický rozbor a moderní ML/AI rozšíření (09/2026)**.
    """
)
