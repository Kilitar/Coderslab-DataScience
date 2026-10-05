"""
Průvodce: Jak publikovat Colab / Jupyter notebook na Kaggle
===========================================================
Návod na exportování vypracovaných úkolů z Colabu / lokálního prostředí na Kaggle,
propojení s datasety, úpravu cest a vytvoření reprezentativního Data Science portfolia.
"""

import pandas as pd
import streamlit as st

st.title("🌐 Průvodce: Jak publikovat notebook z Colabu na Kaggle")
st.caption(
    "Kompletní praktický průvodce přenosem vypracovaných projektů z Google Colab nebo lokálního Jupyter prostředí "
    "na platformu **Kaggle** – včetně nahrání datasetů, správy cest (`/kaggle/input`), verzování a budování veřejného portfolia."
)

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Export", ".ipynb formát", delta="Z Colabu / Jupyteru")
c2.metric("2. Import", "Kaggle Create", delta="New Notebook -> Import")
c3.metric("3. Data Path", "/kaggle/input/...", delta="Specifická struktura")
c4.metric("4. Publikace", "Public Visibility", delta="Odkaz pro portfolio / CV")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Krok za krokem: Export & Publikace",
    "📁 2. Správa datasetů & Rozdíly v cestách",
    "🥊 3. Srovnání: Google Colab vs. Kaggle Kernels",
    "⭐ 4. Kaggle Portfolio: Best Practices pro zaměstnavatele"
])

# ==============================================================================
# TAB 1: KROK ZA KROKEM
# ==============================================================================
with tab1:
    st.subheader("Postup publikace notebooku na Kaggle")

    col_steps1, col_steps2 = st.columns(2)

    with col_steps1:
        st.markdown(
            """
            #### 1️⃣ Stažení notebooku z Google Colab / lokálního IDE
            - V Google Colab zvolte **File $\\rightarrow$ Download $\\rightarrow$ Download .ipynb**.
            - Formát `.ipynb` (Interactive Python Notebook) zachovává veškeré buňky, markdown formátování i předpočítané výstupy a grafy.
            
            #### 2️⃣ Vytvoření nového notebooku na Kaggle
            - Přejděte na [Kaggle.com](https://www.kaggle.com) a přihlaste se.
            - V levém navigačním panelu klikněte na modré tlačítko **Create** a zvolte **New Notebook**.
            - Otevře se interaktivní prostředí Kaggle Kernels.
            
            #### 3️⃣ Import hotového souboru
            - V horním menu klikněte na **File $\\rightarrow$ Import Notebook**.
            - Přetáhněte stažený `.ipynb` soubor do dialogového okna nebo jej vyberte přes průzkumník souborů.
            - Klikněte na **Import**. Kód se okamžitě načte do rozhraní Kaggle.
            """
        )

    with col_steps2:
        st.markdown(
            """
            #### 4️⃣ Připojení datasetu (Add Data)
            - V pravém bočním panelu klikněte na sekci **Input $\\rightarrow$ Add Data**.
            - Můžete vyhledat existující veřejný dataset (např. *Titanic*, *Auto MPG*, *McDonald's Store Reviews*), nebo nahrát vlastní `.csv` soubor přes **Upload a Dataset**.
            
            #### 5️⃣ Úprava cest k souborům v kódu
            - V Colabu bývá cesta např. `'data/auto_mpg.csv'` nebo `'/content/auto_mpg.csv'`.
            - Na Kaggle se data vždy připojují do adresáře:
              `/kaggle/input/<nazev-datasetu>/soubor.csv`.
            
            #### 6️⃣ Uložení verze & Publikace (Public)
            - V pravém horním rohu klikněte na **Save Version** (doporučujeme zvolit *Save & Run All (Commit)* pro validaci).
            - Po uložení klikněte na tlačítko **Share** vedle Save Version.
            - Přepněte viditelnost z **Private** na **Public** a zkopírujte odkaz na své řešení.
            """
        )

# ==============================================================================
# TAB 2: SPRÁVA DATASETŮ A ROZDÍLY V CESTÁCH
# ==============================================================================
with tab2:
    st.subheader("Souborový systém na Kaggle: Jak správně odkazovat na data")

    st.markdown(
        """
        Zatímco v Google Colab je kořenový adresář spuštění `/content/` s volným zápisem, 
        Kaggle striktně odděluje **vstupní data (pouze pro čtení)** a **pracovní adresář pro zápis výstupů**:
        """
    )

    col_paths1, col_paths2 = st.columns(2)

    with col_paths1:
        st.markdown("#### 📥 Vstupní data: `/kaggle/input/` (Read-Only)")
        st.code(
            """import pandas as pd
import os

# Vypsání všech připojených datasetů na Kaggle:
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Typické načtení datasetu na Kaggle:
df = pd.read_csv('/kaggle/input/mcdonalds-store-reviews/mcdonalds_reviews.csv')
""",
            language="python"
        )
        st.info("⚠️ **Pozor:** Do adresáře `/kaggle/input/` nelze zapisovat ani ukládat žádné soubory.")

    with col_paths2:
        st.markdown("#### 📤 Výstupní data a modely: `/kaggle/working/` (Read-Write)")
        st.code(
            """# Ukládání natrénovaného modelu nebo predikcí:
import joblib

# Uložení do pracovního adresáře:
joblib.dump(model, '/kaggle/working/trained_model.joblib')
sub_df.to_csv('/kaggle/working/submission.csv', index=False)

# Vše v /kaggle/working/ je po 'Save Version'
# dostupné ke stažení jako výstup notebooku!
""",
            language="python"
        )
        st.success("💡 Všechny soubory uložené do `/kaggle/working/` se po uložení verze zobrazí v sekci **Output**.")

# ==============================================================================
# TAB 3: SROVNÁNÍ COLAB VS KAGGLE
# ==============================================================================
with tab3:
    st.subheader("Google Colab vs. Kaggle Kernels: Výhody a limity")

    cmp_data = pd.DataFrame([
        {
            "Vlastnost / Aspekt": "GPU Akcelerace zdarma",
            "Google Colab (Free)": "1x Nvidia T4 (omezená výpočetní kvóta)",
            "Kaggle Kernels (Free)": "2x Nvidia T4 (30 hodin GPU týdně)",
            "Kdy zvolit": "Kaggle nabízí vyšší a transparentní týdenní kvótu"
        },
        {
            "Vlastnost / Aspekt": "TPU Akcelerace",
            "Google Colab (Free)": "TPU v2 (dostupnost bývá proměnlivá)",
            "Kaggle Kernels (Free)": "TPU v3-8 (20 hodin týdně)",
            "Kdy zvolit": "Kaggle pro rozsáhlé trénování hlubokých sítí"
        },
        {
            "Vlastnost / Aspekt": "Diskový prostor",
            "Google Colab (Free)": "~100 GB dočasný disk (maže se po odpojení)",
            "Kaggle Kernels (Free)": "20 GB scratch disk, ale neomezené persistentní datasety",
            "Kdy zvolit": "Kaggle pro opakovanou práci s velkými datasety"
        },
        {
            "Vlastnost / Aspekt": "Propojení s Google Drive",
            "Google Colab (Free)": "Nativní `drive.mount('/content/drive')`",
            "Kaggle Kernels (Free)": "Pouze přes API / přímý upload",
            "Kdy zvolit": "Colab pro soukromé soubory na osobním Disku"
        },
        {
            "Vlastnost / Aspekt": "Komunita & Zpětná vazba",
            "Google Colab (Free)": "Privátní sdílení přes link",
            "Kaggle Kernels (Free)": "Veřejné portfolio, upvoty, diskuze, medaile",
            "Kdy zvolit": "Kaggle pro budování odborného profilu a CV"
        }
    ])
    st.dataframe(cmp_data, hide_index=True, width="stretch")

# ==============================================================================
# TAB 4: BEST PRACTICES PRO KAGGLE PORTFOLIO
# ==============================================================================
with tab4:
    st.subheader("Zlatá pravidla reprezentativního Kaggle Notebooku")

    st.markdown(
        """
        Pokud chcete své notebooky sdílet na **LinkedIn**, v **životopise (CV)** nebo při **pohovoru na pozici Junior Data Scientist / Data Analyst**, 
        dodržujte tyto osvědčené principy:
        """
    )

    col_bp1, col_bp2 = st.columns(2)

    with col_bp1:
        st.markdown(
            """
            #### 📝 1. Struktura a příběh (Storytelling)
            - **Úvodní hlavička:** Stručný popis byznysového cíle, zdroje dat a přehled použitých knihoven.
            - **Hierarchie nadpisů:** Používejte `#`, `##` a `###` pro jasnou navigaci v obsahu notebooku.
            - **Komentáře u grafů:** Ke každé vizualizaci připište **3–4 věty interpretace** (proč je výsledek důležitý pro byznys).
            
            #### 🧹 2. Čistota kódu a skrytí zbytečných výstupů
            - Skryjte dlouhé instalační logy (`pip install ... -q` nebo potlačení varování).
            - Neponechávejte v notebooku prázdné buňky ani buňky s chybovými hláškami (Traceback).
            """
        )

    with col_bp2:
        st.markdown(
            """
            #### 🔄 3. Reprodukovatelnost (Reproducibility)
            - **Fixujte náhodné generátory:** Vždy nastavte `random_state=42` v `train_test_split`, `RandomForest`, `Keras` i `Word2Vec`.
            - **Ověřte "Save & Run All":** Před publikací notebook jednou kompletně restartujte a spusťte odshora dolů.
            
            #### 🏅 4. Získání komunitních upvotů (Kaggle Medals)
            - Vytvořte poutavý název notebooku (např. *„McDonald's Sentiment: Word2Vec + SVM with Aspect Insights“* místo *„Homework 7“*).
            - Přidejte relevantní tagy (*NLP*, *Classification*, *Sentiment Analysis*, *Word2Vec*).
            """
        )

    st.markdown("---")
    st.success(
        "🎓 **Gratulujeme k dokončení série domácích úkolů Session 2!**  \n"
        "Úspěšně jste vypracovali a do aplikace integrovali všech 7 praktických úloh:  \n"
        "1. Titanic (Random Forest – Klasifikace)  \n"
        "2. Ceny automobilů (Random Forest – Regrese)  \n"
        "3. Diabetes (XGBoost – Klasifikace s ydata_profiling)  \n"
        "4. Kalorie při cvičení (XGBoost – Regrese s RandomizedSearchCV)  \n"
        "5. Sonar (Keras MLP – Akustická klasifikace)  \n"
        "6. Auto MPG (Keras MLP – Regrese spotřeby paliva)  \n"
        "7. McDonald's (Gensim Word2Vec + LinearSVC – NLP 3-třídní sentiment)"
    )
