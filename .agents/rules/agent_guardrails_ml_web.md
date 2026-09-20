# Závazná pravidla a Guardrails pro vývoj ML webové aplikace (Coderslab Portfolio)

Tento dokument je **závazným pravidlem (rule)** pro všechny AI agenty pracující v tomto repozitáři. Shrnuje klíčová architektonická a implementační pravidla, aby se předešlo opakování chyb (pády ASGI, zamrzání aplikace, varování KaTeXu, syntaktické chyby a datové nekonzistence).

---

## 1. Stabilita cloudového běhu (Streamlit Community Cloud & ASGI)

### Pravidlo 1.1: ZÁKAZ těžkého trénování přímo ve webovém vlákně
- **Problém:** Streamlit Community Cloud běží na omezeném sdíleném CPU (~0.5 vCPU). Spouštění `fit()`, křížové validace (`RidgeCV`, `LassoCV`, `ElasticNetCV`) nebo generování tisíců polynomiálních příznaků ve webovém skriptu zablokuje vlákno na 1–3 minuty.
- **Důsledek:** Asyncio smyčka přestane odpovídat na WebSocket pingy, spojení se rozpadne a vyvolá pád ASGI:  
  `RuntimeError: WebSocket is not connected. Need to call "accept" first.`
- **Řešení:**
  1. Veškeré časově náročné výpočty, metriky a data pro grafy **vždy předpočítejte offline** do lehkého JSON souboru (`01_Regression/data/<nazev>_precomputed.json`).
  2. Webová stránka ve `views/` musí data načítat primárně z tohoto JSON souboru za < 5 ms.
  3. Trénovací kód ponechte pouze jako fallback pro případ, že JSON chybí.

### Pravidlo 1.2: Pevné verzování Pythonu
- V souboru `.python-version` v kořenu repozitáře **MUSÍ být trvale nastavena hodnota `3.11`**.
- Nikdy nenastavovat experimentální verze (např. 3.14), pro které neexistují linuxové binární balíčky (wheels) pro `scikit-learn`, `scipy` a `xgboost`.

### Pravidlo 1.3: Zákaz `use_container_width`
- V moderním Streamlitu (1.40+) je `use_container_width=True` deprecated.
- **VŽDY** používat `width="stretch"` pro roztažení komponent (`st.dataframe`, `st.plotly_chart`, `st.image`, `st.download_button`).

---

## 2. Typografie, KaTeX a Markdown

### Pravidlo 2.1: VŽDY používat raw stringy `r"""..."""` pro text s LaTeXem
- Pokud text obsahuje LaTeX vzorce se zpětnými lomítky (`\alpha`, `\sum`, `\beta`, `\times`, `\frac`), **MUSÍ** být řetězec uvozen jako raw string: `st.markdown(r"""...""")`.
- Běžný řetězec `"""..."""` způsobuje v Pythonu 3.12+ `SyntaxWarning: invalid escape sequence '\s'` (v budoucích verzích fatální SyntaxError).

### Pravidlo 2.2: ZÁKAZ používání symbolu `$` pro měnu
- Měnu v textech, grafech a tabulkách **VŽDY** uvádějte textem: `USD` nebo `Kč`.
- Nikdy nepsat `$118 000` nebo `118 000 $` – KaTeX to interpretuje jako začátek matematického módu, což převrátí formátování a chrlí laviny chyb `unicodeTextInMathMode` pro česká písmena.

### Pravidlo 2.3: Zákaz destruktivních globálních regulárních výrazů
- Nikdy nepoužívat globální nahrazení typu `r"\$(\d+)" -> r"\1 USD"`. Tento regex uřízne otevírací dolar u matematických čísel (např. `$0{,}0012$` nebo `$50$`) a rozbije paritu dolarů v celém dokumentu.

### Pravidlo 2.4: Běžné zkratky metrik nepatří do KaTeXu
- Zkratky `MAE`, `MSE`, `RMSE`, `R2` v běžném textu pište přímo jako text, nikoliv v LaTeXových blocích `$\text{MAE}$` nebo `$MAE$`.

---

## 3. Metodická pravidla pro Machine Learning (ml-best-practices)

### Pravidlo 3.1: Striktní featurizace a prevence Data Leakage
1. **Rozdělení dat PŘED jakoukoliv transformací:**  
   Nejprve provést `train_test_split(..., random_state=42)`.
2. **Škálování a transformace:**  
   `StandardScaler` nebo `PolynomialFeatures` se **fituje výhradně na trénovací sadě** (`scaler.fit_transform(X_train)`), na testovací sadě se volá pouze `scaler.transform(X_test)`.
3. **Pipeline:**  
   Při ladění hyperparametrů a křížové validaci **vždy používat `sklearn.pipeline.Pipeline`**.

### Pravidlo 3.2: Polynomiální regrese a kletba dimenzionality (Curse of Dimensionality)
- **Zadání:** Diamanty mají 53 940 řádků a po One-Hot kódování ~26 sloupců.
- **Kardinální chyba ze zadání kurzu:** Aplikovat `PolynomialFeatures(degree=2 nebo 3)` slepě na všechny sloupce včetně dummy proměnných:
  - U binárních příznaků platí $\text{cut\_Premium}^2 = \text{cut\_Premium}$ (redundantní, vzniká dokonalá multikolinearita).
  - Stupeň 3 na 26 příznacích generuje $\approx 3\,654$ sloupců, což na 54k řádcích vyžaduje gigabajty paměti a vede k přeučení (overfitting) a kolapsu OLS.
- **Správný expertní přístup v rozboru:**
  - Porovnat variantu dle zadání (učebnicový postup) s expertní variantou: polynomiální expanze pouze na spojitých fyzikálních veličinách (`carat`, `x`, `y`, `z`, `depth`, `table`) v kombinaci s regularizací (Ridge / Lasso).
