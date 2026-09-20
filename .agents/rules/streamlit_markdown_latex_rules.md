# Pravidla a prevence chyb: Streamlit, Markdown, LaTeX a Regex

Tento dokument slouží jako **závazné pravidlo (rule)** pro celý repozitář s cílem eliminovat opakující se chyby při renderování Markdownu, LaTeXu, regulárních výrazů a zobrazení ve Streamlitu.

---

## 1. Příčina a popis chyb (Root Cause Analysis)

### Chyba A: ASCII Bell (`\x07`) při zápisu `\alpha` (Python escape bug)
- **Mechanismus:** V běžném Python řetězci `"..."` je sekvence `\a` speciální escape sekvence pro **ASCII Bell (zvonek, kód `\x07`)**.
- Pokud do kódu nebo JSON generátoru napíšeme:
  ```python
  text = "ladění hyperparametru \alpha"  # <-- CHYBA! Vznikne "ladění hyperparametru \x07lpha"
  ```
- **Důsledek ve Streamlitu:** KaTeX se pokusí zpracovat `$\x07lpha$`. Nerozpozná neplatný řídicí znak a vykreslí v prohlížeči **červené chybové pole `\alpha`**!

### Chyba B: Kolize dolarů (`$`) jako měny s KaTeX matematickými delimitery
- **Mechanismus:** Streamlit používá knihovnu **KaTeX** pro vykreslování matematických vzorců. KaTeX interpretuje znak `$` jako začátek a konec inline matematického výrazu (`$...$`).
- Pokud se v textu zkombinuje matematický výraz s dolarem jako měnou:
  ```markdown
  Nejnižší $\text{MAE} = \$783.83$ při $\alpha = 1.0$.   <-- CHYBA!
  ```
- KaTeX interpretuje první `$` jako začátek vzorce a znak `\$` jako ukončení/chybu, což rozbije zbytek odstavce a vykreslí **červené chybové značky `\text{MAE}` a `\alpha`**.

### Chyba C: Ořezávání dlouhých popisků v postranním panelu Streamlitu
- **Mechanismus:** Postranní lišta Streamlitu (Sidebar) má omezenou šířku.
- Názvy stránek delší než cca 38 znaků (např. `Cvičení 6: Regularizace diamantů – Jupyter Notebook (.ipynb)`) se na běžných monitorech zalamují nebo ořezávají na ošklivé `(Jp...`.

---

## 2. Závazná pravidla pro psaní kódu a Markdownu

### Pravidlo 1: VŽDY používat raw string (`r"..."`) pro LaTeX a Regex
Pokud řetězec obsahuje zpětná lomítka (`\alpha`, `\beta`, `\text{}`, `\frac{}` nebo regulární výrazy `\d+`, `\s+`), **MUSÍ** být uvozen předponou `r`:
```python
# SPRÁVNĚ:
text = r"ladění hyperparametru $\alpha$"
coef_desc = r"$\beta_1 x_1 + \beta_2 x_2$"

# ŠPATNĚ:
text = "ladění hyperparametru $\alpha$"
```

### Pravidlo 2: Pro měnu NIKDY nepoužívat symbol `$` v blízkosti matematických výrazů
- V technických popisech a vzorcích **vždy uvádět měnu slovy nebo ISO kódem**: `USD` nebo `Kč`.
- Příklad:
  ```markdown
  # SPRÁVNĚ:
  Nejnižší MAE = 783.83 USD při $\alpha = 1.0$.
  $\text{MAE} = 783.83\text{ USD}$
  Cena vzrostla o 15 000 USD.

  # ZAKÁZÁNO:
  Nejnižší $\text{MAE} = \$783.83$ při $\alpha = 1.0$.
  Cena vzrostla o $15 000 a $R^2 = 0.85$.
  ```

### Pravidlo 3: Stručné a úderné názvy stránek v `app.py`
Pro názvy stránek v navigaci volit kompaktní formát (max 35 znaků), aby nedocházelo k ořezávání v bočním menu:
```python
# SPRÁVNĚ:
p_reg_diam_nb = st.Page("views/01_diamonds_regularization_nb.py", title="Cvičení 6: Diamanty – Notebook", icon="📓")

# ŠPATNĚ (příliš dlouhé):
p_reg_diam_nb = st.Page("views/01_diamonds_regularization_nb.py", title="Cvičení 6: Regularizace diamantů – Jupyter Notebook (.ipynb)", icon="📓")
```

### Pravidlo 4: Zákaz `use_container_width`, povinně používat `width="stretch"`
V moderním Streamlitu (verze 1.40+) je parametr `use_container_width` napříč všemi komponentami (`st.dataframe`, `st.image`, `st.plotly_chart`, `st.download_button`) označen jako deprecated a zahlcuje systémové logy varováními.
- **Pro roztažení na celou šířku:** VŽDY používat `width="stretch"`.
- **Pro přizpůsobení obsahu:** VŽDY používat `width="content"`.

```python
# SPRÁVNĚ:
st.dataframe(df, width="stretch", hide_index=True)
st.plotly_chart(fig, width="stretch")
st.image(img, width="stretch")
st.download_button("Stáhnout", data=b, width="stretch")

# ZAKÁZÁNO (generuje desítky varování v logu):
st.dataframe(df, use_container_width=True)
st.plotly_chart(fig, use_container_width=True)
```

---

## 3. Automatická obrana (Sanitizer) v `notebook_renderer.py`

Jako systémovou pojistku pro případ, že externí nebo importovaný Jupyter sešit obsahuje nechtěné znaky, zavádíme v `views/notebook_renderer.py` automatickou sanitaci:
1. Automatické nahrazení `\x07lpha` zpět na `\alpha`.
2. Automatické ošetření `\$` uvnitř matematických bloků a převod na formát `USD`.

