# Workflow: Zpracování teoretických sekcí z PDF resources

Tento dokument definuje závazný postup pro zpracování teoretických podkladů a prezentací ve formátu PDF v rámci kurzu Data Science / Machine Learning.

---

## Cíle a principy zpracování

1. **Důkladné prostudování teorie**:
   - PDF materiály nejsou praktické úkoly k odevzdání, ale teoretický základ kurzu.
   - Text a struktura se extrahují a analyzují v celém rozsahu (definice, matematické odvození, příklady, kódové ukázky).

2. **Logická tematická adresářová struktura**:
   - Pro každý modul kurzu vytvořit dedikovanou složku `theory/` (např. `01_Regression/theory/`, `02_Classification/theory/` atd.).
   - Teoretická shrnutí ukládat jako detailní české Markdown dokumenty (např. `01_linear_regression_theory.md`).

3. **Struktura teoretického dokumentu**:
   - **Originální teorie kurzu (v češtině)**: Přehledně strukturovaný výtah konceptů, matematických formulí, předpokladů a postupů.
   - **Praktické implementační aspekty**: Mapování na Scikit-learn a Python kód, interpretace výsledků a diagnostika.
   - **Jasně odlišené připomínky a kritické zhodnocení**: Upozornění na zjednodušení v kurzu (např. problematika multikolinearity, zavádějící interpretace neškálovaných koeficientů, nebezpečí ignorování heteroskedasticity).

4. **Moderní ML & AI kontext (Stav k 09/2026)**:
   - Kurzové materiály odpovídají starším standardům (cca 2023). V každém shrnutí musí být povinně zařazena sekce moderního rozšíření:
     - **Automatizované a AI-assisted Feature Engineering**: LLM-driven generování hypotéz a příznaků (např. CAAFE), automatická tvorba interakčních členů.
     - **Moderní algoritmy a hybridy**: Explainable Boosting Machines (EBM / InterpretML), Symbolic Regression (PySR), moderní regularizované lineární modely v GPU akceleraci (cuML, JAX/PyTorch).
     - **Pokročilá interpretovatelnost (XAI)**: SHAP hodnoty, Partial Dependence Plots (PDP), Accumulated Local Effects (ALE).
     - **Robustní MLOps a validace**: `sklearn.pipeline.Pipeline` jako prevence data leakage, moderní křížová validace, sledování metrik (MLflow, Weights & Biases).
