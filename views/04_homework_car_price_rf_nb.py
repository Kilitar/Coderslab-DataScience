from views.notebook_renderer import render_jupyter_notebook

render_jupyter_notebook(
    nb_rel_path="04_Homework/02_car_price_random_forest.ipynb",
    title="📓 DÚ: Random Forest (Ceny aut) – Notebook",
    description="Kompletní vypracované řešení cvičení Random Forest Regressor na predikci cen ojetých vozů (car_data.csv) s inspekcí přes ydata_profiling, nápravou Excel date auto-format bugu, filtrací anomálií, One-Hot kódováním, laděním hyperparametrů pomocí GridSearchCV přes MSE a vyhodnocením přes MAE a RMSE."
)
