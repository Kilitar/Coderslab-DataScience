"""
Den 3: Neuronové sítě – Cvičení 1 (Klasifikace číslic MNIST) – Výsledky zadání
=============================================================================
Zpracování oficiálního zadání kurzu v Keras:
1. Načtení dat keras.datasets.mnist (60 000 train, 10 000 test).
2. Předzpracování: Normalizace pixelů (0-1) a to_categorical na labelech.
3. Architektura: Vstupní vrstva Flatten (784) -> Skrytá vrstva Dense(128, relu) -> Výstupní Dense(10, softmax).
4. Kompilace s Adam, categorical_crossentropy a metrikou accuracy.
5. Trénování sítě na trénovacích datech (15 epoch, batch_size=256, validation_split=0.15).
6. Predikce na testovací sadě X_test.
7. Inspekce a vizualizace vybraného obrázku (.imshow) s predikovaným a skutečným štítkem.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_mnist_data():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "mnist_mlp_exercise_1_precomputed.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_neural_network_mnist_exercise_1_view():
    st.title("🎯 Cvičení 1: Rozpoznávání číslic MNIST (Keras MLP)")
    st.markdown(
        r"""
        V tomto cvičení stavíme plně propojenou neuronovou síť (**Multilayer Perceptron – MLP**) v knihovně 
        **TensorFlow / Keras** pro rozpoznávání rukou psaných číslic **0 až 9** ze slavného datasetu **MNIST**.
        """
    )

    data = load_mnist_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `03_Advanced_ML_Neural_Networks/08_neural_network_mnist_exercise_1.py`.")
        return

    meta = data["metadata"]
    metrics = data["metrics"]
    history = data["history"]
    samples = data["sample_predictions"]

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Přesnost na testovací sadě",
            value=f"{metrics['test_accuracy']*100:.2f} %",
            delta="9 750 z 10 000 správně"
        )
    with k2:
        st.metric(
            label="Trénovatelné váhy",
            value=f"{meta['parameters_breakdown']['total_trainable_params']:,}",
            delta="784 -> 128 -> 10 neuronů"
        )
    with k3:
        st.metric(
            label="Doba trénování (15 epoch)",
            value=f"{meta['training_duration_s']:.2f} s",
            delta="Batch size: 256"
        )
    with k4:
        st.metric(
            label="Trénovací vzorky",
            value=f"{meta['n_train_split']:,}",
            delta=f"Validační split: {meta['n_val_split']:,} (15 %)"
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Výsledky & Křivky učení",
        "🔍 Interaktivní vizualizátor číslic",
        "🔢 Matice záměn (Confusion Matrix)",
        "💻 Zdrojový kód zadání"
    ])

    # =========================================================================
    # TAB 1: VÝSLEDKY & KŘIVKY
    # =========================================================================
    with tab1:
        st.subheader("1. Průběh trénování sítě přes 15 epoch")
        st.markdown(
            r"""
            Model byl zkompilován s optimalizátorem **Adam**, ztrátovou funkcí **`categorical_crossentropy`** 
            a trénován po dobu 15 epoch na dávkách po 256 vzorcích.
            """
        )

        epochs = history["epoch"]
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=epochs, y=history["train_acc"], mode="lines", name="Trénovací přesnost", line=dict(color="#3b82f6", width=2)))
        fig_h.add_trace(go.Scatter(x=epochs, y=history["val_acc"], mode="lines+markers", name="Validační přesnost", line=dict(color="#10b981", width=2)))
        fig_h.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Trénovací ztráta (Loss)", line=dict(color="#f59e0b", dash="dash")))
        fig_h.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Validační ztráta (Loss)", line=dict(color="#ef4444", dash="dash")))

        fig_h.update_layout(
            title="Vývoj přesnosti (Accuracy) a ztrátové funkce (Loss) během trénování",
            xaxis_title="Epocha",
            yaxis_title="Hodnota metriky",
            height=430,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_h)

        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.info(
                f"""
                #### 📈 Finální hodnoty v epoše 15:
                - **Trénovací přesnost:** {history['train_acc'][-1]*100:.2f} %
                - **Validační přesnost:** {history['val_acc'][-1]*100:.2f} %
                - **Trénovací ztráta:** {history['train_loss'][-1]:.4f}
                - **Validační ztráta:** {history['val_loss'][-1]:.4f}
                """
            )
        with c_a2:
            st.success(
                f"""
                #### 🎯 Výsledek na testovací sadě:
                - **Testovací Accuracy:** **{metrics['test_accuracy']*100:.2f} %**
                - **Chybovost:** Pouze 2.50 % (250 chyb z 10 000 obrázků)
                - Na jednoduchou plně propojenou síť bez konvolucí jde o vynikající výsledek!
                """
            )

    # =========================================================================
    # TAB 2: INTERAKTIVNÍ VIZUALIZÁTOR ČÍSLIC
    # =========================================================================
    with tab2:
        st.subheader("2. Interaktivní vizualizace predikcí pro vybrané obrázky")
        st.markdown(
            r"""
            V souladu se zadáním provádíme inspekci konkrétního obrázku pomocí **`plt.imshow()`**, 
            kontrolujeme skutečný štítek (**True Label**) a predikci modelu (**Predicted Label**).
            """
        )

        sample_options = [
            f"Vzorek #{s['test_index']} (Skutečnost: {s['true_label']}, Predikce: {s['predicted_label']})"
            + (" ✅" if s["is_correct"] else " ❌ CHYBA")
            for s in samples
        ]
        chosen_opt = st.selectbox("Vyberte testovací vzorek k zobrazení:", sample_options)
        chosen_idx = sample_options.index(chosen_opt)
        item = samples[chosen_idx]

        col_img, col_bar = st.columns([1, 1])
        with col_img:
            # Vykreslení 28x28 matice jako grayscale heatmapy
            pixels = np.array(item["image_pixels"])
            fig_img = px.imshow(
                pixels,
                color_continuous_scale="gray",
                title=f"Číslice #{item['test_index']} – Skutečnost: {item['true_label']}"
            )
            fig_img.update_layout(
                coloraxis_showscale=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_img)

            if item["is_correct"]:
                st.success(f"✅ Model správně rozpoznal číslici **{item['predicted_label']}**!")
            else:
                st.error(f"❌ Model se spletl! Predikoval **{item['predicted_label']}**, ale ve skutečnosti jde o **{item['true_label']}**.")

        with col_bar:
            # Sloupcový graf pravděpodobností výstupu Softmax
            digits = list(range(10))
            probs = [p * 100 for p in item["probabilities"]]
            bar_colors = ["#10b981" if d == item["true_label"] else ("#ef4444" if d == item["predicted_label"] and not item["is_correct"] else "#94a3b8") for d in digits]

            fig_bar = go.Figure(data=[
                go.Bar(x=digits, y=probs, marker_color=bar_colors)
            ])
            fig_bar.update_layout(
                title="Výstupní pravděpodobnosti Softmax (%)",
                xaxis=dict(tickmode="linear", tick0=0, dtick=1, title="Číslice (0-9)"),
                yaxis=dict(title="Pravděpodobnost (%)", range=[0, 105]),
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar)

    # =========================================================================
    # TAB 3: CONFUSION MATRIX
    # =========================================================================
    with tab3:
        st.subheader("3. Matice záměn (Confusion Matrix) pro 10 tříd")
        cm_data = np.array(metrics["confusion_matrix"])

        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predikovaná číslice", y="Skutečná číslice", color="Počet vzorků"),
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Matice záměn na 10 000 testovacích obrázcích"
        )
        fig_cm.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm)

        st.markdown("#### Detailní metriky pro jednotlivé číslice")
        rep = metrics["classification_report"]
        rep_rows = []
        for d in range(10):
            d_str = str(d)
            if d_str in rep:
                rep_rows.append({
                    "Číslice": d,
                    "Precision": f"{rep[d_str]['precision']*100:.2f} %",
                    "Recall": f"{rep[d_str]['recall']*100:.2f} %",
                    "F1-Score": f"{rep[d_str]['f1-score']*100:.2f} %",
                    "Podpora (Počet v testu)": int(rep[d_str]['support'])
                })
        st.dataframe(pd.DataFrame(rep_rows))

    # =========================================================================
    # TAB 4: ZDROJOVÝ KÓD
    # =========================================================================
    with tab4:
        st.subheader("4. Kompletní kód řešení dle zadání")
        st.code(
            r'''
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Načtení dat přímo z knihovny Keras
mnist_dataset = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_dataset.load_data()

# 2. Předzpracování: Normalizace pixelů na interval [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Převod cílové proměnné na One-Hot (kategoriální) formát
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# 3. Sestavení architektury sítě
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Vstupní vrstva: 784 pixelů
    Dense(128, activation="relu"),          # Skrytá vrstva: 128 neuronů s ReLU
    Dense(10, activation="softmax")         # Výstupní vrstva: 10 tříd se Softmax
])

# 4. Kompilace modelu
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Trénování sítě
history = model.fit(
    X_train,
    y_train_cat,
    epochs=15,
    batch_size=256,
    validation_split=0.15,
    verbose=1
)

# 6. Predikce na testovací sadě X_test
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# 7. Kontrola a vizualizace vybraného obrázku
sample_idx = 42
actual_label = y_test[sample_idx]
predicted_label = y_pred_labels[sample_idx]

print(f"Skutečný štítek (True Label):      {actual_label}")
print(f"Predikovaný štítek (Predicted):   {predicted_label}")

plt.figure(figsize=(4, 4))
plt.imshow(X_test[sample_idx], cmap="gray")
plt.title(f"Predikce: {predicted_label} (Skutečnost: {actual_label})")
plt.axis("off")
plt.show()
            ''',
            language="python"
        )


if __name__ == "__main__":
    render_neural_network_mnist_exercise_1_view()
