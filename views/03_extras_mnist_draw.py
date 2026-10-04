"""
Den 3 Extras: ✍️ Nakresli číslici – živé rozpoznávání MNIST neuronovou sítí
==========================================================================
- Vlastní kreslicí komponenta (components/digit_canvas) bez externích závislostí.
- Předzpracování ve stylu MNIST: ořez bounding boxu -> zmenšení na 20x20 -> vložení do 28x28
  -> vycentrování podle těžiště (center of mass).
- Inference čistě v numpy z exportovaných vah (784 -> 128 ReLU -> 10 Softmax), žádný TensorFlow/PyTorch.
- Záložka "Test robustnosti": posuny, rotace a šum na testovacích číslicích => proč MLP potřebuje CNN.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import ndimage

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "03_Advanced_ML_Neural_Networks" / "data" / "mnist_mlp_weights.npz"
SAMPLES_PATH = BASE_DIR / "03_Advanced_ML_Neural_Networks" / "data" / "mnist_mlp_exercise_1_precomputed.json"
CANVAS_DIR = BASE_DIR / "components" / "digit_canvas"

_digit_canvas = components.declare_component("digit_canvas", path=str(CANVAS_DIR))


@st.cache_resource
def load_weights():
    w = np.load(WEIGHTS_PATH)
    return w["W1"], w["b1"], w["W2"], w["b2"], float(w["test_accuracy"][0])


@st.cache_data
def load_test_samples():
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sample_predictions"]


def forward(x28: np.ndarray):
    """Dopředný průchod MLP v numpy. Vrací (pravděpodobnosti, aktivace skryté vrstvy)."""
    W1, b1, W2, b2, _ = load_weights()
    x = x28.reshape(1, 784).astype(np.float32)
    h = np.maximum(0.0, x @ W1 + b1)
    logits = h @ W2 + b2
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    return p.ravel(), h.ravel()


def mnist_preprocess(img: np.ndarray):
    """Převede libovolný obrázek číslice (0..1, bílá na černé) do formátu MNIST 28x28."""
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    mask = img > 0.05
    if mask.sum() < 5:
        return None
    rows, cols = np.where(mask)
    crop = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    small = np.clip(ndimage.zoom(crop, scale, order=1), 0.0, 1.0)
    sh, sw = small.shape
    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - sh) // 2, (28 - sw) // 2
    out[top:top + sh, left:left + sw] = small[:28 - top, :28 - left]
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (14 - cy, 14 - cx), order=1, mode="constant", cval=0.0)
    return np.clip(out, 0.0, 1.0)


def probability_bar(probs: np.ndarray, highlight: int, title: str, true_label=None):
    colors = []
    for d in range(10):
        if true_label is not None and d == true_label:
            colors.append("#10b981")
        elif d == highlight:
            colors.append("#6366f1" if true_label is None or highlight == true_label else "#ef4444")
        else:
            colors.append("#64748b")
    fig = go.Figure(go.Bar(x=list(range(10)), y=probs * 100, marker_color=colors,
                           text=[f"{p * 100:.1f}%" for p in probs], textposition="outside"))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(tickmode="linear", dtick=1, title="Číslice"),
                      yaxis=dict(range=[0, 112], title="Pravděpodobnost (%)"))
    return fig


def digit_heatmap(img: np.ndarray, title: str, height: int = 300):
    fig = px.imshow(img, color_continuous_scale="gray", zmin=0, zmax=1, title=title)
    fig.update_layout(coloraxis_showscale=False, height=height, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def entropy_bits(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


# =============================================================================
# UI
# =============================================================================
st.title("✍️ Nakresli číslici – živé rozpoznávání neuronovou sítí")
st.caption(
    "Neuronová síť 784 → 128 (ReLU) → 10 (Softmax) natrénovaná na 60 000 číslicích MNIST. "
    "Inference běží přímo v numpy z exportovaných vah – žádný TensorFlow ani PyTorch na serveru."
)

if not WEIGHTS_PATH.exists():
    st.error("Chybí soubor s vahami. Spusťte `03_Advanced_ML_Neural_Networks/10_day3_extras_precompute.py`.")
    st.stop()

W1, b1, W2, b2, test_acc = load_weights()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Přesnost na MNIST testu", f"{test_acc * 100:.2f} %", delta="10 000 testovacích číslic")
k2.metric("Trénovatelné váhy", "101 770", delta="784·128+128 + 128·10+10")
k3.metric("Velikost modelu", "≈ 0.4 MB", delta="float32, komprimované npz")
k4.metric("Inference", "< 1 ms", delta="2 násobení matic v numpy")

st.markdown("---")
tab_draw, tab_robust, tab_inside = st.tabs([
    "🎨 Kreslicí plátno",
    "🧪 Test robustnosti (posun, rotace, šum)",
    "🔬 Co vidí neurony (váhy 1. vrstvy)",
])

# -----------------------------------------------------------------------------
# TAB 1: KRESLENÍ
# -----------------------------------------------------------------------------
with tab_draw:
    c_canvas, c_result = st.columns([1, 1.25])
    with c_canvas:
        st.markdown("##### 1️⃣ Nakreslete číslici")
        value = _digit_canvas(key="mnist_canvas", default=None)
        st.caption("Tip: kreslete velkou číslici přes většinu plátna. Po puštění myši se predikce aktualizuje.")

    with c_result:
        st.markdown("##### 2️⃣ Predikce sítě")
        pixels = value.get("pixels") if isinstance(value, dict) else None
        if pixels is None:
            st.info("👈 Nakreslete číslici na černé plátno vlevo.")
        else:
            raw = np.array(pixels, dtype=np.float32)
            x28 = mnist_preprocess(raw)
            if x28 is None:
                st.warning("Plátno je téměř prázdné – zkuste nakreslit výraznější tah.")
            else:
                probs, hidden = forward(x28)
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                H = entropy_bits(probs)

                m1, m2, m3 = st.columns(3)
                m1.metric("Predikovaná číslice", f"{pred}")
                m2.metric("Jistota (max Softmax)", f"{conf * 100:.1f} %")
                m3.metric("Entropie nejistoty", f"{H:.2f} bit", delta="0 = jistota, 3.32 = náhoda", delta_color="off")

                if conf >= 0.9:
                    st.success(f"Síť si je jistá: je to **{pred}**.")
                elif conf >= 0.6:
                    second = int(np.argsort(probs)[-2])
                    st.warning(f"Síť váhá – nejspíš **{pred}**, ale zvažuje i **{second}** ({probs[second] * 100:.1f} %).")
                else:
                    st.error("Síť je zmatená. Zkuste číslici nakreslit zřetelněji nebo větší.")

                st.plotly_chart(probability_bar(probs, pred, "Výstup Softmax vrstvy"), key="draw_bar")

    if pixels is not None and (x28 := mnist_preprocess(np.array(pixels, dtype=np.float32))) is not None:
        st.markdown("##### 3️⃣ Co přesně dostane síť na vstup (předzpracování ve stylu MNIST)")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.plotly_chart(digit_heatmap(np.array(pixels), "Surové plátno (56×56)", 260), key="raw56")
        with p2:
            st.plotly_chart(digit_heatmap(x28, "Vstup sítě 28×28 (vycentrováno)", 260), key="in28")
        with p3:
            st.markdown(
                r"""
                **Pipeline předzpracování:**
                1. Ořez na bounding box tahu
                2. Zmenšení delší strany na **20 px** (zachování poměru stran)
                3. Vložení do plátna **28×28**
                4. Posun **těžiště** do středu (14, 14)
                5. `Flatten` → vektor **784** hodnot v intervalu ⟨0, 1⟩

                Přesně tak vznikl původní dataset MNIST (NIST, LeCun 1998).
                Bez kroků 1–4 by přesnost na ručně kreslených číslicích dramaticky klesla – viz záložka *Test robustnosti*.
                """
            )

# -----------------------------------------------------------------------------
# TAB 2: ROBUSTNOST
# -----------------------------------------------------------------------------
with tab_robust:
    st.markdown(
        r"""
        Plně propojená síť (MLP) **nemá prostorovou invarianci** – po `Flatten` je pro ni obrázek jen vektor 784 čísel.
        Posun číslice o pár pixelů tak aktivuje úplně jiné váhy. Vyzkoušejte to na testovacích číslicích:
        """
    )
    samples = [s for s in load_test_samples() if s["is_correct"]]
    opts = {f"Testovací vzorek #{s['test_index']} (číslice {s['true_label']})": s for s in samples}
    chosen = opts[st.selectbox("Vyberte testovací číslici:", list(opts.keys()))]
    base_img = np.array(chosen["image_pixels"], dtype=np.float32)

    s1, s2, s3, s4 = st.columns(4)
    dx = s1.slider("Posun vodorovně (px)", -8, 8, 0)
    dy = s2.slider("Posun svisle (px)", -8, 8, 0)
    rot = s3.slider("Rotace (°)", -60, 60, 0, step=5)
    noise = s4.slider("Šum (σ)", 0.0, 0.6, 0.0, step=0.05)

    img = ndimage.rotate(base_img, rot, reshape=False, order=1) if rot else base_img.copy()
    img = ndimage.shift(img, (dy, dx), order=1, mode="constant", cval=0.0)
    if noise > 0:
        img = img + np.random.default_rng(42).normal(0, noise, img.shape)
    img = np.clip(img, 0, 1)

    pr_orig, _ = forward(base_img)
    pr_mod, _ = forward(img)
    pred_mod = int(np.argmax(pr_mod))

    r1, r2, r3 = st.columns([1, 1, 1.4])
    with r1:
        st.plotly_chart(digit_heatmap(base_img, f"Originál – predikce {int(np.argmax(pr_orig))}", 280), key="rob_orig")
    with r2:
        st.plotly_chart(digit_heatmap(img, f"Upraveno – predikce {pred_mod}", 280), key="rob_mod")
    with r3:
        st.plotly_chart(probability_bar(pr_mod, pred_mod, "Softmax po úpravě", true_label=chosen["true_label"]), key="rob_bar")

    # Systematický sken posunů
    st.markdown("##### 📉 Systematický sken: přesnost na všech 30 vzorcích při vodorovném posunu")
    shifts = list(range(-8, 9))
    acc_by_shift = []
    for s in shifts:
        ok = 0
        for smp in samples:
            im = ndimage.shift(np.array(smp["image_pixels"], dtype=np.float32), (0, s), order=1)
            ok += int(np.argmax(forward(im)[0]) == smp["true_label"])
        acc_by_shift.append(ok / len(samples) * 100)
    fig_shift = go.Figure(go.Scatter(x=shifts, y=acc_by_shift, mode="lines+markers", line=dict(color="#6366f1", width=3)))
    fig_shift.add_hrect(y0=95, y1=100, fillcolor="#10b981", opacity=0.08, line_width=0)
    fig_shift.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                            xaxis_title="Vodorovný posun (px)", yaxis_title="Přesnost (%)", yaxis_range=[0, 105])
    st.plotly_chart(fig_shift, key="shift_scan")
    st.info(
        "Už posun o 4–6 px (což je pro člověka zanedbatelné) srazí přesnost MLP dramaticky dolů. "
        "Konvoluční sítě (Conv2D + MaxPooling) sdílí váhy filtrů přes celý obrázek, a proto jsou vůči posunu výrazně odolnější."
    )

# -----------------------------------------------------------------------------
# TAB 3: CO VIDÍ NEURONY
# -----------------------------------------------------------------------------
with tab_inside:
    st.markdown(
        r"""
        Každý ze 128 skrytých neuronů má 784 vah – jednu pro každý pixel. Když je přeskládáme zpět do mřížky 28×28,
        uvidíme **„šablonu“, na kterou neuron reaguje** (modrá = pixel neuron aktivuje, červená = potlačuje).
        """
    )
    n_show = st.slider("Počet zobrazených neuronů", 8, 32, 16, step=8)
    order_mode = st.radio("Pořadí neuronů", ["Podle velikosti vah (L2 norma)", "Podle aktivace na vybrané testovací číslici"], horizontal=True)
    if order_mode.startswith("Podle aktivace"):
        _, hid = forward(base_img)
        order = np.argsort(hid)[::-1][:n_show]
        st.caption(f"Neurony nejvíce aktivované testovací číslicí {chosen['true_label']} (vybranou v záložce Test robustnosti).")
    else:
        order = np.argsort(np.linalg.norm(W1, axis=0))[::-1][:n_show]

    cols_per_row = 8
    for row_start in range(0, n_show, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, neuron in enumerate(order[row_start:row_start + cols_per_row]):
            w = W1[:, neuron].reshape(28, 28)
            lim = float(np.abs(w).max())
            fig = px.imshow(w, color_continuous_scale="RdBu", zmin=-lim, zmax=lim)
            fig.update_layout(coloraxis_showscale=False, height=120, margin=dict(l=0, r=0, t=18, b=0),
                              title=dict(text=f"#{neuron}", font=dict(size=11)))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            cols[j].plotly_chart(fig, key=f"neuron_{neuron}_{row_start}")

    st.markdown("##### Kterou číslici který neuron „podporuje“ (váhy 2. vrstvy W2)")
    w2_df = pd.DataFrame(W2[order], index=[f"#{n}" for n in order], columns=[str(d) for d in range(10)])
    fig_w2 = px.imshow(w2_df, color_continuous_scale="RdBu", zmin=-float(np.abs(W2).max()), zmax=float(np.abs(W2).max()),
                       labels=dict(x="Výstupní číslice", y="Skrytý neuron", color="Váha"), aspect="auto")
    fig_w2.update_layout(height=max(300, 22 * n_show), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_w2, key="w2_heat")
