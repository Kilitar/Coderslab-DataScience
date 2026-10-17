"""
Den 5: Neřízené učení – Expertní rozbor cvičení LOF na datech blobs_dataset.csv
==============================================================================
Hluboká metodická analýza a kritika:
1. Matematika kontaminace: Proč LOF skóre nemění hodnotu a jak funguje percentilový posun offsetu.
2. Kompromis Precision vs. Recall a interaktivní Cost-Matrix optimalizátor prahu.
3. Volba parametru k (n_neighbors), fenomén maskování anomálií (Outlier Masking).
4. Srovnání detektorů anomálií: LOF vs. Isolation Forest vs. One-Class SVM vs. Elliptic Envelope.
5. Produkční guardrails pro nasazení LOF v praxi 2026 (Novelty Detection vs. Outlier Detection, O(n^2)).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Cvičení: Blobs – Expertní rozbor", page_icon="🔬", layout="wide")

st.title("🔬 Expertní rozbor: Local Outlier Factor & Kalibrace kontaminace")
st.caption(
    "Pokročilá metodická analýza detekce anomálií na datech blobs_dataset.csv: Matematika percentilového řezu, "
    "ekonomická optimalizace rozhodovacího prahu přes nákladovou matici, fenomén maskování a produkční limity LOF."
)


@st.cache_data
def load_lof_blobs_critique_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/lof_blobs_exercise_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/lof_blobs_exercise_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "points" in d and "metrics" in d:
                        return d
            except Exception:
                pass

    # Rychlý fallback výpočet
    data_candidates = [
        Path("data/Downloadable materials for Session 3/blobs_dataset.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/blobs_dataset.csv",
        Path("data/blobs_dataset.csv"),
    ]
    csv_path = None
    for cand in data_candidates:
        if cand.exists():
            csv_path = cand
            break

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.5, random_state=42)
        df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

    X_scaled = StandardScaler().fit_transform(df[["feature_1", "feature_2"]].values)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    lof.fit_predict(X_scaled)
    scores = -lof.negative_outlier_factor_

    pts = []
    for i in range(len(df)):
        pts.append({
            "id": i,
            "f1_raw": float(df["feature_1"].iloc[i]),
            "f2_raw": float(df["feature_2"].iloc[i]),
            "lof_score": float(scores[i])
        })

    return {
        "metrics": {"total_samples": len(df)},
        "points": pts
    }


crit_data = load_lof_blobs_critique_data()
pts_df = pd.DataFrame(crit_data["points"])

tab_contam, tab_cost, tab_kparam, tab_bench, tab_prod = st.tabs([
    "📐 Matematika kontaminace",
    "💰 Nákladová matice & Práh",
    "👥 Volba sousedů (k) & Maskování",
    "⚔️ LOF vs. iForest vs. OC-SVM",
    "🛡️ Produkční guardrails 2026"
])

# -----------------------------------------------------------------------------
# TAB 1: Matematika kontaminace
# -----------------------------------------------------------------------------
with tab_contam:
    st.subheader("📐 Jak Scikit-learn vnitřně zachází s parametrem contamination?")
    st.markdown(
        r"""
        Mnoho datových vědců se mylně domnívá, že parametr `contamination` ovlivňuje samotný výpočet lokálních hustot 
        nebo skóre LOF. **To je omyl.**
        
        ### Algoritmický princip dvou fází:
        1. **Fáze 1 (Strukturální výpočet):**
           - Spočte se matice vzdáleností mezi všemi body.
           - Pro každý bod se nalezne jeho $k$-okolí, určí se dosažitelné vzdálenosti a spočte se **lokální dosažitelná hustota** ($\text{lrd}$).
           - Výsledkem je spojité **LOF skóre** pro každé pozorování $i$:
             $$\text{LOF}(p) = \frac{\frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \text{lrd}(o)}{\text{lrd}(p)}$$
           - Scikit-learn ukládá tuto hodnotu do atributu `negative_outlier_factor_ = -LOF`.
        
        2. **Fáze 2 (Aplikace rozhodovacího řezu):**
           - Teprve zde vstupuje do hry parametr `contamination` ($c \in (0, 0.5]$).
           - Vypočítá se empirický percentil z pole záporných skóre:
             $$\text{offset\_} = \text{np.percentile}(\text{negative\_outlier\_factor\_}, 100 \times c)$$
           - Následně se binárně rozhodne:
             $$\hat{y}_i = \begin{cases} -1 \text{ (Outlier)} & \text{pokud } \text{negative\_outlier\_factor\_}_i < \text{offset\_} \\ +1 \text{ (Inlier)} & \text{pokud } \text{negative\_outlier\_factor\_}_i \ge \text{offset\_} \end{cases}$$
        """
    )

    st.info(
        "💡 **Klíčový důsledek pro praxi:** Pokud změníte `contamination` z 2 % na 5 %, množina anomálií při 2 % je "
        "**vždy striktní podmnožinou** anomálií při 5 %. Žádný bod označený za anomálii při 2 % se nemůže při 5 % stát inlierem!"
    )

    # Vizuální percentilový graf
    sorted_lofs = np.sort(pts_df["lof_score"].values)
    percentiles = np.linspace(0, 100, len(sorted_lofs))

    fig_perc = go.Figure()
    fig_perc.add_trace(
        go.Scatter(
            x=percentiles,
            y=sorted_lofs,
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name="Kumulativní kvantily LOF"
        )
    )
    # 2% bod (98. percentil)
    p98_val = np.percentile(sorted_lofs, 98)
    p95_val = np.percentile(sorted_lofs, 95)

    fig_perc.add_trace(
        go.Scatter(
            x=[98, 98],
            y=[sorted_lofs.min(), p98_val],
            mode="lines+markers",
            line=dict(color="#ef4444", dash="dash"),
            name=f"2 % anomálií (Práh = {p98_val:.3f})"
        )
    )
    fig_perc.add_trace(
        go.Scatter(
            x=[95, 95],
            y=[sorted_lofs.min(), p95_val],
            mode="lines+markers",
            line=dict(color="#f59e0b", dash="dot"),
            name=f"5 % anomálií (Práh = {p95_val:.3f})"
        )
    )
    fig_perc.update_layout(
        title="Kvantilová funkce LOF skóre (Empirický přechod od jádra k anomáliím)",
        xaxis_title="Percentil pozorování (%)",
        yaxis_title="LOF hodnota",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_perc, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: Nákladová matice & Práh
# -----------------------------------------------------------------------------
with tab_cost:
    st.subheader("💰 Ekonomická optimalizace rozhodovacího prahu: Precision vs. Recall")
    st.markdown(
        """
        V reálném byznysu (detekce platebních podvodů, monitoring selhání výrobních strojů) není volba 
        mezi 2 % a 5 % kontaminace otázkou intuice, ale **ekonomických nákladů**.
        """
    )

    col_cost_input1, col_cost_input2 = st.columns(2)
    with col_cost_input1:
        cost_fp = st.number_input(
            "Náklad na Falešný poplach (False Positive, Kč)",
            min_value=10,
            max_value=5000,
            value=200,
            step=50,
            help="Náklad na zbytečnou manuální kontrolu nevinného klienta nebo rutinní zastavení linky."
        )
    with col_cost_input2:
        cost_fn = st.number_input(
            "Náklad na Nezjištěnou anomálii (False Negative, Kč)",
            min_value=500,
            max_value=100000,
            value=5000,
            step=500,
            help="Přímá finanční ztráta z neprovedené blokace podvodu nebo destrukce neudržovaného ložiska."
        )

    # Interaktivní simulace nákladů napříč různými prahy kontaminace
    contam_range = np.linspace(0.005, 0.10, 20)
    sim_costs = []
    # Předpokládejme, že "pravých anomálií" v datech je cca 25 (odpovídá izolovaným bodům)
    true_anomaly_ids = set(pts_df.sort_values(by="lof_score", ascending=False).head(25)["id"].tolist())

    for c in contam_range:
        cutoff = np.percentile(pts_df["lof_score"], 100 * (1 - c))
        pred_outlier_ids = set(pts_df[pts_df["lof_score"] >= cutoff]["id"].tolist())

        # FP: Predikováno jako outlier, ale není mezi pravými
        fp = len(pred_outlier_ids - true_anomaly_ids)
        # FN: Je pravá anomálie, ale nepredikována
        fn = len(true_anomaly_ids - pred_outlier_ids)

        total_cost = (fp * cost_fp) + (fn * cost_fn)
        sim_costs.append({
            "contamination": c * 100,
            "threshold": cutoff,
            "n_alerts": len(pred_outlier_ids),
            "FP": fp,
            "FN": fn,
            "total_cost": total_cost
        })

    cost_df = pd.DataFrame(sim_costs)
    opt_row = cost_df.loc[cost_df["total_cost"].idxmin()]

    st.success(
        f"🎯 **Ekonomicky optimální nastavení:** Kontaminace **{opt_row['contamination']:.1f} %** "
        f"(Práh LOF >= **{opt_row['threshold']:.3f}**), očekávaná celková ztráta **{int(opt_row['total_cost']):,} Kč**."
    )

    fig_cost = px.line(
        cost_df,
        x="contamination",
        y="total_cost",
        markers=True,
        title="Křivka celkových nákladů (Cost Curve) v závislosti na parametru kontaminace",
        labels={"contamination": "Nastavená kontaminace (%)", "total_cost": "Celkové finanční náklady (Kč)"}
    )
    fig_cost.add_vline(
        x=opt_row["contamination"],
        line_dash="dash",
        line_color="#10b981",
        annotation_text=f"Optimum: {opt_row['contamination']:.1f} %",
        annotation_position="top left"
    )
    fig_cost.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_cost, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: Volba sousedů (k) & Maskování
# -----------------------------------------------------------------------------
with tab_kparam:
    st.subheader("👥 Volba počtu sousedů k (n_neighbors) & Maskovací efekt")
    st.markdown(
        r"""
        Hodnota hyperparametru $k$ určuje měřítko, na kterém je lokální hustota měřena:
        
        ### 1. Riziko příliš malého $k$ ($k < 10$): Fenomén maskování (Outlier Masking)
        Pokud se dvě anomálie náhodou nacházejí blízko sebe (např. dvojice podvodných transakcí se stejnou částkou 
        z téže IP adresy):
        - Při $k=2$ jsou tyto dva body navzájem svými nejbližšími sousedy.
        - Jejich vzdálenost je malá, proto mají zdánlivě vysokou hustotu.
        - **Výsledek:** Model selže a obě anomálie označí za běžné body! Proto Breunig et al. (2000) doporučují $k \ge 10$ až $20$.
        
        ### 2. Riziko příliš velkého $k$ ($k > 50$): Degenerace na globální hustotu
        Pokud $k$ roste k celkové velikosti shluku:
        - $k$-okolí bodu začne zahrnovat body z jiných shluků i prázdný mezishlukový prostor.
        - LOF ztrácí schopnost adaptace na lokální proměnlivou hustotu a chová se jako triviální globální Z-skóre.
        """
    )

    st.markdown("#### Shrnutí doporučení pro volbu $k$ v závislosti na povaze dat:")
    k_rules = pd.DataFrame([
        {"Rozsah k": "k < 10", "Dopad na model": "Vysoká citlivost na šum, riziko maskování dvojic anomálií", "Doporučení": "Nedoporučeno"},
        {"Rozsah k": "10 <= k <= 20", "Dopad na model": "Optimální rovnováha mezi lokálním rozlišením a robustností vůči šumu", "Doporučení": "Standardní volba (Scikit-learn default: 20)"},
        {"Rozsah k": "20 < k <= 50", "Dopad na model": "Vyhlazení okrajů shluků, vhodné pro rozsáhlé datasety s velkými shluky", "Doporučení": "Vhodné pro N > 5 000"},
        {"Rozsah k": "k > 50", "Dopad na model": "Ztráta lokálního kontextu, nárůst výpočetní náročnosti O(N * k)", "Doporučení": "Nevhodné"},
    ])
    st.dataframe(k_rules, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# TAB 4: LOF vs. iForest vs. OC-SVM
# -----------------------------------------------------------------------------
with tab_bench:
    st.subheader("⚔️ Kdy zvolit LOF a kdy sáhnout po konkurenčních metodách?")
    st.markdown(
        """
        V machine learningu neexistuje univerzální vítěz (No Free Lunch Theorem). Zde je systematické srovnání čtyř 
        hlavních algoritmů pro detekci anomálií v Scikit-learn:
        """
    )

    bench_table = pd.DataFrame([
        {
            "Algoritmus": "Local Outlier Factor (LOF)",
            "Klíčový princip": "Poměr lokální dosažitelné hustoty bodu k hustotě k-sousedů",
            "Silná stránka": "Exceluje při proměnlivé hustotě shluků a lokálních vzorech",
            "Slabá stránka": "Výpočetní náročnost O(N²), transductivní charakter",
            "Ideální pro": "Shluková prostorová data, N < 50 000, D < 15"
        },
        {
            "Algoritmus": "Isolation Forest (iForest)",
            "Klíčový princip": "Počet náhodných řezů potřebných k izolaci bodu ve stromu",
            "Silná stránka": "Škálovatelnost O(N log N), funguje i ve vyšších dimenzích",
            "Slabá stránka": "Citlivý na lokální hustotní anomálie uvnitř shluků",
            "Ideální pro": "Vysokodimenzionální tabulková data, N > 100 000"
        },
        {
            "Algoritmus": "One-Class SVM",
            "Klíčový princip": "Nalezení optimální nadroviny oddělující data od počátku v RKHS",
            "Silná stránka": "Teoretická záruka pro nelineární hranice (RBF jádro)",
            "Slabá stránka": "Extrémně citlivý na hyperparametry (nu, gamma)",
            "Ideální pro": "Semi-supervised detekce (čistá trénovací data bez anomálií)"
        },
        {
            "Algoritmus": "Elliptic Envelope",
            "Klíčový princip": "Robustní odhad kovarianční matice (Minimum Covariance Determinant)",
            "Silná stránka": "Matematicky optimální pro jednoshluková normální data",
            "Slabá stránka": "Zcela selhává u multimodálních shluků a nelineárních struktur",
            "Ideální pro": "Jediný eliptický shluk s gaussovským rozdělením"
        }
    ])
    st.dataframe(bench_table, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# TAB 5: Produkční guardrails 2026
# -----------------------------------------------------------------------------
with tab_prod:
    st.subheader("🛡️ Závazná pravidla pro nasazení LOF v produkčním prostředí (2026)")
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(
            """
            <div style="background-color: #fef2f2; padding: 18px; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #991b1b;">1. Pozor na rozdíl: Outlier vs. Novelty Detection!</h4>
                <p style="font-size: 0.9rem; color: #7f1d1d; margin-top: 8px;">
                    Standardně je LOF <b>transductivní</b>: nemá metodu <code>predict()</code> pro nová pozorování! 
                    Pokud potřebujete model nasadit do REST API a skórovat příchozí transakce za běhu, 
                    <b>musíte explicitně nastavit <code>novelty=True</code></b> při inicializaci. Jinak zavolání 
                    <code>predict()</code> vyvolá <code>AttributeError</code>!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="background-color: #fffbeb; padding: 18px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #92400e;">2. Problém kvadratické škálovatelnosti O(N²)</h4>
                <p style="font-size: 0.9rem; color: #78350f; margin-top: 8px;">
                    Při N = 10 000 zabere výpočet milisekundy. Při N = 500 000 LOF zahltí RAM i CPU 
                    (výpočet matice vzdáleností). Pro velká data použijte aproximaci (např. <code>algorithm='ball_tree'</code>) 
                    nebo předtrénujte model na reprezentativním vzorku s <code>novelty=True</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g_col2:
        st.markdown(
            """
            <div style="background-color: #eff6ff; padding: 18px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #1e40af;">3. Prokletí dimenzionality (Curse of Dimensionality)</h4>
                <p style="font-size: 0.9rem; color: #1e3a8a; margin-top: 8px;">
                    Pokud má dataset více než 20–30 příznaků, eukleidovské vzdálenosti se stávají ekvidistantními 
                    (všechny body jsou od sebe stejně daleko). Skóre LOF zkolabuje k hodnotě 1.0. 
                    <b>Pravidlo:</b> Před LOF vždy proveďte selekci příznaků nebo redukci dimenzionality (PCA / UMAP).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="background-color: #f0fdf4; padding: 18px; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 16px;">
                <h4 style="margin: 0; color: #065f46;">4. Povinnost standardizace (Feature Scaling)</h4>
                <p style="font-size: 0.9rem; color: #064e3b; margin-top: 8px;">
                    Jelikož LOF počítá k-vzdálenost v prostoru znaků, příznak s velkým rozptylem 
                    (např. roční příjem v dolarech) zcela přehluší příznak s malým rozptylem (např. věk). 
                    Vždy zařaďte <code>StandardScaler</code> nebo <code>RobustScaler</code> jako první krok pipeline.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
