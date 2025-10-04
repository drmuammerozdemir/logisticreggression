# app.py
# ============================================================
# 🔬 Univariate & Multivariate Lojistik Regresyon (Streamlit)
# Yazar: ChatGPT (GPT-5 Thinking)
# Açıklama: CSV/XLSX/SAV yükleyin → Bağımlı değişkeni ve aday değişkenleri seçin →
#           Univariate tarama + Multivariate model (OR, %95 GA, p, AIC/BIC, HL testi, ROC AUC)
# ============================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st

# İstatistik paketleri
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# SAV için
try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False

st.set_page_config(page_title="Lojistik Regresyon Aracı", layout="wide")
st.title("🧮 Lojistik Regresyon (Univariate + Multivariate)")

# ===================== Yardımcı Fonksiyonlar ===================== #

def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    if name.endswith(".sav"):
        # Streamlit UploadedFile bir yol (path) değildir; pyreadstat path bekler.
        # BytesIO üzerinden okuyoruz (veya gerekirse temp dosyaya yazıp okuruz).
        if not HAS_PYREADSTAT:
            st.error(".sav dosyası için 'pyreadstat' gerekir. Lütfen 'pip install pyreadstat' kurun.")
            st.stop()
        try:
            file.seek(0)
            buffer = io.BytesIO(file.read())
            df, meta = pyreadstat.read_sav(buffer)
            return df
        except Exception as e:
            # Alternatif: geçici dosyaya yazıp path üzerinden dene
            import tempfile, os
            try:
                file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                df, meta = pyreadstat.read_sav(tmp_path)
                os.unlink(tmp_path)
                return df
            except Exception as e2:
                st.error(f".sav okunamadı: {e2}")
                st.stop()
    st.error("Desteklenmeyen dosya türü. CSV/XLSX/SAV yükleyin.")
    st.stop()
        df, meta = pyreadstat.read_sav(file)
        return df
    st.error("Desteklenmeyen dosya türü. CSV/XLSX/SAV yükleyin.")
    st.stop()


def is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    return len(vals) == 2


def format_or_ci(or_val, lo, hi):
    if any([pd.isna(or_val), pd.isna(lo), pd.isna(hi)]):
        return "NA"
    return f"{or_val:.3f} ({lo:.3f}–{hi:.3f})"


def hosmer_lemeshow(y_true, y_prob, g=10):
    """HL gof testi (grup=decile). Dönüş: (chi2, p, tablo)
    Kaynak: Standart HL uygulaması (O=observe, E=expect)."""
    data = pd.DataFrame({"y": y_true, "p": y_prob})
    data = data.sort_values("p").reset_index(drop=True)
    data["group"] = pd.qcut(data["p"], q=g, duplicates='drop')
    # grup bazlı özet
    tbl = data.groupby("group").agg(n=("y","size"), obs=("y","sum"), p_mean=("p","mean"))
    tbl["exp"] = tbl["n"] * tbl["p_mean"]
    tbl["chi"] = (tbl["obs"] - tbl["exp"])**2 / (tbl["exp"] * (1 - tbl["p_mean"]).clip(lower=1e-12))
    chi2 = tbl["chi"].sum()
    # serbestlik derecesi g-2
    from scipy.stats import chi2 as chi2_dist
    df_hl = max(int(tbl.shape[0] - 2), 1)
    pval = 1 - chi2_dist.cdf(chi2, df_hl)
    return chi2, pval, tbl.reset_index()


def build_formula(dv, predictors, cat_info):
    """patsy formülünü kur. cat_info: {var: reference_value or None}"""
    terms = []
    for v in predictors:
        if v in cat_info and cat_info[v] is not None:
            ref = cat_info[v]
            terms.append(f"C({v}, Treatment(reference='{ref}'))")
        else:
            terms.append(v)
    rhs = " + ".join(terms)
    return f"{dv} ~ {rhs}" if rhs else f"{dv} ~ 1"


def fit_logit(formula, data):
    model = smf.logit(formula, data=data)
    res = model.fit(disp=0, maxiter=200)
    return model, res


def extract_or_table(res):
    params = res.params
    conf = res.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    pvals = res.pvalues
    out = pd.concat([params, conf, pvals], axis=1)
    out.columns = ["coef", "ci_low", "ci_high", "p"]
    out["OR"] = np.exp(out["coef"])  # odds ratio
    out["OR_low"] = np.exp(out["ci_low"])  # CI düşük
    out["OR_high"] = np.exp(out["ci_high"])  # CI yüksek
    # sadece katsayı satırlarını döndür (sabit dahil)
    return out.reset_index().rename(columns={"index": "variable"})


def make_confusion(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return cm, acc, sens, spec


# ===================== 1) Veri Yükleme ===================== #

st.sidebar.header("1) Veri Yükle")
file = st.sidebar.file_uploader("CSV / XLSX / SAV yükleyin", type=["csv","xlsx","xls","sav"]) 

if not file:
    st.info("Başlamak için bir veri dosyası yükleyin. (CSV/XLSX/SAV)")
    st.stop()

df = read_any(file)
st.write("**Örnek İlk Satırlar:**")
st.dataframe(df.head())

# ===================== 2) Değişken Seçimi ===================== #

st.sidebar.header("2) Değişken Seçimi")
all_cols = df.columns.tolist()

# DV seçimi
dv = st.sidebar.selectbox("Bağımlı Değişken (Binary)", options=all_cols)

# Pozitif sınıf seçimi (eğer 0/1 değilse)
unique_dv_vals = df[dv].dropna().unique().tolist()
map_needed = not (set(unique_dv_vals) <= {0,1})

y = df[dv].copy()
if map_needed:
    st.sidebar.markdown("**Bağımlı değişken 0/1 değil. Pozitif sınıfı seçin:**")
    pos_label = st.sidebar.selectbox("Pozitif sınıf (1) değeri", options=sorted(unique_dv_vals, key=str))
    y_mapped = (df[dv] == pos_label).astype(int)
    df["__DV__"] = y_mapped
    dv_use = "__DV__"
    st.sidebar.caption(f"Seçilen pozitif sınıf: {pos_label} → 1, diğerleri → 0")
else:
    dv_use = dv

# Aday bağımsız değişkenler
candidate_vars = [c for c in all_cols if c != dv]
ivs = st.sidebar.multiselect("Aday Bağımsız Değişkenler", options=candidate_vars, default=candidate_vars)

if len(ivs) == 0:
    st.warning("En az bir bağımsız değişken seçin.")
    st.stop()

# Kategorik olanlar
st.sidebar.header("3) Kategorik Tanımları")
cat_vars = st.sidebar.multiselect("Kategorik değişkenler (varsa)", options=ivs, default=[c for c in ivs if df[c].dtype == 'object'])

cat_ref = {}
for c in cat_vars:
    levels = sorted([str(x) for x in pd.Series(df[c]).dropna().unique()])
    if levels:
        ref = st.sidebar.selectbox(f"Referans kategori – {c}", options=levels, index=0, key=f"ref_{c}")
        cat_ref[c] = ref
    else:
        cat_ref[c] = None

# Eksik gözlemleri ele al
use_cols = [dv_use] + ivs
work = df[use_cols].copy()
# string kategorikleri olduğu gibi bırak, sayısalları float'a çevirirken hatalara karşı yumuşak ol
for c in ivs:
    if c not in cat_vars:
        try:
            work[c] = pd.to_numeric(work[c], errors='coerce')
        except Exception:
            pass

n_before = work.shape[0]
work = work.dropna()
n_after = work.shape[0]
st.sidebar.caption(f"Eksik veri nedeniyle düşen gözlem: {n_before - n_after}")

# ===================== 3) Univariate Taraması ===================== #

st.header("🔹 Univariate Lojistik Regresyon")
st.caption("Her bir değişken tek tek modele alınır. OR, %95 GA ve p-değeri raporlanır.")

uni_rows = []
for var in ivs:
    try:
        fml = build_formula(dv_use, [var], cat_ref)
        _, res = fit_logit(fml, work)
        tab = extract_or_table(res)
        # sadece ilgili katsayı satırlarını (sabit hariç) filtrele
        rows = tab[~tab["variable"].str.contains("Intercept", case=False, na=False)].copy()
        # Çok kategorik değişkende birden fazla satır çıkabilir → p değerini global Wald testi ile almak isteyebiliriz.
        # Basit yaklaşım: en küçük p'yi göster ve "(çok düzeyli)" notu düş
        p_display = rows["p"].min() if rows.shape[0] > 0 else np.nan
        # OR & CI: tek düzeyse direkt; çok düzeyse NA yaz
        if rows.shape[0] == 1:
            OR = rows["OR"].iloc[0]
            lo = rows["OR_low"].iloc[0]
            hi = rows["OR_high"].iloc[0]
        else:
            OR, lo, hi = (np.nan, np.nan, np.nan)
        uni_rows.append({
            "değişken": var,
            "OR (95% GA)": format_or_ci(OR, lo, hi),
            "p": p_display,
            "AIC": res.aic,
            "BIC": res.bic
        })
    except Exception as e:
        uni_rows.append({"değişken": var, "OR (95% GA)": "NA", "p": np.nan, "AIC": np.nan, "BIC": np.nan})

uni_df = pd.DataFrame(uni_rows).sort_values("p", na_position='last')
st.dataframe(uni_df, use_container_width=True)

# İndirilebilir CSV
csv_uni = uni_df.to_csv(index=False).encode('utf-8')
st.download_button("Univariate sonuçlarını indir (CSV)", data=csv_uni, file_name="univariate_logit.csv", mime="text/csv")

# Otomatik eşik ve seçim
st.subheader("Univariate'e göre Multivariate aday seçimi")
p_thresh = st.slider("p-değeri eşiği", 0.001, 0.20, 0.05, 0.001)
preselect = uni_df.loc[uni_df["p"] <= p_thresh, "değişken"].tolist()
st.caption(f"Eşik altı değişkenler: {', '.join(preselect) if preselect else '(yok)'}")

manual_multi = st.multiselect("Multivariate modele girecek değişkenler (elle düzenleyin)", options=ivs, default=preselect)

# ===================== 4) Multivariate Model ===================== #

st.header("🔹 Multivariate Lojistik Regresyon")
st.caption("Univariate'de anlamlı görünenleri (veya klinik olarak önemli gördüklerinizi) birlikte modele alın.")

if manual_multi:
    fml_multi = build_formula(dv_use, manual_multi, cat_ref)
    with st.expander("Kullanılan formül (patsy)"):
        st.code(fml_multi)
    try:
        model_m, res_m = fit_logit(fml_multi, work)
        multi_tab = extract_or_table(res_m)
        multi_tab["OR (95% GA)"] = multi_tab.apply(lambda r: format_or_ci(np.exp(r["coef"]), np.exp(r["ci_low"]), np.exp(r["ci_high"])) if pd.notna(r["coef"]) else "NA", axis=1)
        # Görsel tablo
        pretty = multi_tab[["variable", "OR (95% GA)", "p"]].copy()
        st.subheader("Model Katsayıları (OR, %95 GA, p)")
        st.dataframe(pretty, use_container_width=True)

        # AIC/BIC, McFadden R2
        llf = res_m.llf
        llnull = res_m.llnull if hasattr(res_m, 'llnull') else np.nan
        pseudo_r2 = 1 - (llf / llnull) if not (np.isnan(llf) or np.isnan(llnull)) else np.nan
        col1, col2, col3 = st.columns(3)
        col1.metric("AIC", f"{res_m.aic:.2f}")
        col2.metric("BIC", f"{res_m.bic:.2f}")
        col3.metric("McFadden R²", f"{pseudo_r2:.3f}" if not pd.isna(pseudo_r2) else "NA")

        # Tahmin ve HL testi
        y_true = work[dv_use].astype(int).values
        y_prob = res_m.predict()
        chi_hl, p_hl, tbl_hl = hosmer_lemeshow(y_true, y_prob, g=10)
        st.write(f"**Hosmer–Lemeshow**: χ² = {chi_hl:.3f}, p = {p_hl:.3f}")
        with st.expander("HL Gruplama Tablosu"):
            st.dataframe(tbl_hl)

        # ROC & Confusion Matrix (0.5 eşik)
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        st.write(f"**ROC AUC**: {auc:.3f}")
        # Çizim
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Eğrisi')
        st.pyplot(fig, use_container_width=True)

        cm, acc, sens, spec = make_confusion(y_true, y_prob, threshold=0.5)
        st.write("**Sınıflandırma (eşik=0.5)**")
        st.write(pd.DataFrame(cm, index=["Gerçek 0","Gerçek 1"], columns=["Tahmin 0","Tahmin 1"]))
        col1, col2, col3 = st.columns(3)
        col1.metric("Doğruluk", f"{acc:.3f}")
        col2.metric("Duyarlılık (Sens)", f"{sens:.3f}" if not pd.isna(sens) else "NA")
        col3.metric("Seçicilik (Spec)", f"{spec:.3f}" if not pd.isna(spec) else "NA")

        # İndirme: multivariate tablo
        csv_multi = pretty.to_csv(index=False).encode('utf-8')
        st.download_button("Multivariate sonuçlarını indir (CSV)", data=csv_multi, file_name="multivariate_logit.csv", mime="text/csv")

        # Tam özet (isteğe bağlı)
        with st.expander("Statsmodels özet (ayrıntı)"):
            st.text(res_m.summary2().as_text())

    except Exception as e:
        st.error(f"Model kurulumunda hata: {e}")
else:
    st.info("Multivariate için en az bir değişken seçin.")

# ===================== 5) Raporlama Notları ===================== #

st.markdown(
    """
    ---
    **Raporlama İpuçları (SPSS uyumu):**
    - Univariate: her değişken için OR, %95 GA ve p-değeri.
    - Multivariate: birlikte modele girenler için düzeltilmiş (adjusted) OR, %95 GA ve p.
    - Uyum: Hosmer–Lemeshow, AIC/BIC, McFadden R²; ayrıca ROC AUC verin.
    - Kategorik değişkenlerde referans kategoriyi yazmayı unutmayın.
    - Düşük frekanslı hücreler veya tam ayrışma (perfect separation) hatalarında kategori birleştirmeyi düşünün.
    """
)
