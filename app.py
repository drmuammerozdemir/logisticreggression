# ============================================================
# 🔬 Binary (Logistic) + Continuous (Linear) Regression Toolkit
# Univariate tarama + Multivariate model + (Binary) ROC–AUC, Youden cut-off, DeLong testi, Firth lojistik
# ============================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from delong import delong_ci, delong_roc_test
from scipy.stats import norm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# Görselleştirme
import matplotlib.pyplot as plt

# SAV için
try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False

st.set_page_config(page_title="Regression + ROC Toolkit", layout="wide")
st.title("🧮 Regression Toolkit: Logistic (ROC) + Linear")

# ===================== Yardımcılar ===================== #

def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    if name.endswith(".sav"):
        if not HAS_PYREADSTAT:
            st.error(".sav dosyası için 'pyreadstat' gerekir. Lütfen 'pip install pyreadstat' kurun.")
            st.stop()
        try:
            file.seek(0)
            buffer = io.BytesIO(file.read())
            df, meta = pyreadstat.read_sav(buffer)
            return df
        except Exception:
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

def format_or_ci(or_val, lo, hi):
    if any([pd.isna(or_val), pd.isna(lo), pd.isna(hi)]):
        return "NA"
    return f"{or_val:.3f} ({lo:.3f}–{hi:.3f})"

def build_formula(dv, predictors, cat_info):
    terms = []
    for v in predictors:
        if v in cat_info and cat_info[v] is not None:
            ref = cat_info[v]
            terms.append(f"C({v}, Treatment(reference='{ref}'))")
        else:
            terms.append(v)
    rhs = " + ".join(terms) if predictors else "1"
    return f"{dv} ~ {rhs}"

def fit_logit(formula, data):
    model = smf.logit(formula, data=data)
    res = model.fit(disp=0, maxiter=200)
    return model, res

def fit_ols(formula, data):
    model = smf.ols(formula, data=data)
    res = model.fit()
    return model, res

def extract_or_table(res):
    params = res.params
    conf = res.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    pvals = res.pvalues
    out = pd.concat([params, conf, pvals], axis=1)
    out.columns = ["coef", "ci_low", "ci_high", "p"]
    out["OR"] = np.exp(out["coef"])
    out["OR_low"] = np.exp(out["ci_low"])
    out["OR_high"] = np.exp(out["ci_high"])
    return out.reset_index().rename(columns={"index": "variable"})

def extract_rrr_table(res, col_idx, class_name):
    # Multinomial sonuçlarını (RRR) çeken yardımcı fonksiyon
    params = res.params.iloc[:, col_idx]
    pvals = res.pvalues.iloc[:, col_idx]
    # Standart hatalar üzerinden basit %95 GA hesabı
    bse = res.bse.iloc[:, col_idx]
    lower = params - 1.96 * bse
    upper = params + 1.96 * bse
    
    df_out = pd.DataFrame({
        "variable": params.index,
        "coef": params.values,
        "ci_low": lower.values,
        "ci_high": upper.values,
        "p": pvals.values
    })
    df_out["RRR"] = np.exp(df_out["coef"])
    df_out["RRR_low"] = np.exp(df_out["ci_low"])
    df_out["RRR_high"] = np.exp(df_out["ci_high"])
    df_out["Class"] = class_name
    return df_out

def make_confusion(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prev = np.mean(y_true)
    ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev)) if not (np.isnan(sens) or np.isnan(spec)) else np.nan
    npv = (spec*(1-prev)) / ((1-sens)*prev + spec*(1-prev)) if not (np.isnan(sens) or np.isnan(spec)) else np.nan
    lr_pos = sens / (1-spec) if (1-spec) > 0 else np.nan
    lr_neg = (1-sens) / spec if spec > 0 else np.nan
    return cm, acc, sens, spec, ppv, npv, lr_pos, lr_neg

def compute_youden(fpr, tpr, thr):
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "threshold": float(thr[idx]),
        "J": float(j[idx]),
        "sens": float(tpr[idx]),
        "spec": float(1 - fpr[idx]),
        "index": idx
    }

def _wilson_ci(success, total, alpha=0.05):
    """İkili orana Wilson (%95 GA) — success/total için (lo, hi) döner."""
    if total == 0:
        return (np.nan, np.nan)
    p = success / total
    z = norm.ppf(1 - alpha/2)
    den = 1 + z**2/total
    center = (p + z**2/(2*total)) / den
    half = (z*np.sqrt(p*(1-p)/total + z**2/(4*total**2))) / den
    return max(0.0, center - half), min(1.0, center + half)

# ---- Unstable kontrolü (uçuk/sonsuz CI vs.) ----
def _is_unstable(orv, lo, hi, fold_limit=1e6):
    if not (np.isfinite(orv) and np.isfinite(lo) and np.isfinite(hi)):
        return True
    if lo <= 0 or hi <= 0:
        return True
    if (hi / lo) > fold_limit:
        return True
    return False

# ---- Firth helpers ----
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def firth_logit(X, y, names=None, maxiter=200, tol=1e-8):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n, p = X.shape
    beta = np.zeros(p)

    converged = False
    for _ in range(maxiter):
        eta = X @ beta
        p_i = _sigmoid(eta)
        W = p_i * (1.0 - p_i)
        XWX = X.T @ (W[:, None] * X)
        try:
            XWX_inv = np.linalg.inv(XWX)
        except np.linalg.LinAlgError:
            XWX_inv = np.linalg.pinv(XWX)

        S = (X * W[:, None]) @ XWX_inv
        h = np.sum(S * X, axis=1)

        a = (0.5 - h) * (1.0 - 2.0 * p_i)
        Ustar = X.T @ (y - p_i + a)

        step = XWX_inv @ Ustar
        beta_new = beta + step

        if np.max(np.abs(step)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    eta = X @ beta
    p_i = _sigmoid(eta)
    W = p_i * (1.0 - p_i)
    XWX = X.T @ (W[:, None] * X)
    try:
        cov = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XWX)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    zcrit = norm.ppf(0.975)
    ci_low = beta - zcrit * se
    ci_high = beta + zcrit * se

    return {
        "coef": beta, "se": se, "z": z, "p": pvals,
        "ci_low": ci_low, "ci_high": ci_high,
        "names": names if names is not None else [f"x{i}" for i in range(len(beta))],
        "converged": converged, "y_prob": p_i
    }

def firth_by_formula(formula, data):
    model = smf.logit(formula, data=data)
    X = model.exog
    y = model.endog
    names = model.exog_names
    return firth_logit(X, y, names=names)

def hosmer_lemeshow(y_true, y_prob, g=10):
    data = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p").reset_index(drop=True)
    data["group"] = pd.qcut(data["p"], q=g, duplicates='drop')
    tbl = data.groupby("group").agg(n=("y", "size"), obs=("y", "sum"), p_mean=("p", "mean"))
    tbl["exp"] = tbl["n"] * tbl["p_mean"]
    tbl["chi"] = (tbl["obs"] - tbl["exp"])**2 / (tbl["exp"] * (1 - tbl["p_mean"]).clip(lower=1e-12))
    chi2 = tbl["chi"].sum()
    from scipy.stats import chi2 as chi2_dist
    df_hl = max(int(tbl.shape[0] - 2), 1)
    pval = 1 - chi2_dist.cdf(chi2, df_hl)
    return chi2, pval, tbl.reset_index()

def _clean_term_for_forest(t):
    # C(var)[T.level] → var: level
    if isinstance(t, str) and t.startswith("C(") and ")[T." in t:
        base = t.split("C(")[1].split(")")[0]
        lev = t.split("[T.")[1].rstrip("]")
        return f"{base}: {lev}"
    return t

def make_forest_plot_linear_clip(df_or, title="Forest Plot (OR, 95% CI)", xmin=None, xmax=None):
    """
    df_or: ['label','OR','OR_low','OR_high','p']  (p opsiyonel)
    - Lineer x-ekseni
    - CI eksen dışına taşıyorsa ok başı ile işaretler
    - xmin/xmax None ise robust (5–95 pct) limit + %20 pay ve OR=1'i kapsayacak şekilde ayarlanır
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df_or.replace([np.inf, -np.inf], np.nan).dropna(subset=["OR","OR_low","OR_high"]).copy()
    if df.empty:
        return None, (np.nan, np.nan), []

    # Robust varsayılan limitler
    if xmin is None or xmax is None:
        vals = np.r_[df["OR_low"].values, df["OR_high"].values]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            xmin, xmax = 0.5, 2.0
        else:
            lo = np.nanpercentile(vals, 5)
            hi = np.nanpercentile(vals, 95)
            pad_lo, pad_hi = 0.8, 1.2
            xmin = lo*pad_lo if xmin is None else xmin
            xmax = hi*pad_hi if xmax is None else xmax
            xmin = max(xmin, 0.01)
            # OR=1 mutlaka görünür olsun
            xmin = min(xmin, 0.9)
            xmax = max(xmax, 1.1)

    df = df.sort_values("OR", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.6*len(df)+1)))

    truncated = []  # hangi değişken kesildi listesi

    for i, r in df.iterrows():
        L, M, H = float(r["OR_low"]), float(r["OR"]), float(r["OR_high"])
        # clip
        left_clip  = L < xmin
        right_clip = H > xmax
        Lc = max(L, xmin)
        Hc = min(H, xmax)

        # CI çizgisi
        ax.plot([Lc, Hc], [y[i], y[i]], color="black", lw=2.5)
        # orta nokta (kare)
        if xmin <= M <= xmax:
            ax.scatter(M, y[i], color="black", s=55, marker="s", zorder=3)
        # ok başları
        if left_clip:
            ax.scatter(xmin, y[i], marker="<", s=70, color="black", zorder=3)
            truncated.append((r["label"], "left"))
        if right_clip:
            ax.scatter(xmax, y[i], marker=">", s=70, color="black", zorder=3)
            truncated.append((r["label"], "right"))

    # OR=1 kalın dikey çizgi
    ax.axvline(1.0, color="black", lw=2)

    # Y ekseni
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Odds Ratio", fontsize=11)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlim(xmin, xmax)
    ax.grid(False)

    # Sağ metin sütunu (OR (95% CI)  p)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    texts = []
    for _, r in df.iterrows():
        ci_txt = f"{r['OR']:.3f} ({r['OR_low']:.3f}-{r['OR_high']:.3f})"
        if "p" in r and pd.notna(r["p"]):
            ptxt = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            ci_txt = f"{ci_txt}    {ptxt}"
        texts.append(ci_txt)
    ax2.set_yticklabels(texts, fontsize=11)
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylabel("")

    plt.tight_layout()
    return fig, (xmin, xmax), truncated

# ===================== 1) Veri Yükleme ===================== #

st.sidebar.header("1) Veri Yükle")
file = st.sidebar.file_uploader("CSV / XLSX / SAV yükleyin", type=["csv", "xlsx", "xls", "sav"])
if not file:
    st.info("Başlamak için bir veri dosyası yükleyin.")
    st.stop()
df = read_any(file)
st.write("**Örnek İlk Satırlar:**")
st.dataframe(df.head())

# ===================== 2) Model Tipi ve Değişken Seçimi ===================== #

st.sidebar.header("2) Model Tipi")
mode = st.sidebar.selectbox("Seçin", [
    "Binary (Logistic)", 
    "Continuous (Linear)", 
    "Multinomial (Logistic)", 
    "Penalized (Lasso/Ridge)"
])

all_cols = df.columns.tolist()

if mode == "Binary (Logistic)":
    # DV seçimi
    dv = st.sidebar.selectbox("Bağımlı Değişken (Binary)", options=all_cols)
    unique_vals = df[dv].dropna().unique().tolist()
    map_needed = not (set(unique_vals) <= {0, 1})
    if map_needed:
        st.sidebar.markdown("**Pozitif sınıfı seçin (1):**")
        pos_label = st.sidebar.selectbox("Pozitif sınıf (1)", options=sorted(unique_vals, key=str))
        df["__DV__"] = (df[dv] == pos_label).astype(int)
        dv_use = "__DV__"
        st.sidebar.caption(f"Seçilen pozitif: {pos_label} → 1, diğerleri → 0")
    else:
        dv_use = dv

    candidates = [c for c in all_cols if c != dv]
    ivs = st.sidebar.multiselect("Aday Bağımsız Değişkenler", options=candidates, default=candidates)
    if not ivs:
        st.warning("En az bir bağımsız değişken seçin.")
        st.stop()

    # Kategorikler
    st.sidebar.header("3) Kategorik Tanımları")
    cat_vars = st.sidebar.multiselect(
        "Kategorik değişkenler (varsa)",
        options=ivs,
        default=[c for c in ivs if df[c].dtype == 'object']
    )
    cat_ref = {}
    for c in cat_vars:
        levels = sorted([str(x) for x in pd.Series(df[c]).dropna().unique()])
        ref = st.sidebar.selectbox(f"Referans – {c}", options=levels, index=0, key=f"ref_{c}")
        cat_ref[c] = ref

    use_cols = [dv_use] + ivs
    work = df[use_cols].copy()
    # numerik dönüştür
    for c in ivs:
        if c not in cat_vars:
            work[c] = pd.to_numeric(work[c], errors='coerce')
    n_before = work.shape[0]
    work = work.dropna()
    n_after = work.shape[0]
    st.sidebar.caption(f"Eksik nedeniyle düşen gözlem: {n_before - n_after}")

        # ---------- Univariate ---------- #
    st.header("🔹 Univariate Logistic Regression")
    uni_rows = []
    for var in ivs:
        try:
            fml = build_formula(dv_use, [var], cat_info=cat_ref)
            _, res = fit_logit(fml, work)

            tab = extract_or_table(res)
            rows = tab[~tab["variable"].str.contains("Intercept", case=False, na=False)].copy()
            p_display = rows["p"].min() if rows.shape[0] > 0 else np.nan

            if rows.shape[0] == 1:
                # Tek katsayı → OR/CI hesapla
                OR = rows["OR"].iloc[0]
                lo = rows["OR_low"].iloc[0]
                hi = rows["OR_high"].iloc[0]

                # ===== MINI FIRTH FALLBACK (yalnızca patlayan/sonsuz/çok geniş CI'da) =====
                if _is_unstable(OR, lo, hi):
                    try:
                        fr_uni = firth_by_formula(fml, work)
                        mask = [not str(nm).lower().startswith("intercept") for nm in fr_uni["names"]]
                        b, lo_b, hi_b = fr_uni["coef"][mask][0], fr_uni["ci_low"][mask][0], fr_uni["ci_high"][mask][0]
                        or_str = f"{np.exp(b):.3f} ({np.exp(lo_b):.3f}–{np.exp(hi_b):.3f}) [Firth]"
                    except Exception:
                        # Firth de başarısızsa "NE" yaz
                        or_str = "NE (unstable/separation)"
                else:
                    # Stabil ise klasik Wald OR(CI)
                    or_str = f"{OR:.3f} ({lo:.3f}–{hi:.3f})"
            else:
                # Çok düzeyli kategorik: tek OR yok → "NA"
                or_str = "NA"

            uni_rows.append({
                "değişken": var,
                "OR (95% GA)": or_str,
                "p": p_display,
                "AIC": res.aic,
                "BIC": res.bic
            })

        except Exception:
            uni_rows.append({"değişken": var, "OR (95% GA)": "NA", "p": np.nan, "AIC": np.nan, "BIC": np.nan})

    uni_df = pd.DataFrame(uni_rows).sort_values("p", na_position='last')
    st.dataframe(uni_df, use_container_width=True)
    st.download_button(
        "Univariate (CSV)",
        uni_df.to_csv(index=False).encode("utf-8"),
        file_name="univariate_logit.csv",
        mime="text/csv"
    )

    # Aday seçimi
    st.subheader("Univariate'e göre Multivariate aday seçimi")
    p_thresh = st.slider("p-eşiği", 0.001, 0.20, 0.05, 0.001)
    preselect = uni_df.loc[uni_df["p"] <= p_thresh, "değişken"].tolist()
    st.caption(f"Eşik altı: {', '.join(preselect) if preselect else '(yok)'}")
    manual_multi = st.multiselect("Multivariate değişkenleri", options=ivs, default=preselect)

    # ---------- Multivariate + ROC ---------- #
    st.header("🔹 Multivariate Logistic + ROC")
    if manual_multi:
        fml_multi = build_formula(dv_use, manual_multi, cat_ref)
        with st.expander("Kullanılan formül"):
            st.code(fml_multi)
        try:
            model_m, res_m = fit_logit(fml_multi, work)

            # Katsayılar
            multi_tab = extract_or_table(res_m)
            multi_tab["OR (95% GA)"] = multi_tab.apply(
                lambda r: format_or_ci(np.exp(r["coef"]), np.exp(r["ci_low"]), np.exp(r["ci_high"]))
                if pd.notna(r["coef"]) else "NA", axis=1
            )
            pretty = multi_tab[["variable", "OR (95% GA)", "p"]].copy()
            st.subheader("Model Katsayıları")
            st.dataframe(pretty, use_container_width=True)
            # === Forest Plot (Multivariate OR, 95% CI, hazırlık) ===
            forest_df = multi_tab.copy()
            forest_df = forest_df[~forest_df["variable"].str.contains("Intercept", case=False, na=False)].copy()

            def _clean_term_for_forest(t):
                if isinstance(t, str) and t.startswith("C(") and ")[T." in t:
                    base = t.split("C(")[1].split(")")[0]
                    lev = t.split("[T.")[1].rstrip("]")
                    return f"{base}: {lev}"
                return t

            forest_df["label"] = forest_df["variable"].map(_clean_term_for_forest)
            forest_df = forest_df.assign(
                OR=forest_df["OR"].astype(float),
                OR_low=forest_df["OR_low"].astype(float),
                OR_high=forest_df["OR_high"].astype(float),
                p=multi_tab.set_index("variable")["p"].reindex(forest_df["variable"]).values
            )[["label","OR","OR_low","OR_high","p"]]

            # forest_df: ["label","OR","OR_low","OR_high","p"]

            # (opsiyonel) kullanıcıya eksen aralığı verelim:
            # robust default'ları önce bir önizlemeden al
            fig_tmp, (def_lo, def_hi), _ = make_forest_plot_linear_clip(forest_df, title="")
            # slider: 0.01–100 arası esnek; default robust
            lo_hi = st.slider("Forest x-ekseni (OR aralığı)", 0.01, 100.0,
                              (float(max(0.01, def_lo)), float(min(100.0, def_hi))), step=0.01)

            fig_forest, (xmin, xmax), truncated = make_forest_plot_linear_clip(
                forest_df, title="Multivariate OR (95% CI)", xmin=lo_hi[0], xmax=lo_hi[1]
            )
            if fig_forest is not None:
                st.pyplot(fig_forest, use_container_width=True)
                if truncated:
                    # kesilenleri kullanıcıya söyle
                    lefts  = [lab for lab, side in truncated if side == "left"]
                    rights = [lab for lab, side in truncated if side == "right"]
                    note = []
                    if lefts:  note.append("sol ucu kesilen: " + ", ".join(lefts))
                    if rights: note.append("sağ ucu kesilen: " + ", ".join(rights))
                    st.caption("Not: CI’ları eksen dışında kalan değişkenler ok başı ile gösterildi (" + "; ".join(note) + ").")
            else:
                st.info("Forest plot için uygun (sonlu) OR ve güven aralığı bulunamadı.")

            # AIC/BIC, McFadden R²
            llf = res_m.llf
            llnull = res_m.llnull if hasattr(res_m, 'llnull') else np.nan
            pseudo_r2 = 1 - (llf / llnull) if not (np.isnan(llf) or np.isnan(llnull)) else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric("AIC", f"{res_m.aic:.2f}")
            c2.metric("BIC", f"{res_m.bic:.2f}")
            c3.metric("McFadden R²", f"{pseudo_r2:.3f}" if not pd.isna(pseudo_r2) else "NA")

            # Tahminler
            y_true = work[dv_use].astype(int).values
            y_prob = res_m.predict()

            # HL
            chi_hl, p_hl, tbl_hl = hosmer_lemeshow(y_true, y_prob, g=10)
            st.write(f"**Hosmer–Lemeshow**: χ² = {chi_hl:.3f}, p = {p_hl:.3f}")
            with st.expander("HL Grup Tablosu"):
                st.dataframe(tbl_hl)

            # ===== VIF (multicollinearity) =====
            try:
                exog = res_m.model.exog
                names = res_m.model.exog_names
                vif_rows = []
                for i, nm in enumerate(names):
                    if nm.lower().startswith("intercept"):
                        continue
                    try:
                        vif_val = variance_inflation_factor(exog, i)
                        vif_rows.append({"Variable": nm, "VIF": float(vif_val), "Tolerance": float(1.0/vif_val)})
                    except Exception:
                        vif_rows.append({"Variable": nm, "VIF": np.nan, "Tolerance": np.nan})
                vif_df = pd.DataFrame(vif_rows).sort_values("VIF", na_position="last")
                def _style_vif(v):
                    try:
                        return "font-weight:bold;" if float(v) > 5 else ""
                    except:
                        return ""
                st.subheader("📎 Kolinearite Kontrolü (VIF)")
                st.dataframe(vif_df.style.applymap(_style_vif, subset=["VIF"]), use_container_width=True)
                st.caption("Not: VIF>5 (bazı kılavuzlarda >10) yüksek kolinearite olarak yorumlanır.")
                st.download_button("VIF (CSV)", vif_df.to_csv(index=False).encode("utf-8"),
                                   file_name="vif_model1.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"VIF hesaplanamadı: {e}")

            # ROC (tek kere)
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            st.write(f"**ROC AUC**: {auc:.3f}")
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('1 - Specificity (FPR)')
            plt.ylabel('Sensitivity (TPR)')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            st.pyplot(fig, use_container_width=True)

            # === ROC Altı Özet Tablo (AUC CI, p; Youden cut-off; Sens/Spec/PPV/NPV + %95 GA) ===
            # AUC için DeLong CI ve p (H0: AUC=0.5)
            auc_m, lo_auc, hi_auc, se_auc = delong_ci(y_true, y_prob)
            z_auc = (auc_m - 0.5) / se_auc if se_auc > 0 else np.nan
            p_auc = 2 * (1 - norm.cdf(abs(z_auc))) if np.isfinite(z_auc) else np.nan

            # Youden’a göre en iyi kesim ve o kesimde karışıklık matrisi
            best = compute_youden(fpr, tpr, thr)
            cut_for_table = float(np.clip(best["threshold"], 0.0, 1.0))
            y_pred_cut = (y_prob >= cut_for_table).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_cut, labels=[0,1]).ravel()

            # Oranlar ve Wilson %95 GA
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            ppv  = tp / (tp + fp) if (tp + fp) else np.nan
            npv  = tn / (tn + fn) if (tn + fn) else np.nan

            sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
            spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
            ppv_lo,  ppv_hi  = _wilson_ci(tp, tp + fp) if (tp + fp) else (np.nan, np.nan)
            npv_lo,  npv_hi  = _wilson_ci(tn, tn + fn) if (tn + fn) else (np.nan, np.nan)

            # Görsel tablo (yüzdeleri resimdeki gibi tam sayıya yuvarlayarak)
            summary_rows = [
                ("AUC (95% CI)",          f"{auc_m:.3f} ({lo_auc:.3f}–{hi_auc:.3f})"),
                ("p-Value",               "<0.001" if (pd.notna(p_auc) and p_auc < 0.001) else f"{p_auc:.4f}"),
                ("Cut-off",               f"≥ {cut_for_table:.2g}"),
                ("Sensitivity (95% CI)",  f"{sens*100:.0f} ({sens_lo*100:.1f}–{sens_hi*100:.1f})"),
                ("Specificity (95% CI)",  f"{spec*100:.0f} ({spec_lo*100:.1f}–{spec_hi*100:.1f})"),
                ("PPV (95% CI)",          f"{ppv*100:.0f} ({ppv_lo*100:.1f}–{ppv_hi*100:.1f})" if pd.notna(ppv) else "NA"),
                ("NPV (95% CI)",          f"{npv*100:.0f} ({npv_lo*100:.1f}–{npv_hi*100:.1f})" if pd.notna(npv) else "NA"),
            ]
            roc_summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
            st.dataframe(roc_summary_df, use_container_width=True)


            # ===== Firth bias-reduced logistic (opsiyonel) =====
            st.subheader("🛡️ Firth (bias-reduced) Lojistik – Ayrışmaya dayanıklı")
            use_firth = st.checkbox("Model 1'i Firth ile yeniden tahmin et", value=False)
            if use_firth:
                try:
                    fr = firth_by_formula(fml_multi, work)
                    fnames = fr["names"]

                    rows_f = []
                    for nm, b, lo_b, hi_b, p_b in zip(
                        fnames, fr["coef"], fr["ci_low"], fr["ci_high"], fr["p"]
                    ):
                        if str(nm).lower().startswith("intercept"):
                            continue
                        ORb, Lb, Hb = np.exp(b), np.exp(lo_b), np.exp(hi_b)
                        rows_f.append({
                            "variable": nm,
                            "OR (95% GA)": f"{ORb:.3f} ({Lb:.3f}–{Hb:.3f})",
                            "p": "<0.001" if p_b < 0.001 else f"{p_b:.3f}"
                        })
                    firth_df = pd.DataFrame(rows_f)
                    st.dataframe(firth_df, use_container_width=True)
                    st.caption("Not: Firth, separation durumlarında stabil OR/GA üretir (Jeffreys prior).")

                    # ROC kıyası
                    fpr_f, tpr_f, thr_f = roc_curve(y_true, fr["y_prob"])
                    auc_f = roc_auc_score(y_true, fr["y_prob"])
                    fig_f = plt.figure()
                    plt.plot(fpr, tpr, label=f"Klasik Model 1 (AUC={auc:.3f})")
                    plt.plot(fpr_f, tpr_f, label=f"Firth Model 1 (AUC={auc_f:.3f})")
                    plt.plot([0, 1], [0, 1], '--')
                    plt.xlabel('1 - Specificity (FPR)'); plt.ylabel('Sensitivity (TPR)')
                    plt.title('ROC: Klasik vs Firth')
                    plt.legend(loc="lower right")
                    st.pyplot(fig_f, use_container_width=True)

                except Exception as e:
                    st.error(f"Firth hesaplanamadı: {e}")

            # ==== DeLong Karşılaştırma (Model vs Tek Belirteç / İki Skor) ====
            st.subheader("ROC Karşılaştırma (DeLong)")
            num_cols = [c for c in work.columns if c != dv_use and np.issubdtype(work[c].dtype, np.number)]
            compare_var = st.selectbox(
                "Karşılaştırılacak ikinci skor/değişken (örn. TRP, 3OHKYN):",
                options=num_cols
            )
            if compare_var:
                try:
                    y2 = pd.to_numeric(work[compare_var], errors="coerce").values
                    mask = ~np.isnan(y2)
                    y_true_cmp = y_true[mask]
                    y_prob_cmp = y_prob[mask]
                    y2_cmp = y2[mask]

                    auc_m, lo_m, hi_m, se_m = delong_ci(y_true_cmp, y_prob_cmp)
                    auc_x, lo_x, hi_x, se_x = delong_ci(y_true_cmp, y2_cmp)
                    res = delong_roc_test(y_true_cmp, y_prob_cmp, y2_cmp)

                    st.write(f"**Model (Predicted Probability)**: AUC={auc_m:.3f} (95% GA {lo_m:.3f}–{hi_m:.3f})")
                    st.write(f"**{compare_var}**: AUC={auc_x:.3f} (95% GA {lo_x:.3f}–{hi_x:.3f})")
                    st.success(f"**DeLong testi**: z={res['z']:.3f}, p={res['p']:.4f} → (H₀: AUC'ler eşit)")
                except Exception as e:
                    st.error(f"DeLong hesaplanamadı: {e}")

            # ROC Koordinatları CSV
            roc_df = pd.DataFrame({
                "threshold": thr,
                "sensitivity": tpr,
                "fpr": fpr,
                "specificity": 1 - fpr,
                "youden_J": tpr - fpr
            })
            st.download_button(
                "ROC Koordinatları (CSV)",
                roc_df.to_csv(index=False).encode("utf-8"),
                file_name="roc_coords.csv",
                mime="text/csv"
            )

            # Youden cut-off
            best = compute_youden(fpr, tpr, thr)
            st.success(
                f"**Youden en iyi cut-off** = {best['threshold']:.4f} | "
                f"Sens={best['sens']:.3f}, Spec={best['spec']:.3f}, J={best['J']:.3f}"
            )

            # Cut-off değerlendirme
            st.subheader("Cut-off Değerlendirme")
            cut_default = float(np.clip(best["threshold"], 0.0, 1.0))
            cut = st.slider("Cut-off (Youden varsayılan)", 0.0, 1.0, cut_default, 0.01)
            cm, acc, sens, spec, ppv, npv, lrpos, lrneg = make_confusion(y_true, y_prob, threshold=cut)
            st.write(pd.DataFrame(cm, index=["Gerçek 0", "Gerçek 1"], columns=["Tahmin 0", "Tahmin 1"]))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Sensitivity", f"{sens:.3f}")
            c3.metric("Specificity", f"{spec:.3f}")
            c4.metric("PPV / NPV", f"{ppv:.3f} / {npv:.3f}")
            c5, c6 = st.columns(2)
            c5.metric("LR+", f"{lrpos:.3f}" if not pd.isna(lrpos) else "NA")
            c6.metric("LR−", f"{lrneg:.3f}" if not pd.isna(lrneg) else "NA")

            # İndir
            st.download_button(
                "Multivariate (CSV)",
                pretty.to_csv(index=False).encode("utf-8"),
                file_name="multivariate_logit.csv",
                mime="text/csv"
            )

            # ================== Yayın Formatı Özet Tablo (Uni + Multi) ==================
            st.subheader("📋 Özet Tablo (Univariate + Multivariate)")

            def fmt_p(p):
                if pd.isna(p):
                    return ""
                try:
                    return "<0.001" if p < 0.001 else f"{p:.3f}"
                except Exception:
                    return str(p)

            multi_raw = res_m.summary2().tables[1].reset_index().rename(columns={"index": "term"})
            ci_m = res_m.conf_int()
            ci_m.columns = ["ci_low", "ci_high"]
            multi_raw = multi_raw.merge(ci_m, left_on="term", right_index=True, how="left")
            multi_raw["OR"] = np.exp(multi_raw["Coef."])
            multi_raw["OR_low"] = np.exp(multi_raw["ci_low"])
            multi_raw["OR_high"] = np.exp(multi_raw["ci_high"])
            multi_raw = multi_raw[~multi_raw["term"].str.contains("Intercept", case=False, na=False)].copy()

            def clean_term(t):
                if t.startswith("C(") and ")[T." in t:
                    base = t.split("C(")[1].split(")")[0]
                    lev = t.split("[T.")[1].rstrip("]")
                    return f"{base}: {lev}"
                return t

            multi_raw["clean"] = multi_raw["term"].map(clean_term)
            multi_map_or = dict(zip(
                multi_raw["clean"],
                [f"{o:.3f} ({lo:.3f}–{hi:.3f})" for o, lo, hi in zip(multi_raw["OR"], multi_raw["OR_low"], multi_raw["OR_high"])]
            ))
            multi_map_p = dict(zip(multi_raw["clean"], multi_raw["P>|z|"]))

            # Univariate özetinden çek
            uni_tmp = uni_df[["değişken", "OR (95% GA)", "p"]].copy()
            uni_tmp.rename(columns={"değişken": "Variable",
                                    "OR (95% GA)": "Univariate OR (95% CI)",
                                    "p": "Univariate P"}, inplace=True)

            st.caption("İsteğe bağlı: bir değişkeni ölçekleyerek (örn. SII/100) ek satır oluştur.")
            add_scaled = st.checkbox("Ölçekli satır ekle (örn. SII/100)", value=False)
            if add_scaled:
                scale_var = st.selectbox("Ölçeklenecek değişken", options=uni_tmp["Variable"].tolist())
                scale_val = st.number_input("Ölçek katsayısı (örn. 100 → SII/100)", min_value=1.0, value=100.0, step=1.0)
                r = uni_tmp.loc[uni_tmp["Variable"] == scale_var].copy()
                if not r.empty:
                    r = r.assign(Variable=scale_var + f" / {int(scale_val)}")
                    uni_tmp = pd.concat([uni_tmp, r], ignore_index=True)

            def take_multi_or(name):
                return multi_map_or.get(name, "/")

            def take_multi_p(name):
                p = multi_map_p.get(name, np.nan)
                return fmt_p(p) if not pd.isna(p) else "/"

            out = uni_tmp.copy()
            out["Multivariate OR (95% CI)"] = out["Variable"].map(take_multi_or)
            out["Multivariate P"] = out["Variable"].map(take_multi_p)
            out["Univariate P"] = out["Univariate P"].apply(fmt_p)

            out = out[["Variable",
                       "Univariate OR (95% CI)", "Univariate P",
                       "Multivariate OR (95% CI)", "Multivariate P"]]

            def highlight_sig(s):
                try:
                    val = s.replace("<", "")
                    return float(val) < 0.05
                except Exception:
                    return False

            styler = out.style.format(na_rep="").applymap(
                lambda v: "font-weight:bold;" if isinstance(v, str) and highlight_sig(v) else "",
                subset=["Univariate P", "Multivariate P"]
            )
            st.dataframe(styler, use_container_width=True)

            st.download_button(
                "Özet Tablo (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="summary_uni_multi.csv",
                mime="text/csv"
            )
            # Excel (xlsxwriter gerekli)
            import io as _io
            buf = _io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                out.to_excel(writer, sheet_name="Summary", index=False)
            st.download_button(
                "Özet Tablo (XLSX)",
                data=buf.getvalue(),
                file_name="summary_uni_multi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            with st.expander("Statsmodels özet"):
                st.text(res_m.summary2().as_text())

        except Exception as e:
            st.error(f"Model kurulumunda hata: {e}")
    else:
        st.info("Multivariate için en az bir değişken seçin.")

elif mode == "Continuous (Linear)":
    # --------- LINEAR (CONTINUOUS) --------- #
    dv = st.sidebar.selectbox("Bağımlı Değişken (Continuous)", options=all_cols)
    candidates = [c for c in all_cols if c != dv]
    ivs = st.sidebar.multiselect("Aday Bağımsız Değişkenler", options=candidates, default=candidates)
    if not ivs:
        st.warning("En az bir bağımsız değişken seçin.")
        st.stop()

    # Kategorikler
    st.sidebar.header("3) Kategorik Tanımları")
    cat_vars = st.sidebar.multiselect(
        "Kategorik (varsa)",
        options=ivs,
        default=[c for c in ivs if df[c].dtype == 'object']
    )
    cat_ref = {}
    for c in cat_vars:
        levels = sorted([str(x) for x in pd.Series(df[c]).dropna().unique()])
        ref = st.sidebar.selectbox(f"Referans – {c}", options=levels, index=0, key=f"lin_ref_{c}")
        cat_ref[c] = ref

    use_cols = [dv] + ivs
    work = df[use_cols].copy()
    for c in [dv] + [x for x in ivs if x not in cat_vars]:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    n_before = work.shape[0]
    work = work.dropna()
    n_after = work.shape[0]
    st.sidebar.caption(f"Eksik nedeniyle düşen gözlem: {n_before - n_after}")

    # Univariate OLS
    st.header("🔹 Univariate Linear Regression")
    uni_rows = []
    for var in ivs:
        try:
            fml = build_formula(dv, [var], cat_info=cat_ref)
            _, res = fit_ols(fml, work)
            coef = res.params[1] if len(res.params) > 1 else np.nan
            ci = res.conf_int()
            lo = ci.iloc[1, 0] if ci.shape[0] > 1 else np.nan
            hi = ci.iloc[1, 1] if ci.shape[0] > 1 else np.nan
            p = res.pvalues[1] if len(res.pvalues) > 1 else np.nan
            uni_rows.append({
                "değişken": var,
                "β (95% GA)": f"{coef:.3f} ({lo:.3f}–{hi:.3f})" if not pd.isna(coef) else "NA",
                "p": p,
                "R²": res.rsquared,
                "AIC": res.aic,
                "BIC": res.bic
            })
        except Exception:
            uni_rows.append({"değişken": var, "β (95% GA)": "NA", "p": np.nan, "R²": np.nan, "AIC": np.nan, "BIC": np.nan})
    uni_df = pd.DataFrame(uni_rows).sort_values("p", na_position='last')
    st.dataframe(uni_df, use_container_width=True)
    st.download_button(
        "Univariate OLS (CSV)",
        uni_df.to_csv(index=False).encode("utf-8"),
        file_name="univariate_ols.csv",
        mime="text/csv"
    )

    # Multivariate OLS
    st.header("🔹 Multivariate Linear Regression")
    p_thresh = st.slider("p-eşiği (Univariate → ön seçim)", 0.001, 0.20, 0.05, 0.001, key="lin_p")
    preselect = uni_df.loc[uni_df["p"] <= p_thresh, "değişken"].tolist()
    st.caption(f"Eşik altı: {', '.join(preselect) if preselect else '(yok)'}")
    manual_multi = st.multiselect("Multivariate değişkenleri", options=ivs, default=preselect, key="lin_multi")

    if manual_multi:
        fml = build_formula(dv, manual_multi, cat_ref)
        with st.expander("Kullanılan formül"):
            st.code(fml)
        try:
            model, res = fit_ols(fml, work)
            summ = res.summary2().tables[1].reset_index().rename(columns={"index": "variable"})
            ci = res.conf_int()
            ci.columns = ["ci_low", "ci_high"]
            summ = summ.merge(ci, left_on="variable", right_index=True, how="left")
            summ["β (95% GA)"] = summ.apply(
                lambda r: f"{r['Coef.']:.3f} ({r['ci_low']:.3f}–{r['ci_high']:.3f})" if pd.notna(r["Coef."]) else "NA", axis=1
            )
            pretty = summ[["variable", "β (95% GA)", "P>|t|"]]
            st.subheader("Model Katsayıları")
            st.dataframe(pretty, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("R² / Adj-R²", f"{res.rsquared:.3f} / {res.rsquared_adj:.3f}")
            c2.metric("AIC", f"{res.aic:.2f}")
            c3.metric("BIC", f"{res.bic:.2f}")

            # Artık grafikleri
            st.subheader("Artık (Residual) İnceleme")
            fitted = res.fittedvalues
            resid = res.resid
            fig1 = plt.figure()
            plt.scatter(fitted, resid)
            plt.axhline(0, linestyle='--')
            plt.xlabel("Fitted")
            plt.ylabel("Residuals")
            plt.title("Residuals vs Fitted")
            st.pyplot(fig1, use_container_width=True)

            from statsmodels.graphics.gofplots import qqplot
            fig2 = plt.figure()
            qqplot(resid, line='s', ax=plt.gca())
            plt.title("QQ Plot (Residuals)")
            st.pyplot(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Linear model hatası: {e}")
    else:
        st.info("Multivariate için en az bir değişken seçin.")

elif mode == "Multinomial (Logistic)":
    # --------- MULTINOMIAL --------- #
    # 1. Seçimler
    dv = st.sidebar.selectbox("Bağımlı Değişken (Kategorik > 2)", options=all_cols)
    levels = sorted([str(x) for x in df[dv].dropna().unique()])
    
    if len(levels) < 3:
        st.warning(f"Dikkat: '{dv}' değişkeninin {len(levels)} seviyesi var. Binary model daha uygun olabilir.")

    ref_cat = st.sidebar.selectbox("Referans Kategori", options=levels, index=0)
    
    candidates = [c for c in all_cols if c != dv]
    ivs = st.sidebar.multiselect("Bağımsız Değişkenler", options=candidates, default=candidates)
    
    if ivs:
        # Kategorik tanımları
        cat_vars = st.sidebar.multiselect("Kategorik Bağımsızlar", options=ivs, default=[c for c in ivs if df[c].dtype == 'object'])
        cat_ref = {}
        for c in cat_vars:
            lvs = sorted([str(x) for x in pd.Series(df[c]).dropna().unique()])
            ref = st.sidebar.selectbox(f"Referans – {c}", options=lvs, index=0, key=f"mn_ref_{c}")
            cat_ref[c] = ref
        
        st.header("🔹 Multinomial Logistic Regression")
        
        # 2. Veri Hazırlığı
        use_cols = [dv] + ivs
        work = df[use_cols].dropna().copy()
        
        # Hedef değişkeni string yap ve kategorik olarak sırala (Referans en başa)
        work[dv] = work[dv].astype(str)
        ref_cat_str = str(ref_cat)
        
        unique_cats = sorted(work[dv].unique())
        if ref_cat_str in unique_cats:
            unique_cats.remove(ref_cat_str)
            unique_cats.insert(0, ref_cat_str) # Referansı ilk sıraya koy
        
        # Pandas Categorical tipine çevir
        work[dv] = pd.Categorical(work[dv], categories=unique_cats, ordered=True)
        
        # 3. Formül Oluşturma
        terms = []
        for v in ivs:
            if v in cat_ref:
                terms.append(f"C({v}, Treatment(reference='{cat_ref[v]}'))")
            else:
                terms.append(v)
        rhs = " + ".join(terms)
        formula_str = f"{dv} ~ {rhs}"
        
        st.code(formula_str, language="python")
        
        try:
            model = smf.mnlogit(formula_str, data=work)
            res = model.fit(disp=0, maxiter=500)
            
            st.write(f"**Pseudo R² (McFadden):** {res.prsquared:.4f}")
            st.caption("Not: Katsayılar Relative Risk Ratio (RRR) olarak verilmiştir.")
            
            # Sonuçları sekmelere böl (Her sınıf vs Referans)
            comp_classes = res.params.columns.tolist() 
            tabs = st.tabs([f"{c} vs {ref_cat}" for c in comp_classes])
            
            all_dfs = []
            for idx, cls_name in enumerate(comp_classes):
                with tabs[idx]:
                    tbl = extract_rrr_table(res, idx, cls_name)
                    # Format
                    tbl["RRR (95% CI)"] = tbl.apply(lambda r: f"{r['RRR']:.3f} ({r['RRR_low']:.3f}–{r['RRR_high']:.3f})", axis=1)
                    tbl["p"] = tbl["p"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
                    
                    st.dataframe(tbl[["variable", "RRR (95% CI)", "p"]], use_container_width=True)
                    all_dfs.append(tbl)
            
            if all_dfs:
                final_res = pd.concat(all_dfs, ignore_index=True)
                st.download_button("Tüm Sonuçlar (CSV)", final_res.to_csv(index=False).encode("utf-8"), "multinomial_results.csv")
                
        except Exception as e:
            st.error(f"Multinomial Model Hatası: {e}")
            st.warning("Değişken sayınız örneklem sayısına göre çok fazla olabilir veya sınıflarda yeterli dağılım yok.")
    else:
        st.info("Lütfen bağımsız değişken seçin.")

elif mode == "Penalized (Lasso/Ridge)":
    # --------- PENALIZED --------- #
    st.header("🔹 Penalized Regression (Lasso / Ridge)")
    st.markdown("Yüksek boyutlu verilerde değişken seçimi ve regularization için.")
    
    pen_type = st.sidebar.radio("Metod", ["Lasso (L1)", "Ridge (L2)", "ElasticNet"], horizontal=True)
    target_type = st.sidebar.radio("Hedef Tipi", ["Continuous", "Binary"], horizontal=True)
    
    dv = st.sidebar.selectbox("Bağımlı Değişken", options=all_cols)
    candidates = [c for c in all_cols if c != dv]
    ivs = st.sidebar.multiselect("Bağımsız Değişkenler", options=candidates, default=candidates)
    
    if ivs:
        work = df[[dv] + ivs].dropna().copy()
        
        # X ve y Hazırlığı (Dummies + Scaling)
        y = work[dv].values
        X_raw = work[ivs]
        # Kategorikleri dummy'e çevir (drop_first=True ile)
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        feature_names = X_encoded.columns.tolist()
        
        # Scaling şarttır
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Model seçimi
        try:
            model = None
            if target_type == "Continuous":
                if pen_type == "Lasso (L1)":
                    model = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
                elif pen_type == "Ridge (L2)":
                    model = RidgeCV(cv=5).fit(X_scaled, y)
                else:
                    model = ElasticNetCV(cv=5, random_state=42).fit(X_scaled, y)
            else: # Binary
                penalty = 'l1' if pen_type == "Lasso (L1)" else 'l2'
                if pen_type == "ElasticNet": penalty = 'elasticnet'
                solver = 'liblinear' if penalty=='l1' else 'lbfgs'
                if penalty=='elasticnet': solver='saga'
                
                # LogisticRegressionCV regularization path'i otomatik tarar
                model = LogisticRegressionCV(cv=5, penalty=penalty, solver=solver, random_state=42, max_iter=1000).fit(X_scaled, y)

            st.success(f"Model ({pen_type}) tamamlandı.")
            
            # Katsayıları çek
            if target_type == "Continuous":
                coefs = model.coef_
                alpha_val = model.alpha_
            else:
                coefs = model.coef_[0]
                alpha_val = model.C_[0] # Inverse regularization strength
                
            st.write(f"**Seçilen Alpha/Lambda:** {alpha_val:.4f}")
            
            coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
            coef_df["Abs"] = coef_df["Coefficient"].abs()
            coef_df = coef_df.sort_values("Abs", ascending=False).drop(columns="Abs")
            
            # Görselleştirme
            st.subheader("Katsayı Önem Düzeyleri")
            st.dataframe(coef_df.style.background_gradient(cmap="coolwarm", subset=["Coefficient"]), use_container_width=True)
            
            # Sıfırlananlar (Lasso için önemli)
            if pen_type != "Ridge (L2)":
                zeros = coef_df[coef_df["Coefficient"].abs() < 1e-5]["Feature"].tolist()
                if zeros:
                    st.info(f"**Modelden çıkarılan (sıfırlanan) değişkenler:** {', '.join(zeros)}")
                else:
                    st.info("Hiçbir değişken sıfırlanmadı.")
                    
            fig, ax = plt.subplots(figsize=(8, len(feature_names)*0.3 + 2))
            indices = np.argsort(coefs)
            ax.barh(range(len(indices)), coefs[indices], align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(np.array(feature_names)[indices])
            ax.set_xlabel('Coefficient Value (Scaled)')
            ax.set_title(f'{pen_type} Coefficients')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Hata: {e}")
            st.caption("Binary hedef için verinin 0/1 olduğundan ve kategorik değişkenlerin düzgün olduğundan emin olun.")

# ===================== Raporlama İpuçları ===================== #
st.markdown(
"""
---
**Raporlama notları:**
- **Binary (Logistic):** Univariate ve Multivariate için OR, %95 GA, p; HL testi; ROC AUC; **Youden cut-off** ve o kesimde Sens/Spec/PPV/NPV/LR±; **DeLong testi** ile AUC farkı; gerekirse **Firth** ile stabil sonuç.
- **Linear (OLS):** β, %95 GA, p; R²/Adj-R²; AIC/BIC; residual incelemeleri.
- Kategorik değişkenlerde **referans kategori**yi belirtmeyi unutmayın.
"""
)
