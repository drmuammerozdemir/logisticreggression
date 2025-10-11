# ============================================================
# 🔬 Binary (Logistic) + Continuous (Linear) Regression Toolkit
# Univariate tarama + Multivariate model + (Binary) ROC–AUC, Youden cut-off, DeLong testi
# ============================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from delong import delong_ci, delong_roc_test

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
mode = st.sidebar.selectbox("Seçin", ["Binary (Logistic)", "Continuous (Linear)"])

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
                OR, lo, hi = rows["OR"].iloc[0], rows["OR_low"].iloc[0], rows["OR_high"].iloc[0]
            else:
                OR, lo, hi = (np.nan, np.nan, np.nan)
            uni_rows.append({
                "değişken": var,
                "OR (95% GA)": format_or_ci(OR, lo, hi),
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

            # ROC (TEK KERE)
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

            # ==== DeLong Karşılaştırma (Model vs Tek Belirteç / İki Skor) ====
            st.subheader("ROC Karşılaştırma (DeLong)")

            # 1) Model skorları zaten y_prob
            # 2) Karşılaştırılacak ikinci skoru seç: herhangi bir sayısal sütun
            num_cols = [c for c in work.columns if c != dv_use and np.issubdtype(work[c].dtype, np.number)]
            compare_var = st.selectbox(
                "Karşılaştırılacak ikinci skor/değişken (örn. TRP, 3OHKYN):",
                options=num_cols
            )

            if compare_var:
                try:
                    y2 = pd.to_numeric(work[compare_var], errors="coerce").values
                    mask = ~np.isnan(y2)
                    # Ortak örneklem
                    y_true_cmp = y_true[mask]
                    y_prob_cmp = y_prob[mask]
                    y2_cmp = y2[mask]

                    # DeLong: AUC GA'ları ve fark testi
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

            # 1) Yardımcı: p formatı ve kalın vurgulama
            def fmt_p(p):
                if pd.isna(p):
                    return ""
                try:
                    return "<0.001" if p < 0.001 else f"{p:.3f}"
                except Exception:
                    return str(p)

            # 2) Multivariate sonuçlarını sözlüğe çek (Intercept hariç)
            multi_raw = res_m.summary2().tables[1].reset_index().rename(columns={"index":"term"})
            ci = res_m.conf_int()
            ci.columns = ["ci_low","ci_high"]
            multi_raw = multi_raw.merge(ci, left_on="term", right_index=True, how="left")
            multi_raw["OR"] = np.exp(multi_raw["Coef."])
            multi_raw["OR_low"] = np.exp(multi_raw["ci_low"])
            multi_raw["OR_high"] = np.exp(multi_raw["ci_high"])
            multi_raw = multi_raw[~multi_raw["term"].str.contains("Intercept", case=False, na=False)].copy()

            # C(kat) isimlerini biraz insanileştir
            def clean_term(t):
                # C(var)[T.level] -> var (level)
                if t.startswith("C(") and ")[T." in t:
                    base = t.split("C(")[1].split(")")[0]
                    lev = t.split("[T.")[1].rstrip("]")
                    return f"{base}: {lev}"
                return t

            multi_raw["clean"] = multi_raw["term"].map(clean_term)
            multi_map_or = dict(zip(multi_raw["clean"], 
                                    [f"{orv:.3f} ({lo:.3f}–{hi:.3f})" 
                                     for orv, lo, hi in zip(multi_raw["OR"], multi_raw["OR_low"], multi_raw["OR_high"])]))
            multi_map_p  = dict(zip(multi_raw["clean"], multi_raw["P>|z|"]))

            # 3) Univariate tablosundan (zaten app’te hazır) çek
            # uni_df: kolon isimleri -> ["değişken","OR (95% GA)","p","AIC","BIC"]
            uni_tmp = uni_df[["değişken","OR (95% GA)","p"]].copy()
            uni_tmp.rename(columns={"değişken":"Variable",
                                    "OR (95% GA)":"Univariate OR (95% CI)",
                                    "p":"Univariate P"}, inplace=True)

            # 4) (İsteğe bağlı) ölçekli gösterim: örn. SII/100 ikinci satır
            st.caption("İsteğe bağlı: bir değişkeni ölçekleyerek (örn. SII/100) ek satır oluştur.")
            add_scaled = st.checkbox("Ölçekli satır ekle (örn. SII/100)", value=False)
            if add_scaled:
                scale_var = st.selectbox("Ölçeklenecek değişken", options=uni_tmp["Variable"].tolist())
                scale_val = st.number_input("Ölçek katsayısı (örn. 100 → SII/100)", min_value=1.0, value=100.0, step=1.0)
                # Univariate için yeni isimli satır kopyası (OR aynı; sadece isim değişir – rapor amaçlı)
                r = uni_tmp.loc[uni_tmp["Variable"] == scale_var].copy()
                if not r.empty:
                    r = r.assign(Variable = scale_var + f" / {int(scale_val)}")
                    uni_tmp = pd.concat([uni_tmp, r], ignore_index=True)
                # Multivariate için de aynı ismi aramaya çalışırız (bulunmazsa "/")
                # Not: Gerçek OR'u farklı ölçekle yeniden fit etmek istersen ek model gerekir.

            # 5) Birleştir
            # Multivariate değerlerini isimle eşleştir (aynı isimde satır varsa doldurur; yoksa "/")
            def take_multi_or(name):
                return multi_map_or.get(name, "/")
            def take_multi_p(name):
                p = multi_map_p.get(name, np.nan)
                return fmt_p(p) if not pd.isna(p) else "/"

            out = uni_tmp.copy()
            out["Multivariate OR (95% CI)"] = out["Variable"].map(take_multi_or)
            out["Multivariate P"] = out["Variable"].map(take_multi_p)
            # Univariate p biçimlendir
            out["Univariate P"] = out["Univariate P"].apply(fmt_p)

            # Yayın sırası için sütunları düzenle
            out = out[["Variable", 
                       "Univariate OR (95% CI)", "Univariate P", 
                       "Multivariate OR (95% CI)", "Multivariate P"]]

            # 6) Stil: p<0.05 vurgusu
            def highlight_sig(s):
                try:
                    # "<0.001" veya "0.012" gibi biçimlere bak
                    val = s.replace("<","")
                    return float(val) < 0.05
                except Exception:
                    return False

            styler = out.style.format(na_rep="").applymap(
                lambda v: "font-weight:bold;" if isinstance(v, str) and highlight_sig(v) else ""
            , subset=["Univariate P","Multivariate P"])

            st.dataframe(styler, use_container_width=True)

            # 7) İndirme
            st.download_button(
                "Özet Tablo (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="summary_uni_multi.csv",
                mime="text/csv"
            )
            # Excel de isteyenler için
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

else:
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
            # β ve %95 GA
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

            # Basit QQ
            from statsmodels.graphics.gofplots import qqplot
            fig2 = plt.figure()
            qqplot(resid, line='s', ax=plt.gca())
            plt.title("QQ Plot (Residuals)")
            st.pyplot(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Linear model hatası: {e}")
    else:
        st.info("Multivariate için en az bir değişken seçin.")

# ===================== Raporlama İpuçları ===================== #
st.markdown(
"""
---
**Raporlama notları:**
- **Binary (Logistic):** Univariate ve Multivariate için OR, %95 GA, p; HL testi; ROC AUC; **Youden cut-off** ve o kesimde Sens/Spec/PPV/NPV/LR±; **DeLong testi** ile AUC farkı.
- **Linear (OLS):** β, %95 GA, p; R²/Adj-R²; AIC/BIC; residual incelemeleri.
- Kategorik değişkenlerde **referans kategori**yi belirtmeyi unutmayın.
"""
)
