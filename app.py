# app.py
# ============================================================
# 🔬 Univariate & Multivariate Lojistik Regresyon (Streamlit)
# Yazar: ChatGPT (GPT-5 Thinking)
# Açıklama: CSV/XLSX/SAV yükleyin → Bağımlı değişkeni ve aday değişkenleri seçin →
# Univariate tarama + Multivariate model (OR, %95 GA, p, AIC/BIC, HL testi, ROC AUC)
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
if not HAS_PYREADSTAT:
st.error(".sav dosyası için 'pyreadstat' gerekir. Lütfen 'pip install pyreadstat' kurun.")
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



