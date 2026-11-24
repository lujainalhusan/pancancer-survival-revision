#!/usr/bin/env python3
"""
Revision analyses for "An Evaluation of Tumor Type and Cohort Effects in
Pan Cancer Survival Prediction" (IMU D 26 00709).

The analyses answer the comments of Reviewer 1 (R1.1 to R1.8) and
Reviewer 2 (R2.1 to R2.10). Every feature level step (imputation, variance
filter, scaling, PCA, L1 selection and the clinical encoders) is fitted inside
each training fold only and then applied unchanged to the matching test fold.

Inputs (command line arguments expr and clin)
  expr    UCSC Xena TCGA PANCAN expression matrix, genes by samples, tab
          separated, optionally gzipped. The first column holds gene names.
          The matrix used in the manuscript is the PanCanAtlas EB++ batch
          adjusted RSEM matrix (log2(x + 1), 20,531 genes by 11,069 samples).
  clin    Clinical table. Either the UCSC Xena survival file keyed by sample
          (tcga_survival.tsv) or the TCGA Clinical Data Resource workbook.

Usage is shown in README.md. Outputs are CSV tables, summary.txt and
software_versions.txt in the output folder. File names start with the reviewer comment they answer.
"""
import argparse, json, os, platform, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42

# Column names in the clinical table
C_ID, C_TYPE, C_AGE, C_SEX = ("bcr_patient_barcode", "type",
                               "age_at_initial_pathologic_diagnosis", "gender")
C_STAGE, C_CSTAGE, C_YEAR = ("ajcc_pathologic_tumor_stage", "clinical_stage",
                             "initial_pathologic_dx_year")
C_OS, C_OST = "OS", "OS.time"

# Candidate genes reported in the submitted manuscript (Table 6)
TOP_GENES = ["UPF3B", "RNF212", "VSIG4", "COBLL1", "C21orf29"]

# TCGA sample type codes kept as the primary tumor of each patient:
# 01 primary solid tumor, 03 primary blood derived cancer (LAML)
PRIMARY_CODES = ["01", "03"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# Data loading and patient level cohort

def load_clinical(path):
    """Reads the clinical table and returns one row per patient.

    The Xena survival file is keyed by sample, so a patient with a tumor and a
    normal sample appears twice. One row per patient is kept, preferring a
    primary tumor sample. Survival and clinical fields are patient level, so
    the choice of row does not change them."""
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="TCGA-CDR")
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    n_records = len(df)
    if "sample" in df.columns and C_ID not in df.columns:
        df = df.rename(columns={"cancer type abbreviation": C_TYPE})
        df[C_ID] = df["sample"].str[:12]
        code = df["sample"].str[13:15]
        df["_not_primary"] = (~code.isin(PRIMARY_CODES)).astype(int)
        df = (df.sort_values(["_not_primary", "sample"])
                .drop_duplicates(C_ID).drop(columns="_not_primary"))
    for c in (C_AGE, C_YEAR, C_OS, C_OST):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["n_records"] = n_records
    return df


def stage_group(s):
    """Maps an AJCC stage string to I, II, III or IV.

    Sub stages are collapsed to their main group (Stage IIA to II, Stage IVB
    to IV, Stage IB1 to I, IIa to II). Stage IS (testicular germ cell) maps to
    I. Strings without a main group (Stage 0, Stage X, I/II NOS, Discrepancy,
    Unknown, empty) are returned as NA."""
    s = str(s).strip().upper()
    rest = s[6:] if s.startswith("STAGE ") else s
    for g in ("IV", "III", "II", "I"):
        if rest.startswith(g) and not rest.startswith(g + "/"):
            return g
    return "NA"


def build_cohort(expr_path, clin_path, out):
    """Builds the patient level cohort and writes the patient flow table (R2.4,
    R1.8). Returns the expression matrix (patients by genes), the clinical
    table aligned to it, and the full clinical table of patients with complete
    overall survival (used for thresholds and the exclusion analysis)."""
    flow = []
    clin_raw = load_clinical(clin_path)
    flow.append(("Clinical records in source file (sample level rows)", clin_raw.attrs["n_records"]))
    clin = clin_raw.drop_duplicates(C_ID)
    flow.append(("Unique patients in clinical table", len(clin)))
    clin_full = clin.dropna(subset=[C_OS, C_OST]).copy()
    flow.append(("Patients with complete OS (status and time)", len(clin_full)))

    log("reading expression matrix")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0).astype(np.float32)
    flow.append(("Expression samples (all columns of the matrix)", expr.shape[1]))
    expr = expr.reindex(sorted(expr.columns), axis=1)
    code = expr.columns.str[13:15]
    for c, lab in (("11", "solid tissue normal (11)"), ("06", "metastatic (06)"),
                   ("02", "recurrent tumor (02)"), ("05", "additional new primary (05)"),
                   ("07", "additional metastatic (07)")):
        flow.append((f"Removed: {lab}", int((code == c).sum())))
    keep = code.isin(PRIMARY_CODES)
    expr = expr.loc[:, keep]
    flow.append(("Primary tumor samples (01, or 03 for LAML)", expr.shape[1]))
    pid = expr.columns.str[:12]
    dup = pid.duplicated()
    flow.append(("Removed: repeated primary tumor aliquots of the same patient "
                 "(first aliquot in barcode order kept)", int(dup.sum())))
    expr = expr.loc[:, ~dup]
    expr.columns = expr.columns.str[:12]
    flow.append(("Unique patients with a primary tumor expression profile", expr.shape[1]))

    common = sorted(set(expr.columns) & set(clin_full[C_ID]))
    flow.append(("Patients with both expression and complete OS", len(common)))
    clin = clin_full.set_index(C_ID).loc[common].copy()
    X = expr[common].T.astype(np.float64)
    clin["stage"] = clin[C_STAGE].map(stage_group)
    if C_CSTAGE in clin:
        m = clin["stage"].eq("NA")
        clin.loc[m, "stage"] = clin.loc[m, C_CSTAGE].map(stage_group)
    clin["site"] = clin.index.str[5:7]          # tissue source site code from the barcode
    clin["sex"] = clin[C_SEX].astype(str).str.upper().replace({"NAN": "NA"})
    pd.DataFrame(flow, columns=["step", "n"]).to_csv(f"{out}/R2_4_patient_flow.csv", index=False)

    # Missing data in the clinical variables used by the models (R1.8)
    miss = pd.DataFrame({
        "variable": ["age at diagnosis", "sex", "stage (pathologic, clinical used when missing)",
                     "pathologic stage only", "year of diagnosis", "expression values (cells)"],
        "n_missing": [int(clin[C_AGE].isna().sum()), int((clin["sex"] == "NA").sum()),
                      int((clin["stage"] == "NA").sum()),
                      int((clin[C_STAGE].map(stage_group) == "NA").sum()),
                      int(clin[C_YEAR].isna().sum()), int(np.isnan(X.values).sum())],
        "handling": ["training fold median imputation", "own category",
                     "own category (unknown stage)", "clinical stage substituted",
                     "training fold median imputation", "training fold gene mean imputation"]})
    miss["n_patients"] = len(clin)
    miss.to_csv(f"{out}/R1_8_missing_data.csv", index=False)
    return X, clin, clin_full


# Risk labels

def global_thresholds(clin_full):
    """Median and 75th percentile of survival time among confirmed deaths."""
    d = clin_full.loc[clin_full[C_OS] == 1, C_OST]
    return float(d.median()), float(d.quantile(0.75))


def make_labels(clin, mode="global", t=None, clin_full=None):
    """Returns 1 for high risk, 0 for low risk and NaN for excluded patients.

    global    died before the median (high); died at or after the median, or
              censored after the 75th percentile (low); other censored excluded
    per_type  the same rule with thresholds computed within each tumor type
    landmark  died before t days (high); followed beyond t days, alive or not
              (low); censored before t days excluded. With t equal to the
              global median this is the symmetric version of the global rule."""
    T, E = clin[C_OST].values, clin[C_OS].values
    y = np.full(len(clin), np.nan)
    if mode == "global":
        hi, lo = t
        y[(E == 1) & (T < hi)] = 1
        y[(E == 1) & (T >= hi)] = 0
        y[(E == 0) & (T > lo)] = 0
    elif mode == "per_type":
        ref = clin_full if clin_full is not None else clin
        for ty, idx in clin.groupby(C_TYPE).indices.items():
            d = ref.loc[(ref[C_TYPE] == ty) & (ref[C_OS] == 1), C_OST]
            if len(d) < 10:
                continue
            hi, lo = d.median(), d.quantile(0.75)
            Ti, Ei = T[idx], E[idx]
            yi = np.full(len(idx), np.nan)
            yi[(Ei == 1) & (Ti < hi)] = 1
            yi[(Ei == 1) & (Ti >= hi)] = 0
            yi[(Ei == 0) & (Ti > lo)] = 0
            y[idx] = yi
    elif mode == "landmark":
        h = t
        y[(E == 1) & (T < h)] = 1
        y[T >= h] = 0
    return y


# Feature construction

def tx_transformer(n_pc=150, C=0.05):
    """Transcriptome only representation. Tumor type never enters this block.
    PCA scores are standardized before L1 selection so that the penalty treats
    all components alike."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("var", VarianceThreshold(0.5)),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pc, random_state=SEED, svd_solver="randomized")),
        ("sc_pc", StandardScaler()),
        ("l1", SelectFromModel(LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                                  class_weight="balanced",
                                                  random_state=SEED))),
    ])


class ClinEncoder:
    """Encodes tumor type, age, sex, stage, year and site; fitted on the
    training fold only. Categorical columns are one hot coded (unseen levels
    in a test fold are coded as all zeros); numeric columns are median imputed
    and standardized."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, d):
        self.cat = [c for c in self.cols if c in ("type", "sex", "stage", "site")]
        self.num = [c for c in self.cols if c in ("age", "year")]
        self.ohe = (OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[self.cat])
                    if self.cat else None)
        if self.num:
            self.imp = SimpleImputer(strategy="median").fit(d[self.num])
            self.sc = StandardScaler().fit(self.imp.transform(d[self.num]))
        return self

    def transform(self, d):
        parts = []
        if self.cat:
            parts.append(self.ohe.transform(d[self.cat]))
        if self.num:
            parts.append(self.sc.transform(self.imp.transform(d[self.num])))
        return np.hstack(parts)

    def names(self):
        out = list(self.ohe.get_feature_names_out()) if self.cat else []
        return out + list(self.num)


def clin_frame(clin):
    return pd.DataFrame({"type": clin[C_TYPE].astype(str).values,
                         "age": clin[C_AGE].values,
                         "sex": clin["sex"].values,
                         "stage": clin["stage"].values,
                         "year": clin[C_YEAR].values,
                         "site": clin["site"].values}, index=clin.index)


# Predictor sets (R2.2, R1.1). The transcriptome block never sees tumor type.
FEATURE_SETS = {
    "transcriptome_only": ("tx", []),
    "tumor_type_only": (None, ["type"]),
    "age_only": (None, ["age"]),
    "site_only": (None, ["site"]),
    "clinical_only": (None, ["type", "age", "sex", "stage"]),
    "clinical_plus_year": (None, ["type", "age", "sex", "stage", "year"]),
    "transcriptome_plus_tumor_type": ("tx", ["type"]),
    "combined": ("tx", ["type", "age", "sex", "stage"]),
}


def clf():
    return LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                              max_iter=2000, random_state=SEED)


def other_classifiers():
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    return {"svm_rbf": CalibratedClassifierCV(SVC(C=1.0, gamma="scale", class_weight="balanced",
                                                  random_state=SEED), method="sigmoid", cv=3),
            "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                    n_jobs=-1, random_state=SEED)}


def fit_predict_sets(Xtr, Xte, Ctr, Cte, ytr, sets=FEATURE_SETS, n_pc=150, other_models=False):
    """Fits the transcriptome transformer once per fold and every predictor set
    on top of it. Returns test fold probabilities and the fitted transformer."""
    need_tx = any(v[0] for v in sets.values())
    tx = None
    if need_tx:
        tx = tx_transformer(n_pc).fit(Xtr, ytr)
        Ztr, Zte = tx.transform(Xtr), tx.transform(Xte)
    preds = {}
    for name, (t, cols) in sets.items():
        A, B = [], []
        if t:
            A.append(Ztr); B.append(Zte)
        if cols:
            enc = ClinEncoder(cols).fit(Ctr)
            A.append(enc.transform(Ctr)); B.append(enc.transform(Cte))
        A, B = np.hstack(A), np.hstack(B)
        preds[name] = clf().fit(A, ytr).predict_proba(B)[:, 1]
        if other_models and name in ("transcriptome_only", "transcriptome_plus_tumor_type"):
            for tag, est in other_classifiers().items():
                preds[f"{name}|{tag}"] = clone(est).fit(A, ytr).predict_proba(B)[:, 1]
    return preds, tx


# Metrics

MIN_PER_CLASS = 3


def within_type_auc(y, p, types, min_n=10, min_class=MIN_PER_CLASS):
    """AUC inside each tumor type with at least min_n test patients and at
    least min_class patients in each risk group, averaged with weights equal
    to the number of test patients. Returns the weighted mean and the per type
    rows."""
    rows = []
    for ty in np.unique(types):
        m = types == ty
        h = int(y[m].sum())
        if m.sum() >= min_n and min(h, int(m.sum()) - h) >= min_class:
            rows.append((ty, int(m.sum()), int(y[m].sum()), roc_auc_score(y[m], p[m])))
    if not rows:
        return np.nan, rows
    w = np.array([r[1] for r in rows], float)
    return float(np.average([r[3] for r in rows], weights=w)), rows


def calibration(y, p):
    """Calibration slope and intercept (logistic recalibration) and expected
    calibration error over deciles of predicted risk."""
    eps = 1e-6
    q = np.clip(p, eps, 1 - eps)
    lp = np.log(q / (1 - q))
    m = LogisticRegression(C=1e6, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    slope, inter = float(m.coef_[0, 0]), float(m.intercept_[0])
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    ece = float(np.sum([abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                        for b in np.unique(bins)]))
    return slope, inter, ece


def delong(y, p1, p2):
    """Paired DeLong test for two correlated AUCs. Returns AUC1, AUC2, the
    difference, z and the two sided p value."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)

    def comp(p):
        X, Y = p[pos], p[neg]
        V10 = ((X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])).mean(1)
        V01 = ((X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])).mean(0)
        return V10.mean(), V10, V01
    a1, v10a, v01a = comp(p1)
    a2, v10b, v01b = comp(p2)
    S = np.cov(np.vstack([v10a, v10b])) / m + np.cov(np.vstack([v01a, v01b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, a1 - a2, z, 2 * stats.norm.sf(abs(z))


def boot_auc_ci(y, p, n_boot=2000, seed=SEED):
    """Percentile bootstrap interval for an AUC, resampling within each class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(df, by, cols):
    """Mean and 2.5th and 97.5th percentiles across splits."""
    g = df.groupby(by)
    r = g[cols].mean().add_suffix("_mean")
    for c in cols:
        r[c + "_lo"] = g[c].quantile(0.025)
        r[c + "_hi"] = g[c].quantile(0.975)
    return r.reset_index()


# Repeated cross validation (R2.2, R2.5, R2.6, R2.8, R1.3, R1.4)

def strat_labels(y, types, n_splits):
    """Joint risk group by tumor type strata; rare strata fall back to the risk
    group alone so that every stratum can be split."""
    strat = pd.Series(y.astype(int).astype(str)) + "_" + pd.Series(types)
    vc = strat.value_counts()
    return np.where(strat.map(vc) < n_splits, y.astype(int).astype(str), strat)


def repeated_cv(X, C, y, types, n_splits=5, repeats=20, sets=FEATURE_SETS, n_pc=150, tag="",
                other_models=False):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    rows, diffs, per_type, n_sel = [], [], [], []
    for k, (tr, te) in enumerate(cv.split(X, strat_labels(y, types, n_splits))):
        preds, tx = fit_predict_sets(X[tr], X[te], C.iloc[tr], C.iloc[te], y[tr], sets, n_pc,
                                     other_models)
        if tx is not None:
            n_sel.append(dict(split=k, n_components=int(tx.named_steps["l1"].get_support().sum()),
                              n_genes=int(tx.named_steps["var"].get_support().sum())))
        res = {}
        for name, p in preds.items():
            auc = roc_auc_score(y[te], p)
            wt, per = within_type_auc(y[te], p, types[te])
            sl, ic, ece = calibration(y[te], p)
            res[name] = (auc, wt)
            rows.append(dict(split=k, feature_set=name, pooled_auc=auc, within_type_auc=wt,
                             pooling_gap=auc - wt, brier=brier_score_loss(y[te], p),
                             cal_slope=sl, cal_intercept=ic, ece=ece))
            if name in ("transcriptome_only", "clinical_only", "combined"):
                per_type += [dict(split=k, feature_set=name, type=t, n_test=n, n_high=h, auc=a)
                             for t, n, h, a in per]
        for a, b in [("transcriptome_only", "clinical_only"), ("combined", "clinical_only"),
                     ("combined", "transcriptome_only"), ("transcriptome_only", "tumor_type_only"),
                     ("transcriptome_plus_tumor_type", "tumor_type_only"),
                     ("clinical_plus_year", "clinical_only")]:
            if a in res and b in res:
                diffs.append(dict(split=k, comparison=f"{a} minus {b}",
                                  d_pooled=res[a][0] - res[b][0],
                                  d_within=res[a][1] - res[b][1]))
        if k % 10 == 0:
            log(f"{tag} split {k + 1}/{n_splits * repeats}")
    return pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(per_type), pd.DataFrame(n_sel)


# Negative controls (R2.2, R2.7)

def permuted_within_type(X, C, y, types, repeats=5):
    """Shuffles risk labels within each tumor type. Any pooled AUC that remains
    comes only from differences in event rates between tumor types."""
    rng = np.random.default_rng(SEED)
    out = []
    for r in range(repeats):
        yp = y.copy()
        for ty in np.unique(types):
            m = np.where(types == ty)[0]
            yp[m] = rng.permutation(yp[m])
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=yp, random_state=SEED + r)
        preds, _ = fit_predict_sets(X[tr], X[te], C.iloc[tr], C.iloc[te], yp[tr],
                                    {k: FEATURE_SETS[k] for k in ("transcriptome_only", "tumor_type_only")})
        for k, p in preds.items():
            out.append(dict(rep=r, feature_set=k, pooled_auc=roc_auc_score(yp[te], p),
                            within_type_auc=within_type_auc(yp[te], p, types[te])[0]))
    return pd.DataFrame(out)


def leave_one_type_out(X, C, y, types, min_n=40):
    """Trains on all other tumor types and tests on the held out type."""
    out = []
    for ty in np.unique(types):
        te = types == ty
        h = int(y[te].sum())
        if te.sum() < min_n or min(h, int(te.sum()) - h) < MIN_PER_CLASS:
            continue
        tr = ~te
        preds, _ = fit_predict_sets(X[tr], X[te], C.loc[tr], C.loc[te], y[tr],
                                    {"transcriptome_only": FEATURE_SETS["transcriptome_only"]})
        out.append(dict(held_out_type=ty, n=int(te.sum()), n_high=int(y[te].sum()),
                        auc=roc_auc_score(y[te], preds["transcriptome_only"])))
    return pd.DataFrame(out)


# Original single split: per tumor type accounting (R1.4) and what the
# transcriptome only components encode (R1.1, R2.7)

def per_type_accounting(y, types_all_cohort, types, te, preds, out, min_n=20):
    """For all tumor types: test set size, patients per risk group, whether the
    type entered the within type average and why not. Adds bootstrap intervals
    for each within type AUC and the pooled AUC restricted to the included
    types."""
    yt, tt = y[te], types[te]
    rows = []
    for ty in sorted(set(types_all_cohort)):
        m = tt == ty
        n, h = int(m.sum()), int(yt[m].sum())
        if n == 0:
            reason = "no labelled patients with expression in the test set"
        elif n < min_n:
            reason = f"fewer than {min_n} test patients"
        elif h == 0 or h == n:
            reason = "only one risk group in the test set"
        elif min(h, n - h) < MIN_PER_CLASS:
            reason = f"fewer than {MIN_PER_CLASS} test patients in one risk group"
        else:
            reason = ""
        r = dict(type=ty, n_cohort=int((types == ty).sum()), n_test=n, n_test_high=h,
                 n_test_low=n - h, included=reason == "", reason_excluded=reason)
        if reason == "":
            for k in ("transcriptome_only", "clinical_only", "combined"):
                a = roc_auc_score(yt[m], preds[k][m])
                lo, hi = boot_auc_ci(yt[m], preds[k][m])
                r[f"auc_{k}"], r[f"auc_{k}_lo"], r[f"auc_{k}_hi"] = a, lo, hi
        rows.append(r)
    acc = pd.DataFrame(rows)
    acc.to_csv(f"{out}/R1_4_per_type_accounting.csv", index=False)

    inc = np.isin(tt, acc.loc[acc.included, "type"].values)
    pooled = []
    for k, p in preds.items():
        if "|" in k:
            continue
        lo, hi = boot_auc_ci(yt, p)
        lo19, hi19 = boot_auc_ci(yt[inc], p[inc])
        pooled.append(dict(feature_set=k, pooled_auc_all_types=roc_auc_score(yt, p),
                           ci_lo=lo, ci_hi=hi, n_test_all=len(yt),
                           pooled_auc_included_types=roc_auc_score(yt[inc], p[inc]),
                           ci_lo_included=lo19, ci_hi_included=hi19, n_test_included=int(inc.sum()),
                           within_type_auc=within_type_auc(yt, p, tt, min_n)[0]))
    pd.DataFrame(pooled).to_csv(f"{out}/R1_4_pooled_auc_included_types.csv", index=False)
    return acc


def encoding_recovery(Ztr, Zte, Ctr, Cte, out, min_site=20):
    """Recovers tumor type, sex and contributing site from the transcriptome
    only components (fitted on the training split, scored on the test split)."""
    rows = []
    m = LogisticRegression(max_iter=3000, random_state=SEED).fit(Ztr, Ctr["type"])
    rows.append(dict(target="tumor type", metric="accuracy",
                     value=accuracy_score(Cte["type"], m.predict(Zte)),
                     chance=float(Cte["type"].value_counts(normalize=True).iloc[0]),
                     n_classes=int(Ctr["type"].nunique()), n_test=len(Cte)))
    ok_tr, ok_te = Ctr["sex"].isin(["MALE", "FEMALE"]).values, Cte["sex"].isin(["MALE", "FEMALE"]).values
    m = LogisticRegression(max_iter=3000, random_state=SEED).fit(Ztr[ok_tr], Ctr["sex"][ok_tr] == "FEMALE")
    rows.append(dict(target="sex", metric="AUC",
                     value=roc_auc_score(Cte["sex"][ok_te] == "FEMALE", m.predict_proba(Zte[ok_te])[:, 1]),
                     chance=0.5, n_classes=2, n_test=int(ok_te.sum())))
    big = Ctr["site"].value_counts()[lambda s: s >= min_site].index
    a, b = Ctr["site"].isin(big).values, Cte["site"].isin(big).values
    m = LogisticRegression(max_iter=3000, random_state=SEED).fit(Ztr[a], Ctr["site"][a])
    rows.append(dict(target="contributing site", metric="accuracy",
                     value=accuracy_score(Cte["site"][b], m.predict(Zte[b])),
                     chance=float(Cte["site"][b].value_counts(normalize=True).iloc[0]),
                     n_classes=len(big), n_test=int(b.sum())))
    pd.DataFrame(rows).to_csv(f"{out}/R1_1_encoding_recovery.csv", index=False)


# Cohort effects (R2.7, R1.5)

def cohort_effects(Z, C, out, min_site=15, min_sites=3):
    """Site recovery within tumor types, so that tumor composition cannot explain
    it, compared with a label permutation baseline; site recovery after the
    components are residualized on age, sex and stage; and partitioning of
    component variance into tumor type, site within type and clinical parts."""
    rows = []
    rng = np.random.default_rng(SEED)
    enc_c = ClinEncoder(["sex", "stage", "age"]).fit(C)
    D_c = np.hstack([np.ones((len(C), 1)), enc_c.transform(C)])
    for ty, g in C.groupby("type"):
        vc = g["site"].value_counts()
        sites = vc[vc >= min_site].index
        if len(sites) < min_sites:
            continue
        m = (C["type"] == ty).values & C["site"].isin(sites).values
        Zi, si = Z[m], C.loc[m, "site"].values
        # remove the part of each component explained by age, sex and stage
        Dc = D_c[m]
        b, *_ = np.linalg.lstsq(Dc, Zi, rcond=None)
        Zr = Zi - Dc @ b
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, C=0.1))])
        acc = cross_val_score(model, Zi, si, cv=cv).mean()
        acc_r = cross_val_score(model, Zr, si, cv=cv).mean()
        perm = np.mean([cross_val_score(model, Zi, rng.permutation(si), cv=cv).mean() for _ in range(5)])
        rows.append(dict(type=ty, n=int(m.sum()), n_sites=len(sites),
                         majority_baseline=float(pd.Series(si).value_counts(normalize=True).iloc[0]),
                         permuted_accuracy=perm, site_accuracy=acc,
                         site_accuracy_after_clinical_adjustment=acc_r))
    pd.DataFrame(rows).to_csv(f"{out}/R2_7_site_within_type.csv", index=False)

    def r2(design, z):
        b, *_ = np.linalg.lstsq(design, z, rcond=None)
        return 1 - ((z - design @ b) ** 2).sum(0) / ((z - z.mean(0)) ** 2).sum(0)
    one = np.ones((len(C), 1))
    D_t = np.hstack([one, pd.get_dummies(C["type"]).values.astype(float)])
    D_ts = np.hstack([D_t, pd.get_dummies(C["site"]).values.astype(float)])
    D_tsc = np.hstack([D_ts, enc_c.transform(C)])
    a, b, c = r2(D_t, Z), r2(D_ts, Z), r2(D_tsc, Z)
    pd.DataFrame({"component": range(1, Z.shape[1] + 1), "R2_type": a,
                  "R2_site_given_type": b - a, "R2_clinical_given_type_site": c - b,
                  "R2_unexplained": 1 - c}).to_csv(f"{out}/R2_7_variance_partition.csv", index=False)


# Censoring aware survival models on all patients (R2.1, R2.3, R2.9, R1.6)

def survival_analyses(X, C, T, E, types, out, n_pc=150, folds=5, repeats=2):
    """Cox models fitted to all patients with overall survival data, so no
    censored patient is excluded. The transcriptome block is unsupervised PCA,
    so no step uses the binary risk labels. Reports Harrell C, Uno C (tau 5
    years), within type C, time dependent AUC, integrated Brier score, 2 year
    calibration and Schoenfeld tests of proportional hazards."""
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                cumulative_dynamic_auc, integrated_brier_score)
    from sksurv.util import Surv

    horizons = np.array([365, 730, 1095, 1825])
    rows, calib = [], []
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)

    def unsup_tx():
        return Pipeline([("imp", SimpleImputer()), ("var", VarianceThreshold(0.5)),
                         ("sc", StandardScaler()),
                         ("pca", PCA(n_components=n_pc, random_state=SEED, svd_solver="randomized")),
                         ("sc_pc", StandardScaler())])

    specs = {
        "clinical": (False, ["age", "sex", "stage", "type"], False),
        "transcriptome": (True, [], False),
        "combined": (True, ["age", "sex", "stage", "type"], False),
        "clinical_stratified_by_type": (False, ["age", "sex", "stage"], True),
        "transcriptome_stratified_by_type": (True, [], True),
        "combined_stratified_by_type": (True, ["age", "sex", "stage"], True),
    }
    for k, (tr, te) in enumerate(cv.split(X, strat_labels(E, types, folds))):
        tx = unsup_tx().fit(X[tr])
        Ztr, Zte = tx.transform(X[tr]), tx.transform(X[te])
        s_tr = Surv.from_arrays(E[tr].astype(bool), T[tr])
        tmax_tr = T[tr].max()
        for name, (use_tx, cols, strata) in specs.items():
            A, B = [], []
            if use_tx:
                A.append(Ztr); B.append(Zte)
            if cols:
                enc = ClinEncoder(cols).fit(C.iloc[tr])
                A.append(enc.transform(C.iloc[tr])); B.append(enc.transform(C.iloc[te]))
            A, B = np.hstack(A), np.hstack(B)
            keep = A.std(0) > 0
            A, B = A[:, keep], B[:, keep]
            fcols = [f"f{i}" for i in range(A.shape[1])]
            dtr = pd.DataFrame(A, columns=fcols); dtr["T"], dtr["E"] = T[tr], E[tr]
            dte = pd.DataFrame(B, columns=fcols); dte["T"], dte["E"] = T[te], E[te]
            ok = np.ones(len(te), bool)
            if strata:
                dtr["type"], dte["type"] = types[tr], types[te]
                ok = np.isin(types[te], np.unique(types[tr]))   # unseen strata cannot be scored
                dte = dte[ok]
            cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.5)
            cph.fit(dtr, "T", "E", strata=["type"] if strata else None)
            risk = cph.predict_partial_hazard(dte).reindex(dte.index).values.ravel()
            Tt, Et, ty_t = T[te][ok], E[te][ok].astype(bool), types[te][ok]
            harrell = concordance_index_censored(Et, Tt, risk)[0]
            wc, wn = [], []
            for ty in np.unique(ty_t):
                m = ty_t == ty
                if Et[m].sum() >= 5:
                    wc.append(concordance_index_censored(Et[m], Tt[m], risk[m])[0]); wn.append(m.sum())
            # IPCW metrics need test times inside the training follow up range
            inr = Tt < tmax_tr
            s_te = Surv.from_arrays(Et[inr], Tt[inr])
            tmax = min(T[tr][E[tr] == 1].max(), Tt[inr].max()) - 1
            hz = horizons[horizons < tmax]
            try:
                uno = concordance_index_ipcw(s_tr, s_te, risk[inr], tau=min(1825, tmax))[0]
            except Exception:
                uno = np.nan
            try:
                tdauc, mean_auc = cumulative_dynamic_auc(s_tr, s_te, risk[inr], hz)
            except Exception:
                tdauc, mean_auc = [np.nan] * len(hz), np.nan
            try:
                # stratified lifelines models return columns grouped by stratum, so
                # realign the predictions to the row order of the test frame
                d_in = dte[inr]
                sf = cph.predict_survival_function(d_in, times=hz)[d_in.index].T.values
                ibs = integrated_brier_score(s_tr, s_te, sf, hz) if len(hz) > 1 else np.nan
            except Exception:
                ibs = np.nan
            r = dict(split=k, model=name, n_test=int(ok.sum()), harrell_c=harrell, uno_c=uno,
                     within_type_c=np.average(wc, weights=wn) if wc else np.nan,
                     mean_td_auc=mean_auc, ibs=ibs)
            for h, a in zip(hz, tdauc):
                r[f"td_auc_{int(h)}d"] = a
            rows.append(r)
            # 2 year calibration: predicted risk against Kaplan Meier observed risk by decile
            if 730 in hz:
                p2 = 1 - cph.predict_survival_function(dte, times=[730])[dte.index].values.ravel()
                dec = pd.qcut(p2, 10, labels=False, duplicates="drop")
                for d in np.unique(dec):
                    m = dec == d
                    km = KaplanMeierFitter().fit(Tt[m], Et[m])
                    calib.append(dict(split=k, model=name, decile=int(d), predicted_2y_risk=p2[m].mean(),
                                      observed_2y_risk=1 - float(km.predict(730))))
        log(f"survival split {k + 1}/{folds * repeats}")
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/R2_1_9_survival_cv_raw.csv", index=False)
    summarize(res, "model", [c for c in res.columns if c not in ("split", "model")]).to_csv(
        f"{out}/R2_1_9_survival_cv_summary.csv", index=False)
    cal = pd.DataFrame(calib).groupby(["model", "decile"]).mean(numeric_only=True).reset_index()
    cal.to_csv(f"{out}/R2_9_calibration_2y.csv", index=False)
    # calibration slope of observed on predicted 2 year risk across deciles
    cs = []
    for mname, g in cal.groupby("model"):
        sl, ic = np.polyfit(g["predicted_2y_risk"], g["observed_2y_risk"], 1)
        cs.append(dict(model=mname, slope_observed_on_predicted=sl, intercept=ic,
                       mean_abs_error=float((g["predicted_2y_risk"] - g["observed_2y_risk"]).abs().mean())))
    pd.DataFrame(cs).to_csv(f"{out}/R2_9_calibration_2y_summary.csv", index=False)

    # Proportional hazards diagnostics on a compact model stratified by tumor type
    tx = unsup_tx().fit(X)
    Z = tx.transform(X)[:, :10]
    enc = ClinEncoder(["age", "sex", "stage"]).fit(C)
    A = np.hstack([Z, enc.transform(C)])
    names = [f"PC{i + 1}" for i in range(10)] + enc.names()
    d = pd.DataFrame(A, columns=names).loc[:, lambda x: x.std() > 0]
    d["T"], d["E"], d["type"] = T, E, types
    cph = CoxPHFitter(penalizer=0.01).fit(d, "T", "E", strata=["type"])
    ph = proportional_hazard_test(cph, d, time_transform="rank")
    ph.summary.to_csv(f"{out}/R2_9_schoenfeld_tests.csv")
    return res


# Selection bias (R2.3)

def ipw_sensitivity(clin_cohort, y_all, X, C, y, types, out, n_pc=150):
    """Inverse probability of retention weighting. P(retained | type, year,
    stage, age, site) is modelled on all patients with expression and the
    held out AUC is reweighted so that it represents the full cohort."""
    retained = ~np.isnan(y_all)
    Cf = clin_frame(clin_cohort)
    enc = ClinEncoder(["type", "stage", "site", "age", "year"]).fit(Cf)
    pm = LogisticRegression(C=1.0, max_iter=5000).fit(enc.transform(Cf), retained)
    p_ret = pm.predict_proba(enc.transform(Cf))[:, 1]
    w = (1 / np.clip(p_ret, 0.02, 1))[retained]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    preds, _ = fit_predict_sets(X[tr], X[te], C.iloc[tr], C.iloc[te], y[tr], n_pc=n_pc)
    rows = []
    for k, p in preds.items():
        rows.append(dict(feature_set=k, auc_unweighted=roc_auc_score(y[te], p),
                         auc_ipw=roc_auc_score(y[te], p, sample_weight=w[te]),
                         within_type_unweighted=within_type_auc(y[te], p, types[te])[0]))
    pd.DataFrame(rows).to_csv(f"{out}/R2_3_ipw_sensitivity.csv", index=False)


def exclusion_characteristics(clin_full, thr, out):
    """Compares excluded and retained patients on the full patient level
    clinical table (Section 8.1)."""
    y = make_labels(clin_full, "global", thr)
    d = clin_full.copy()
    d["group"] = np.where(np.isnan(y), "excluded", "retained")
    d["stage_g"] = d[C_STAGE].map(stage_group)
    if C_CSTAGE in d:
        m = d["stage_g"].eq("NA")
        d.loc[m, "stage_g"] = d.loc[m, C_CSTAGE].map(stage_group)
    d["site"] = d[C_ID].str[5:7]
    ex, re_ = d[d.group == "excluded"], d[d.group == "retained"]
    rows = [dict(item="n", excluded=len(ex), retained=len(re_), p=np.nan)]
    for col, lab in ((C_YEAR, "median year of diagnosis"), (C_AGE, "median age"),
                     (C_OST, "median follow up (days)")):
        a, b = ex[col].dropna(), re_[col].dropna()
        rows.append(dict(item=lab, excluded=a.median(), retained=b.median(),
                         p=stats.mannwhitneyu(a, b)[1]))
    st = d[d.stage_g != "NA"]
    tab = pd.crosstab(st.group, st.stage_g)
    pct = tab.div(tab.sum(1), axis=0) * 100
    for g in ("I", "II", "III", "IV"):
        rows.append(dict(item=f"stage {g} percent (known stage)", excluded=pct.loc["excluded", g],
                         retained=pct.loc["retained", g], p=stats.chi2_contingency(tab)[1]))
    sx = pd.crosstab(d.group, d[C_SEX])
    rows.append(dict(item="female percent", excluded=100 * sx.loc["excluded", "FEMALE"] / sx.loc["excluded"].sum(),
                     retained=100 * sx.loc["retained", "FEMALE"] / sx.loc["retained"].sum(),
                     p=stats.chi2_contingency(sx)[1]))
    bt = pd.crosstab(d[C_TYPE], d.group)
    bt["pct_excluded"] = 100 * bt["excluded"] / bt.sum(1)
    rows.append(dict(item="exclusion percent by tumor type, min and max",
                     excluded=bt.pct_excluded.min(), retained=bt.pct_excluded.max(),
                     p=stats.chi2_contingency(bt[["excluded", "retained"]])[1]))
    bs = pd.crosstab(d.site, d.group)
    bs = bs[bs.sum(1) >= 50]
    bs["pct_excluded"] = 100 * bs["excluded"] / bs[["excluded", "retained"]].sum(1)
    rows.append(dict(item=f"exclusion percent by site ({len(bs)} sites with n >= 50), min and max",
                     excluded=bs.pct_excluded.min(), retained=bs.pct_excluded.max(),
                     p=stats.chi2_contingency(bs[["excluded", "retained"]])[1]))
    rows.append(dict(item="overall exclusion percent", excluded=100 * len(ex) / len(d), retained=np.nan, p=np.nan))
    pd.DataFrame(rows).to_csv(f"{out}/R2_3_retained_vs_excluded.csv", index=False)
    bt.sort_values("pct_excluded", ascending=False).to_csv(f"{out}/R2_3_exclusion_by_type.csv")
    return d


# Candidate genes (R2.10, R1.7)

def gene_adjusted(Xdf, C, T, E, types, genes, out, tr_idx, te_idx, genomewide=False):
    """For each gene, a Cox model stratified by tumor type and adjusted for age,
    sex and stage, fitted to all patients with survival data. Expression is
    used both per standard deviation and as a binary split at the median of
    each tumor type, so that the split cannot reflect differences in
    expression level between tumor types. Train and test partitions are
    reported separately; Benjamini Hochberg correction is applied within each
    partition across the five genes."""
    from lifelines import CoxPHFitter
    enc = ClinEncoder(["age", "sex", "stage"]).fit(C)
    base = pd.DataFrame(enc.transform(C), index=C.index, columns=enc.names())
    base = base.loc[:, base.std() > 0]

    def fit_gene(g, idx, binary=False):
        d = base.iloc[idx].copy()
        v = pd.Series(Xdf[g].values[idx])
        v = v.fillna(v.mean())
        ty = pd.Series(types[idx])
        if binary:
            med = v.groupby(ty.values).transform("median")
            d["gene"] = (v.values > med.values).astype(float)
        else:
            d["gene"] = ((v - v.mean()) / (v.std() + 1e-9)).values
        d["T"], d["E"], d["type"] = T[idx], E[idx], types[idx]
        cph = CoxPHFitter(penalizer=1e-4).fit(d, "T", "E", strata=["type"])
        s = cph.summary.loc["gene"]
        return (float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]),
                float(s["exp(coef) upper 95%"]), float(s["p"]))

    def bh(p):
        p = np.asarray(p); n = len(p); o = np.argsort(p)
        q = p[o] * n / np.arange(1, n + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        r = np.empty(n); r[o] = np.minimum(q, 1); return r

    rows = []
    for g in [g for g in genes if g in Xdf.columns]:
        for part, idx in (("train", tr_idx), ("test", te_idx), ("all", np.arange(len(T)))):
            for coding in ("per_SD", "above_type_median"):
                hr, lo, hi, p = fit_gene(g, idx, binary=coding != "per_SD")
                rows.append(dict(gene=g, partition=part, coding=coding, HR=hr, ci_lo=lo, ci_hi=hi, p=p))
    df = pd.DataFrame(rows)
    for (part, coding), idx in df.groupby(["partition", "coding"]).groups.items():
        df.loc[idx, "p_BH"] = bh(df.loc[idx, "p"])
    df.to_csv(f"{out}/R2_10_genes_adjusted_stratified_cox.csv", index=False)

    if genomewide:      # slow: one stratified Cox model per gene on the training partition
        var = Xdf.iloc[tr_idx].var()
        cand = var[var >= 0.5].index
        log(f"genome wide adjusted screen over {len(cand)} genes")
        res = []
        for i, g in enumerate(cand):
            try:
                res.append((g, *fit_gene(g, tr_idx)))
            except Exception:
                pass
            if i % 1000 == 0:
                log(f"  {i}/{len(cand)}")
        gw = pd.DataFrame(res, columns=["gene", "HR_per_SD", "ci_lo", "ci_hi", "p"])
        gw["p_BH"] = bh(gw["p"])
        top = gw.nsmallest(20, "p").copy()
        conf = [fit_gene(g, te_idx) for g in top.gene]
        top["test_HR"] = [c[0] for c in conf]
        top["test_p"] = [c[3] for c in conf]
        top["test_p_BH"] = bh(top["test_p"])
        gw.to_csv(f"{out}/R2_10_genomewide_train.csv", index=False)
        top.to_csv(f"{out}/R2_10_genomewide_top20_test_confirmation.csv", index=False)
    return df


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True)
    ap.add_argument("--clin", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=20, help="repeats of five fold CV, main analysis")
    ap.add_argument("--sens-repeats", type=int, default=3, help="repeats for the outcome sensitivity analysis")
    ap.add_argument("--surv-repeats", type=int, default=2, help="repeats of five fold CV, Cox models")
    ap.add_argument("--n-pc", type=int, default=150)
    ap.add_argument("--skip-survival", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--genomewide", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(f"{a.out}/software_versions.txt", "w") as f:
        f.write(f"python {sys.version.split()[0]} ({platform.platform()})\n")
        for mod in ("numpy", "pandas", "scipy", "sklearn", "lifelines", "sksurv"):
            try:
                f.write(f"{mod} {__import__(mod).__version__}\n")
            except Exception:
                pass
        f.write(json.dumps(vars(a)) + "\n")

    Xdf, clin, clin_full = build_cohort(a.expr, a.clin, a.out)
    C_all = clin_frame(clin)
    types_all = C_all["type"].values
    T_all, E_all = clin[C_OST].values.astype(float), clin[C_OS].values.astype(int)

    thr = global_thresholds(clin_full)
    y_full = make_labels(clin_full, "global", thr)
    y_all = make_labels(clin, "global", thr)
    lab = ~np.isnan(y_all)
    X, C, y, types = Xdf.values[lab], C_all[lab], y_all[lab].astype(int), types_all[lab]
    summ = {"thresholds_days": thr,
            "clinical_table": {"patients_complete_OS": len(clin_full),
                               "high": int(np.nansum(y_full == 1)), "low": int(np.nansum(y_full == 0)),
                               "low_died": int(((y_full == 0) & (clin_full[C_OS].values == 1)).sum()),
                               "low_censored": int(((y_full == 0) & (clin_full[C_OS].values == 0)).sum()),
                               "excluded": int(np.isnan(y_full).sum())},
            "expression_cohort": {"patients": len(clin), "labelled": int(lab.sum()),
                                  "excluded": int((~lab).sum()), "high": int(y.sum())},
            "high_risk_frac": float(y.mean()),
            "n_types_modelling": int(len(np.unique(types)))}
    flow = pd.read_csv(f"{a.out}/R2_4_patient_flow.csv")
    flow = pd.concat([flow, pd.DataFrame(
        [("Labelled with the global rule (modelling cohort)", int(lab.sum())),
         ("  of which high risk", int(y.sum())),
         ("Excluded: censored at or before the 75th percentile", int((~lab).sum()))],
        columns=["step", "n"])])
    flow.to_csv(f"{a.out}/R2_4_patient_flow.csv", index=False)
    log(summ)
    exclusion_characteristics(clin_full, thr, a.out)

    # Original single split, kept for continuity with the submitted manuscript
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    preds, tx = fit_predict_sets(X[tr], X[te], C.iloc[tr], C.iloc[te], y[tr], n_pc=a.n_pc,
                                 other_models=True)
    summ["single_split"] = {"n_train": len(tr), "n_test": len(te),
                            "n_selected_components": int(tx.named_steps["l1"].get_support().sum()),
                            "n_genes_after_variance_filter": int(tx.named_steps["var"].get_support().sum())}
    cumvar = np.cumsum(tx.named_steps["pca"].explained_variance_ratio_) * 100
    summ["single_split"]["pca_cumvar_pct"] = {k: round(float(cumvar[k - 1]), 1)
                                              for k in (10, 50, 100, a.n_pc) if k <= a.n_pc}
    # Hotelling T2 and Pillai trace on all training PCs of the transcriptome only block
    P = tx[:-1].transform(X[tr]); ytr_ = y[tr]
    A1, A0 = P[ytr_ == 1], P[ytr_ == 0]; n1, n0, k = len(A1), len(A0), P.shape[1]
    Sp = ((n1 - 1) * np.cov(A1.T) + (n0 - 1) * np.cov(A0.T)) / (n1 + n0 - 2)
    dlt = A1.mean(0) - A0.mean(0)
    T2 = float(n1 * n0 / (n1 + n0) * dlt @ np.linalg.solve(Sp, dlt))
    F = T2 * (n1 + n0 - k - 1) / (k * (n1 + n0 - 2))
    summ["hotelling"] = dict(T2=T2, F=F, df1=k, df2=n1 + n0 - k - 1,
                             p=float(stats.f.sf(F, k, n1 + n0 - k - 1)),
                             pillai=float(T2 / (T2 + n1 + n0 - 2)))
    rows = []
    for k, p in preds.items():
        wt, per = within_type_auc(y[te], p, types[te], min_n=20)
        rows.append(dict(feature_set=k, pooled_auc=roc_auc_score(y[te], p), within_type_auc=wt,
                         n_types=len(per)))
        if "|" not in k:
            pd.DataFrame(per, columns=["type", "n_test", "n_high", "auc"]).to_csv(
                f"{a.out}/per_type_auc_{k}.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{a.out}/R2_2_original_split_models.csv", index=False)
    pd.DataFrame({"y": y[te], "type": types[te], **{k: v for k, v in preds.items()}}).to_csv(
        f"{a.out}/single_split_test_predictions.csv", index=False)
    per_type_accounting(y, types_all, types, te, preds, a.out)
    dl = []
    for p1, p2 in [("transcriptome_only", "clinical_only"), ("combined", "clinical_only"),
                   ("combined", "transcriptome_only"), ("transcriptome_only", "tumor_type_only"),
                   ("transcriptome_plus_tumor_type", "tumor_type_only")]:
        a1, a2, d, z, p = delong(y[te], preds[p1], preds[p2])
        dl.append(dict(comparison=f"{p1} vs {p2}", auc1=a1, auc2=a2, diff=d, z=z, p=p))
    pd.DataFrame(dl).to_csv(f"{a.out}/R2_8_delong_original_split.csv", index=False)
    Ztr, Zte = tx.transform(X[tr]), tx.transform(X[te])
    encoding_recovery(Ztr, Zte, C.iloc[tr], C.iloc[te], a.out)
    log("single split done")

    # Repeated stratified five fold CV
    log("repeated CV")
    res, diffs, per_type, nsel = repeated_cv(X, C, y, types, repeats=a.repeats, n_pc=a.n_pc,
                                             tag="main", other_models=True)
    res.to_csv(f"{a.out}/R2_5_repeated_cv_raw.csv", index=False)
    summarize(res, "feature_set", ["pooled_auc", "within_type_auc", "pooling_gap", "brier",
                                   "cal_slope", "cal_intercept", "ece"]).to_csv(
        f"{a.out}/R2_5_repeated_cv_summary.csv", index=False)
    ds = summarize(diffs, "comparison", ["d_pooled", "d_within"])
    pos = diffs.groupby("comparison").agg(frac_splits_positive_pooled=("d_pooled", lambda s: (s > 0).mean()),
                                          frac_splits_positive_within=("d_within", lambda s: (s > 0).mean()))
    ds.merge(pos.reset_index(), on="comparison").to_csv(f"{a.out}/R2_8_paired_differences_cv.csv", index=False)
    per_type.to_csv(f"{a.out}/R1_4_per_type_auc_cv_raw.csv", index=False)
    pt = per_type.groupby(["feature_set", "type"]).agg(
        n_splits=("auc", "size"), mean_n_test=("n_test", "mean"), auc_mean=("auc", "mean"),
        auc_lo=("auc", lambda s: s.quantile(0.025)), auc_hi=("auc", lambda s: s.quantile(0.975)))
    pt.reset_index().to_csv(f"{a.out}/R1_4_per_type_auc_cv_summary.csv", index=False)
    summ["cv_selected_components"] = {"median": float(nsel.n_components.median()),
                                      "min": int(nsel.n_components.min()), "max": int(nsel.n_components.max()),
                                      "median_genes": float(nsel.n_genes.median())}

    # Outcome definition sensitivity (R2.1, R1.2)
    if not a.skip_sensitivity:
        log("threshold sensitivity")
        sens_sets = {k: FEATURE_SETS[k] for k in ("transcriptome_only", "clinical_only", "combined")}
        srows = []
        for name, mode, t in [("global_median_p75", "global", thr), ("per_type_median_p75", "per_type", None),
                              ("landmark_at_global_median", "landmark", thr[0]),
                              ("landmark_1y", "landmark", 365), ("landmark_2y", "landmark", 730),
                              ("landmark_3y", "landmark", 1095)]:
            ys = make_labels(clin, mode, t, clin_full)
            m = ~np.isnan(ys)
            r, _, _, _ = repeated_cv(Xdf.values[m], C_all[m], ys[m].astype(int), types_all[m],
                                     repeats=a.sens_repeats, sets=sens_sets, n_pc=a.n_pc, tag=name)
            s = summarize(r, "feature_set", ["pooled_auc", "within_type_auc", "pooling_gap"])
            s.insert(0, "labelling", name); s.insert(1, "n", int(m.sum())); s.insert(2, "excluded", int((~m).sum()))
            s.insert(3, "high_risk_frac", float(ys[m].mean()))
            srows.append(s)
        pd.concat(srows).to_csv(f"{a.out}/R2_1_threshold_sensitivity.csv", index=False)

    log("negative controls")
    permuted_within_type(X, C, y, types).to_csv(f"{a.out}/control_within_type_permutation.csv", index=False)
    leave_one_type_out(X, C, y, types).to_csv(f"{a.out}/control_leave_one_type_out.csv", index=False)

    log("selection bias sensitivity")
    ipw_sensitivity(clin, y_all, X, C, y, types, a.out, n_pc=a.n_pc)

    log("cohort effects")
    Z_all = np.vstack([Ztr, Zte])
    C_order = pd.concat([C.iloc[tr], C.iloc[te]]).reset_index(drop=True)
    cohort_effects(Z_all, C_order, a.out)

    if not a.skip_survival:
        log("survival analyses on all patients")
        survival_analyses(Xdf.values, C_all.reset_index(drop=True), T_all, E_all, types_all, a.out,
                          n_pc=a.n_pc, repeats=a.surv_repeats)

    log("gene level adjusted analysis")
    tr_all, te_all = train_test_split(np.arange(len(T_all)), test_size=0.2,
                                      stratify=E_all, random_state=SEED)
    gene_adjusted(Xdf, C_all, T_all, E_all, types_all, TOP_GENES, a.out, tr_all, te_all, a.genomewide)

    with open(f"{a.out}/summary.txt", "w") as f:
        f.write(json.dumps(summ, indent=2) + "\n")
        for fn in sorted(os.listdir(a.out)):
            if fn.endswith(".csv") and "raw" not in fn and not fn.startswith("per_type") \
                    and fn != "single_split_test_predictions.csv":
                f.write(f"\n== {fn}\n" + pd.read_csv(f"{a.out}/{fn}").round(4).to_string(index=False) + "\n")
    log("done", a.out)


if __name__ == "__main__":
    main()
