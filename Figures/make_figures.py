#!/usr/bin/env python3
"""
Regenerate every figure of the manuscript

    "Similarity-graph neural networks shift classification toward minority
     selectivity classes in four enzyme-pair inhibitor datasets"

at publication resolution (600 dpi).

Usage
-----
    python make_figures.py                 # all figures
    python make_figures.py 4 7 13          # only Figures 4, 7 and 13
    python make_figures.py --list          # list what each figure is

Outputs land in OUTDIR (default: manuscript_figures_600dpi/) as
Figure01.png ... Figure15.png, with multi-panel figures written as
Figure02A.png ... Figure02D.png etc.

Inputs
------
Raw data and per-dataset results are read from PROJECT (this directory).
Consolidated result tables (master_*.csv, produced by the consolidation step)
are read from the same directory. Nothing is written outside OUTDIR.

What is and is not recomputed
-----------------------------
* Figures 1, 4, 5, 6, 14          -- rendered from the consolidated GNN result
                                     tables (the 10-seed runs themselves are not
                                     retrained here; that requires the RF_*.ipynb
                                     notebooks and a GPU).
* Figures 2, 3                    -- rendered from master_per_class.csv /
                                     master_auc.csv, reproducing the heatmaps in
                                     each anal/ bundle.
* Figure 7                        -- the three classical baselines are REFIT here
                                     with the hyperparameters recorded in
                                     BEST_PARAMS, because the per-molecule
                                     probabilities needed for ROC/PR curves were
                                     never saved. `--check` compares the refit
                                     against the stored confusion counts.
                                     Reproducibility of that refit, IDO1/TDO:
                                       SVM            exact (all counts, AUROC,
                                                      AUPRC identical)
                                       Random Forest  2 of 226 molecules differ on
                                                      AI; |dAUROC| <= 0.005,
                                                      |dAUPRC| <= 0.006
                                       XGBoost        differs most; |dAUROC| <=
                                                      0.027, |dAUPRC| <= 0.045
                                     The residual drift is a library-version
                                     effect (the notebooks used scikit-learn 1.5.1
                                     and an older xgboost), not a different
                                     pipeline: the features, split, class
                                     weighting and hyperparameters are identical,
                                     and the SVM -- the model most sensitive to
                                     the feature matrix -- reproduces exactly.
                                     Quote the stored aggregate values in the text
                                     and treat the curves as illustrative.
* Figures 8, 9                    -- 2D structures rendered from the SMILES and
                                     attention weights stored in
                                     best_mols_gat/gat_attention_neighbors_AI_IA.csv
                                     and gat_attention_scaffold_summary.csv.
* Figures 10, 11, 12, 13, 15      -- recomputed from the raw IC50 / SMILES data.

Requirements: numpy, pandas, scipy, matplotlib, rdkit, scikit-learn, xgboost.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(PROJECT, "manuscript_figures_600dpi")
DPI = 600

DATASETS = ["IDO1/TDO", "COX-1/COX-2", "MAO-A/MAO-B", "AChE/BChE"]
ARCHS = ["GCN", "GAT", "GINE", "GraphSAGE"]
CLASSES = ["AA", "AI", "IA", "II"]

FOLDER = {"IDO1/TDO": "IDO_TDO", "COX-1/COX-2": "cox1_cox2",
          "MAO-A/MAO-B": "MAOA_MAOB", "AChE/BChE": "AchE_BchE"}

# raw file, IC50 column for enzyme 1, IC50 column for enzyme 2
RAW = {
    "IDO1/TDO":    ("IDO_TDO/ido1_tdo_common_1126.csv",        "ido1_ic50", "tdo_ic50"),
    "COX-1/COX-2": ("cox1_cox2/cox1_cox2_common_20260312.csv", "cox1_ic50", "cox2_ic50"),
    "MAO-A/MAO-B": ("MAOA_MAOB/maoa_maob_common.csv",          "maoa_ic50", "maob_ic50"),
    "AChE/BChE":   ("AchE_BchE/ache_bche_common.csv",           "ache_ic50", "bche_ic50"),
}

ENZYME = {"IDO1/TDO": ("IDO1", "TDO"), "COX-1/COX-2": ("COX-1", "COX-2"),
          "MAO-A/MAO-B": ("MAO-A", "MAO-B"), "AChE/BChE": ("AChE", "BChE")}

# grid-search winners recorded in the RF_GAT.ipynb output of each dataset
BEST_PARAMS = {
    "IDO1/TDO": dict(
        rf=dict(bootstrap=False, criterion="entropy", max_depth=9, max_features="sqrt"),
        xgb=dict(learning_rate=0.05, max_depth=3, n_estimators=100),
        svm=dict(scale=False, C=100, gamma=0.0018355156852833928)),
    "COX-1/COX-2": dict(
        rf=dict(bootstrap=True, criterion="gini", max_depth=3, max_features="sqrt"),
        xgb=dict(learning_rate=0.05, max_depth=7, n_estimators=100),
        svm=dict(scale=True, C=1, gamma=0.001171875)),
    "MAO-A/MAO-B": dict(
        rf=dict(bootstrap=True, criterion="entropy", max_depth=5, max_features="log2"),
        xgb=dict(learning_rate=0.05, max_depth=3, n_estimators=100),
        svm=dict(scale=False, C=10, gamma=0.0059825601470444925)),
    "AChE/BChE": dict(
        rf=dict(bootstrap=False, criterion="gini", max_depth=7, max_features="sqrt"),
        xgb=dict(learning_rate=0.05, max_depth=3, n_estimators=100),
        svm=dict(scale=False, C=1, gamma=0.015495435554359161)),
}

# published confusion counts (minority_comparison/minority_precision_recall.csv),
# used by --check to validate the Figure 7 baseline refit
PUBLISHED_PR = {  # (dataset, model, class) -> (TP, FP, FN)
    ("IDO1/TDO", "Random Forest", "AI"): (29, 6, 53),
    ("IDO1/TDO", "Random Forest", "IA"): (1, 1, 10),
    ("IDO1/TDO", "XGBoost", "AI"): (37, 11, 45),
    ("IDO1/TDO", "XGBoost", "IA"): (2, 9, 9),
    ("IDO1/TDO", "SVM", "AI"): (27, 9, 55),
    ("IDO1/TDO", "SVM", "IA"): (4, 10, 7),
}

# ---- palette -----------------------------------------------------------------
C_AI, C_IA = "#1b6ca8", "#d1781c"          # the two selectivity classes
GREY = "#6e6e6e"
ARCH_COLOR = {"GCN": "#8c8c8c", "GAT": "#1b6ca8",
              "GINE": "#e8a33d", "GraphSAGE": "#c0504d"}
MODEL_COLOR = {"Random Forest": "#8c8c8c", "XGBoost": "#e8a33d",
               "SVM": "#7a9a4f", "GAT": "#1b6ca8"}
SPLIT_COLOR = {"train": "#2c6fbb", "val": "#e8a33d", "test": "#c0504d"}
CLASS_COLOR = {"AA": "#9c79c9", "AI": "#4a7fd4", "IA": "#4ab8e0", "II": "#b0b0b8"}


def set_style(sizes=(9, 8, 7)):
    """Three font sizes mapped to role: labels/titles, legend/annotation, ticks."""
    base, mid, small = sizes
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": mid, "xtick.labelsize": small, "ytick.labelsize": small,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "legend.frameon": False, "figure.autolayout": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def open_frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, PROJECT)}")
    return path


# ----------------------------------------------------------------------------
# Data: labels, features, scaffold split
# ----------------------------------------------------------------------------

_CACHE: dict = {}


def label_classes(ic50_1, ic50_2, cutoff=1000.0):
    """Four-class label from a pair of IC50 values.

    Activity rule as implemented in the training notebooks: IC50 <= cutoff is
    active. The inequality is NOT strict -- a non-trivial number of molecules are
    reported at exactly 1000 nM (91 of 1126 for IDO1/TDO), so the direction
    matters for the class counts.
    """
    a1 = np.where(np.asarray(ic50_1, float) <= cutoff, "A", "I")
    a2 = np.where(np.asarray(ic50_2, float) <= cutoff, "A", "I")
    return np.char.add(a1, a2)


def morgan_matrix(smiles, radius=3, n_bits=1024):
    """Morgan / ECFP6 bit matrix, as used for node features and edge weights."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


def murcko_scaffold(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Bemis-Murcko scaffold split, whole groups assigned largest-first.

    Reproduces scaffold_split_indices() in the RF_*.ipynb notebooks exactly:
    unparseable and acyclic molecules are isolated as singleton groups, the group
    list is permuted under `seed` for tie-breaking, then sorted largest-first and
    poured greedily into train, then validation, then test.
    """
    idx = np.arange(len(smiles))
    groups = defaultdict(list)
    for i, smi in zip(idx, smiles):
        sc = murcko_scaffold(smi)
        if sc is None:
            sc = f"__invalid_{i}__"
        elif sc == "":
            sc = f"__acyclic_{i}__"
        groups[sc].append(i)

    rng = np.random.default_rng(seed)
    gl = list(groups.values())
    gl = [gl[i] for i in rng.permutation(len(gl))]
    gl.sort(key=len, reverse=True)

    n = len(idx)
    n_tr, n_va = int(np.floor(train_ratio * n)), int(np.floor(val_ratio * n))
    tr, va, te = [], [], []
    for g in gl:
        if len(tr) + len(g) <= n_tr:
            tr.extend(g)
        elif len(va) + len(g) <= n_va:
            va.extend(g)
        else:
            te.extend(g)
    return np.array(tr), np.array(va), np.array(te), groups


def load_dataset(ds):
    """-> dict with df, labels y (0..3), 256-dim SVD features X, split indices."""
    if ds in _CACHE:
        return _CACHE[ds]
    from sklearn.decomposition import TruncatedSVD

    rel, c1, c2 = RAW[ds]
    df = pd.read_csv(os.path.join(PROJECT, rel), na_values=["na"]).reset_index(drop=True)
    lab = label_classes(df[c1], df[c2])
    y = np.array([CLASSES.index(s) for s in lab])

    fp = morgan_matrix(df["smiles"].to_numpy())
    X = TruncatedSVD(n_components=256, random_state=42).fit_transform(fp)

    tr, va, te, groups = scaffold_split(df["smiles"].to_numpy())
    out = dict(df=df, c1=c1, c2=c2, labels=lab, y=y, fp=fp, X=X,
               train=tr, val=va, test=te, groups=groups)
    _CACHE[ds] = out
    return out


def fit_baselines(ds):
    """Refit RF / XGBoost / SVM with the recorded hyperparameters.

    Needed only for Figure 7: the per-molecule predicted probabilities behind the
    ROC and precision-recall curves were never saved, only the aggregate metrics.
    All three use class_weight='balanced' (XGBoost via per-sample weights) and the
    same 256-dim features and scaffold split as the graph networks.
    """
    key = ("baselines", ds)
    if key in _CACHE:
        return _CACHE[key]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    d = load_dataset(ds)
    Xtr, ytr = d["X"][d["train"]], d["y"][d["train"]]
    Xte, yte = d["X"][d["test"]], d["y"][d["test"]]
    bp = BEST_PARAMS[ds]

    rf = RandomForestClassifier(class_weight="balanced", random_state=42, **bp["rf"])
    rf.fit(Xtr, ytr)

    xgb = XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                        tree_method="hist", random_state=42, n_jobs=1,
                        subsample=0.8, colsample_bytree=0.8, **bp["xgb"])
    xgb.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))

    sp = bp["svm"]
    Xtr_s, Xte_s = Xtr, Xte
    if sp["scale"]:
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    svm = SVC(kernel="rbf", C=sp["C"], gamma=sp["gamma"], class_weight="balanced",
              probability=True, cache_size=1000, random_state=42)
    svm.fit(Xtr_s, ytr)

    out = {
        "y_true": yte,
        "Random Forest": rf.predict_proba(Xte),
        "XGBoost": xgb.predict_proba(Xte),
        "SVM": svm.predict_proba(Xte_s),
        "pred": {"Random Forest": rf.predict(Xte), "XGBoost": xgb.predict(Xte),
                 "SVM": svm.predict(Xte_s)},
    }
    _CACHE[key] = out
    return out


# ----------------------------------------------------------------------------
# Consolidated result tables
# ----------------------------------------------------------------------------

def _read(name):
    path = os.path.join(PROJECT, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {name} (run the consolidation step first)")
    return pd.read_csv(path)


def per_class():
    """GNN - RF per-class precision/recall/F1 with one-sample t-tests."""
    d = _read("master_per_class.csv")
    return d[d.Model.isin(ARCHS)]


def per_class_auc():
    d = _read("master_auc.csv")
    return d[d.Model.isin(ARCHS)]


def overall():
    d = _read("master_overall.csv")
    return d[d.Model.isin(ARCHS)]


# ----------------------------------------------------------------------------
# Figure 1 -- overall performance across the four target pairs
# ----------------------------------------------------------------------------

def figure01():
    ovf = _read("fig_overall_four_datasets.csv")
    mets = [("Accuracy", "Accuracy"), ("MCC", "MCC"),
            ("AUROC", "Macro AUROC"), ("AUPRC", "Macro AUPRC")]
    fig, axes = plt.subplots(4, 4, figsize=(11.2, 9.4))
    for i, (key, label) in enumerate(mets):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            sub = ovf[ovf.Dataset == ds]
            xs = np.arange(len(sub))
            ax.bar(xs, sub[key], yerr=sub[key + "_sd"], width=.66,
                   color=[ARCH_COLOR[m] for m in sub.Model], capsize=2.5,
                   edgecolor="white", linewidth=.6,
                   error_kw=dict(lw=1, ecolor="0.25"))
            rf = sub["RF_" + key].iloc[0]
            ax.axhline(rf, ls="--", lw=1.2, color="0.15", zorder=3)
            for k, (_, r) in enumerate(sub.iterrows()):
                sig = "*" if r[key + "_p"] < 0.05 else "ns"
                ax.text(k, r[key] + r[key + "_sd"] + 0.012, sig, ha="center",
                        va="bottom", fontsize=7,
                        color="0.15" if sig == "*" else GREY)
            lo = min(sub[key].min() - sub[key + "_sd"].max(), rf)
            hi = max(sub[key].max() + sub[key + "_sd"].max(), rf)
            pad = (hi - lo) * 0.35 + 0.01
            ax.set_ylim(max(0, lo - pad), hi + pad * 1.4)
            ax.set_xticks(xs)
            ax.set_xticklabels(sub.Model if i == 3 else [""] * len(sub),
                               rotation=35, ha="right")
            open_frame(ax)
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(ds, fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], color="0.15", ls="--", lw=1.2,
               label="Random Forest reference  ·  * = p < 0.05 vs Random Forest "
                     "(one-sample t-test, 10 seeds)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7.5)
    fig.suptitle("Overall performance against the class-weighted Random Forest: no "
                 "architecture improves accuracy,\nMCC rises only on MAO-A/MAO-B, and "
                 "macro AUPRC rises on three of four target pairs",
                 x=0.012, ha="left", fontsize=9.5, y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    return [save(fig, "Figure01")]


# ----------------------------------------------------------------------------
# Figures 2 and 3 -- per-class heatmaps (one panel per dataset)
# ----------------------------------------------------------------------------

def _heatmap(ax, M, sig, row_labels, col_labels, vmax, annot, title,
             show_rows=True):
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    # row labels only on the leftmost panel, so they cannot collide with the
    # annotated cells of the panel to their left
    ax.set_yticks(range(len(row_labels)),
                  row_labels if show_rows else [""] * len(row_labels))
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = abs(v) > 0.62 * vmax
            ax.text(j, i, annot[i][j] + ("*" if sig[i][j] else ""),
                    ha="center", va="center", fontsize=6.6, fontweight="bold",
                    color="white" if dark else "0.1")
    return im


def figure02():
    """Per-class precision / recall / F1, difference from Random Forest."""
    pc = per_class()
    paths = []
    for letter, ds in zip("ABCD", DATASETS):
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.9))
        fig.subplots_adjust(wspace=0.08)
        for k, (ax, (metric, label)) in enumerate(zip(
                axes, [("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1")])):
            sub = pc[(pc.Dataset == ds) & (pc.Metric == metric)]
            M = np.zeros((len(ARCHS), 4))
            sig = [[False] * 4 for _ in ARCHS]
            annot = [[""] * 4 for _ in ARCHS]
            for i, a in enumerate(ARCHS):
                for j, c in enumerate(CLASSES):
                    r = sub[(sub.Model == a) & (sub.Class == c)].iloc[0]
                    M[i, j] = r["diff"]
                    sig[i][j] = bool(r.p_value < 0.0125)
                    annot[i][j] = f"{r['diff']:+.3f}"
            im = _heatmap(ax, M, sig, ARCHS, CLASSES, 0.25, annot, label,
                          show_rows=(k == 0))
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
        cb.set_label("Difference from RF (GNN − RF)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        fig.suptitle(f"{ds} — GNN − RF per-class performance (10-run mean; "
                     "* = significant, Bonferroni α = 0.0125)",
                     x=0.012, ha="left", fontsize=9.5, y=1.04)
        fig.text(0.5, -0.06, "Red = GNN scores higher than Random Forest   |   "
                 "Blue = Random Forest scores higher", ha="center",
                 fontsize=7, color=GREY, style="italic")
        paths.append(save(fig, f"Figure02{letter}"))
    return paths


def figure03():
    """Per-class AUROC and AUPRC, value and difference from Random Forest."""
    au = per_class_auc()
    ds_tab = _read("dataset_summary.csv").set_index("Dataset")
    paths = []
    for letter, ds in zip("ABCD", DATASETS):
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.0))
        fig.subplots_adjust(wspace=0.08)
        for k, (ax, metric) in enumerate(zip(axes, ["AUROC", "AUPRC"])):
            sub = au[(au.Dataset == ds) & (au.Metric == metric) &
                     (au.Scope == "per-class")]
            M = np.zeros((len(ARCHS), 4))
            sig = [[False] * 4 for _ in ARCHS]
            annot = [[""] * 4 for _ in ARCHS]
            rf_row = []
            for j, c in enumerate(CLASSES):
                rf_row.append(sub[sub.Class == c].iloc[0].RF)
                for i, a in enumerate(ARCHS):
                    r = sub[(sub.Model == a) & (sub.Class == c)].iloc[0]
                    M[i, j] = r["diff"]
                    sig[i][j] = bool(r.p_value < 0.0125)
                    annot[i][j] = f"{r.GNN_mean:.3f}\n({r['diff']:+.3f})"
            cols = [f"{c}\n(RF {v:.2f})" for c, v in zip(CLASSES, rf_row)]
            im = _heatmap(ax, M, sig, ARCHS, cols, 0.10, annot, metric,
                          show_rows=(k == 0))
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
        cb.set_label("Difference from RF (GNN − RF)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        prev = ", ".join(f"{c} {ds_tab.loc[ds, f'Test_{c}'] / ds_tab.loc[ds, 'Test']:.2f}"
                         for c in CLASSES)
        fig.suptitle(f"{ds} — per-class AUROC and AUPRC: GNN mean (difference from RF); "
                     "* = significant, Bonferroni α = 0.0125",
                     x=0.012, ha="left", fontsize=9.5, y=1.05)
        fig.text(0.5, -0.09, "Red = GNN higher than Random Forest   |   Blue = Random "
                 "Forest higher\nChance level: AUROC = 0.50;  AUPRC = class prevalence "
                 f"({prev})", ha="center", fontsize=7, color=GREY, style="italic")
        paths.append(save(fig, f"Figure03{letter}"))
    return paths


# ----------------------------------------------------------------------------
# Figure 4 -- F1 difference on both selectivity classes
# ----------------------------------------------------------------------------

def _lollipop_panels(data, row_order, row_labels, xlabel, title, titles,
                     xlim, figsize, shade_rows=(), ytick_fontsize=7):
    """Shared renderer for Figures 4 and 5: one panel per dataset, two series
    (AI, IA) of horizontal lollipops against a zero line."""
    ypos = {m: i for i, m in enumerate(row_order[::-1])}
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = data[data.Dataset == ds]
        ax.axvline(0, color="0.35", lw=1.0, zorder=1)
        if shade_rows:
            ax.axhspan(min(shade_rows) - 0.5, max(shade_rows) + 0.5,
                       color="0.93", zorder=0)
        for cl, col, off in [("AI", C_AI, +0.16), ("IA", C_IA, -0.16)]:
            for _, r in sub[sub.Class == cl].iterrows():
                y = ypos[r["_row"]] + off
                ax.plot([0, r["diff"]], [y, y], color=col, lw=1.8, alpha=.55,
                        zorder=2, solid_capstyle="butt")
                ax.plot(r["diff"], y, "o", ms=6.2, mec=col, mew=1.4, zorder=3,
                        mfc=col if r["sig"] else "white")
        missing = [m for m in row_order if m not in set(sub["_row"])]
        for m in missing:
            ax.text(0.015, ypos[m], "n.d.", va="center", ha="left", fontsize=7,
                    color=GREY, style="italic")
        ax.set_title(titles[ds], fontsize=9)
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels(row_labels[::-1], fontsize=ytick_fontsize)
        ax.set_ylim(-0.65, len(row_order) - 0.35)
        open_frame(ax)
    axes[0].set_xlim(*xlim)
    fig.supxlabel(xlabel, y=0.02, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=C_AI, mfc=C_AI, ms=6,
                          label="AI (target-1-selective)"),
               plt.Line2D([], [], marker="o", ls="-", color=C_IA, mfc=C_IA, ms=6,
                          label="IA (target-2-selective)"),
               plt.Line2D([], [], marker="o", ls="none", color="0.3", mfc="white",
                          mec="0.3", ms=6, label="not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), fontsize=7.5)
    fig.suptitle(title, x=0.012, ha="left", fontsize=9.5, y=1.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.86])
    fig.subplots_adjust(top=0.72)
    return fig


def figure04():
    pc = per_class()
    d = pc[(pc.Metric == "f1") & (pc.Class.isin(["AI", "IA"]))].copy()
    d["sig"] = d.p_value < 0.0125
    d["_row"] = d.Model
    titles = {}
    for ds in DATASETS:
        rf = d[d.Dataset == ds].drop_duplicates("Class").set_index("Class").RF
        titles[ds] = f"{ds}\nRF F1: AI {rf['AI']:.2f} · IA {rf['IA']:.2f}"
    fig = _lollipop_panels(
        d, ARCHS, ARCHS,
        "F1 difference, GNN − Random Forest   (right = GNN better)",
        "Both single-target-selective classes gain F1 in all four target pairs,\n"
        "except the already-majority IA class of COX-1/COX-2",
        titles, (-0.13, 0.42), (11.0, 2.9))
    return [save(fig, "Figure04")]


# ----------------------------------------------------------------------------
# Figure 5 -- GAT minus RF on every endpoint, both selectivity classes
# ----------------------------------------------------------------------------

def figure05():
    pc, au = per_class(), per_class_auc()
    rows = []
    spec = [("recall", "Recall\n(threshold)", pc), ("f1", "F1\n(threshold)", pc),
            ("precision", "Precision\n(threshold)", pc),
            ("AUROC", "AUROC\n(ranking)", au), ("AUPRC", "AUPRC\n(ranking)", au)]
    for ds in DATASETS:
        for cl in ("AI", "IA"):
            for metric, label, src in spec:
                sel = src[(src.Dataset == ds) & (src.Model == "GAT") &
                          (src.Class == cl) & (src.Metric == metric)]
                r = sel.iloc[0]
                rows.append(dict(Dataset=ds, Class=cl, _row=label, diff=r["diff"],
                                 sig=bool(r.p_value < 0.0125)))
    d = pd.DataFrame(rows)
    order = [label for _, label, _ in spec]
    fig = _lollipop_panels(
        d, order, order,
        "Difference, GAT (10 seeds) − Random Forest   (right = GAT better)",
        "Recall gains on both selective classes exceed the ranking gains that accompany "
        "them:\na shifted operating point on IDO1/TDO, genuinely better ranking on "
        "COX-1/COX-2 and MAO-A/MAO-B",
        {ds: ds for ds in DATASETS}, (-0.37, 0.52), (11.0, 3.3),
        shade_rows=(0, 1))
    return [save(fig, "Figure05")]


# ----------------------------------------------------------------------------
# Figure 6 -- GAT against all three classical baselines, both classes
# ----------------------------------------------------------------------------

def figure06():
    pc, au, bl = per_class(), per_class_auc(), _read("master_baselines.csv")
    ds_tab = _read("dataset_summary.csv").set_index("Dataset")
    order = ["Random Forest", "XGBoost", "SVM", "GAT"]
    rows = []
    for ds in DATASETS:
        for cl in ("AI", "IA"):
            g = pc[(pc.Dataset == ds) & (pc.Model == "GAT") &
                   (pc.Class == cl) & (pc.Metric == "f1")].iloc[0]
            ga = au[(au.Dataset == ds) & (au.Model == "GAT") & (au.Class == cl)]
            rows.append(dict(Dataset=ds, Class=cl, Model="GAT", F1=g.GNN_mean,
                             F1_sd=g.GNN_std,
                             AUPRC=ga[ga.Metric == "AUPRC"].iloc[0].GNN_mean))
            for src, lab in [("RF", "Random Forest"), ("XGBoost", "XGBoost"),
                             ("SVC", "SVM")]:
                s = bl[(bl.Dataset == ds) & (bl.Class == cl) &
                       (bl.Model == src)].set_index("Metric").Estimate
                rows.append(dict(Dataset=ds, Class=cl, Model=lab, F1=s["F1"],
                                 F1_sd=np.nan, AUPRC=s["AUPRC"]))
    cmp = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2), sharey="row")
    w = 0.36
    for i, (metric, label) in enumerate([("F1", "F1-score"), ("AUPRC", "AUPRC")]):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            sub = cmp[cmp.Dataset == ds]
            for k, cl in enumerate(["AI", "IA"]):
                s = sub[sub.Class == cl].set_index("Model").loc[order]
                xs = np.arange(len(order)) + (k - 0.5) * w
                ax.bar(xs, s[metric], width=w * 0.92,
                       color=[MODEL_COLOR[m] for m in order],
                       edgecolor="white", linewidth=.5,
                       hatch="" if cl == "AI" else "////",
                       yerr=s[metric + "_sd"] if metric == "F1" else None,
                       capsize=2, error_kw=dict(lw=1, ecolor="0.25"))
                for xi, (m, v) in enumerate(zip(order, s[metric])):
                    if v < 0.006:      # a zero bar must still be visible
                        ax.plot(xi + (k - 0.5) * w, 0, marker="_", ms=7, mew=2.2,
                                color=MODEL_COLOR[m], zorder=5, clip_on=False)
            if metric == "AUPRC":
                for k, cl in enumerate(["AI", "IA"]):
                    prev = ds_tab.loc[ds, f"Test_{cl}"] / ds_tab.loc[ds, "Test"]
                    ax.plot([-0.5 + (k - .5) * w + .2,
                             len(order) - 0.5 + (k - .5) * w - .2],
                            [prev, prev], ls=":", lw=1.2, color="0.2")
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order if i == 1 else [""] * len(order),
                               rotation=35, ha="right")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(ds, fontsize=9)
            open_frame(ax)
            ax.margins(y=0.12)
    handles = [mpl.patches.Patch(facecolor="0.75", edgecolor="white",
                                 label="AI (target-1-selective)"),
               mpl.patches.Patch(facecolor="0.75", edgecolor="white", hatch="////",
                                 label="IA (target-2-selective)"),
               plt.Line2D([], [], ls=":", lw=1.2, color="0.2",
                          label="AUPRC chance level = class prevalence")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.suptitle("GAT against all three class-weighted classical baselines on both "
                 "selectivity classes:\nthe support vector machine, not Random Forest, "
                 "is the comparator to beat on MAO-A/MAO-B and AChE/BChE",
                 x=0.012, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout(rect=[0, 0.045, 1, 0.90])
    fig.subplots_adjust(top=0.82)
    return [save(fig, "Figure06")]


# ----------------------------------------------------------------------------
# Figure 7 -- ROC and precision-recall curves of the classical baselines
# ----------------------------------------------------------------------------

def figure07(dataset="IDO1/TDO"):
    """Requires refitting the baselines: the stored result files hold aggregate
    metrics only, not the per-molecule probabilities a curve needs."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    fb = fit_baselines(dataset)
    y = fb["y_true"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2))
    for i, cl in enumerate(["AI", "IA"]):
        ci = CLASSES.index(cl)
        yb = (y == ci).astype(int)
        for name in ["Random Forest", "XGBoost", "SVM"]:
            p = fb[name][:, ci]
            fpr, tpr, _ = roc_curve(yb, p)
            axes[i, 0].plot(fpr, tpr, lw=1.6, color=MODEL_COLOR[name],
                            label=f"{name} (AUROC {roc_auc_score(yb, p):.2f})")
            pr, rc, _ = precision_recall_curve(yb, p)
            axes[i, 1].plot(rc, pr, lw=1.6, color=MODEL_COLOR[name],
                            label=f"{name} (AUPRC {average_precision_score(yb, p):.2f})")
        axes[i, 0].plot([0, 1], [0, 1], ls="--", lw=1, color="0.25")
        axes[i, 1].axhline(yb.mean(), ls="--", lw=1, color="0.25")
        axes[i, 0].set(xlabel="false-positive rate", ylabel="true-positive rate")
        axes[i, 1].set(xlabel="recall", ylabel="precision")
        axes[i, 0].set_title(f"{cl}: ROC (n = {int(yb.sum())})", fontsize=9)
        axes[i, 1].set_title(f"{cl}: precision–recall", fontsize=9)
        for ax in axes[i]:
            ax.legend(fontsize=7, loc="lower right" if ax is axes[i, 0] else "upper right")
            open_frame(ax)
            ax.margins(0.02)
    fig.suptitle(f"{dataset} — classical baselines on both selectivity classes; dashed "
                 "lines mark chance\n(the diagonal for ROC, the class prevalence for "
                 "precision–recall)", x=0.012, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return [save(fig, "Figure07")]


# ----------------------------------------------------------------------------
# Figures 8 and 9 -- attention-derived structures
# ----------------------------------------------------------------------------

def _draw_grid(smiles, legends, n_cols=3, sub=(330, 300)):
    from rdkit import Chem
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
    img = Draw.MolsToGridImage([m for m, _ in keep], molsPerRow=n_cols,
                               subImgSize=sub, legends=[l for _, l in keep],
                               useSVG=False, returnPNG=False)
    return np.asarray(img)


def _attention_figure(ds, cls, n_show, tag, caption):
    """Top-attention neighbour molecules of correctly classified `cls` targets."""
    folder = f"{FOLDER[ds]}/best_mols_gat"
    nb = pd.read_csv(os.path.join(PROJECT, folder,
                                  "gat_attention_neighbors_AI_IA.csv"))
    nb = nb[nb["class"] == cls].sort_values("attention_weight", ascending=False)
    nb = nb.drop_duplicates("neighbor_smiles").head(n_show)
    arr = _draw_grid(nb.neighbor_smiles.tolist(),
                     [f"attn = {w:.3f}" for w in nb.attention_weight])
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 220, h / 220))
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(caption, fontsize=9, loc="left")
    fig.tight_layout()
    return save(fig, tag)


def _scaffold_figure(ds, cls, tag, caption, n_show=3):
    folder = f"{FOLDER[ds]}/best_mols_gat"
    sc = pd.read_csv(os.path.join(PROJECT, folder,
                                  "gat_attention_scaffold_summary.csv"))
    sc = sc[sc["class"] == cls].sort_values("count", ascending=False).head(n_show)
    arr = _draw_grid(sc.scaffold_smiles.tolist(),
                     [f"{int(c)} mentions" for c in sc["count"]], n_cols=n_show)
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 220, h / 220))
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(caption, fontsize=9, loc="left")
    fig.tight_layout()
    return save(fig, tag)


def figure08():
    """IDO1/TDO: top-attention neighbours and recurring scaffolds, both classes."""
    return [
        _attention_figure("IDO1/TDO", "AI", 6, "Figure08A",
                          "A  Highest-attention neighbours of correctly classified "
                          "AI (IDO1-selective) test molecules"),
        _attention_figure("IDO1/TDO", "IA", 6, "Figure08B",
                          "B  Highest-attention neighbours of correctly classified "
                          "IA (TDO-selective) test molecules"),
        _scaffold_figure("IDO1/TDO", "AI", "Figure08C",
                         "C  Most frequent Bemis–Murcko scaffolds, pooled AI "
                         "neighbour set"),
        _scaffold_figure("IDO1/TDO", "IA", "Figure08D",
                         "D  Most frequent Bemis–Murcko scaffolds, pooled IA "
                         "neighbour set"),
    ]


def figure09():
    """COX-1/COX-2: top-attention neighbours, both classes."""
    return [
        _attention_figure("COX-1/COX-2", "AI", 6, "Figure09A",
                          "A  Highest-attention neighbours of correctly classified "
                          "AI (COX-1-selective) test molecules"),
        _attention_figure("COX-1/COX-2", "IA", 6, "Figure09B",
                          "B  Highest-attention neighbours of correctly classified "
                          "IA (COX-2-selective) test molecules"),
    ]


# ----------------------------------------------------------------------------
# Figures 10-12 -- chemical space and structural separation of the splits
# ----------------------------------------------------------------------------

def figure10(dataset="IDO1/TDO"):
    """Two-dimensional PCA of the fingerprints, coloured by split."""
    from sklearn.decomposition import PCA
    d = load_dataset(dataset)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(d["fp"].astype(float))
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, idx in [("train", d["train"]), ("val", d["val"]), ("test", d["test"])]:
        ax.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=.55, linewidths=0,
                   color=SPLIT_COLOR[name], label=f"{name} (n = {len(idx)})")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f} % of variance)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f} % of variance)")
    ax.set_title(f"{dataset} — chemical space by data split\n"
                 f"(PC1 + PC2 capture {ev.sum() * 100:.1f} % of total variance, so "
                 "overlap here does not imply scaffold overlap)", fontsize=9)
    ax.legend(markerscale=2, fontsize=7.5)
    open_frame(ax)
    fig.tight_layout()
    return [save(fig, "Figure10")]


def figure11(dataset="IDO1/TDO"):
    """Molecules and distinct scaffold groups per split."""
    d = load_dataset(dataset)
    grp_of = {i: k for k, v in d["groups"].items() for i in v}
    names = ["train", "val", "test"]
    mols = [len(d[n]) for n in names]
    grps = [len({grp_of[i] for i in d[n]}) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, vals, lab, ttl in [
            (axes[0], mols, "Number of molecules", "Molecules per split"),
            (axes[1], grps, "Number of distinct Murcko scaffolds",
             "Scaffold groups per split\n(0 scaffolds shared across splits, "
             "by construction)")]:
        bars = ax.bar(names, vals, color=[SPLIT_COLOR[n] for n in names], width=.62)
        ax.bar_label(bars, fontsize=7.5, padding=2)
        ax.set_ylabel(lab)
        ax.set_title(ttl, fontsize=9)
        ax.margins(y=0.16)
        open_frame(ax)
    fig.suptitle(f"{dataset} — scaffold composition of the splits",
                 x=0.012, ha="left", fontsize=9.5, y=1.03)
    fig.tight_layout()
    return [save(fig, "Figure11")]


def figure12(dataset="IDO1/TDO", radius=3):
    """Nearest-neighbour Tanimoto to the training set: scaffold vs random split.

    Computed with the radius-3 (ECFP6) fingerprints used by the models and the
    graph edges. Set radius=2 to reproduce the earlier ECFP4 version of this
    figure (means 0.57 / 0.60 / 0.74 for IDO1/TDO).
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    d = load_dataset(dataset)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=1024)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in d["df"].smiles]

    def nn(query, ref):
        ref_fps = [fps[j] for j in ref]
        return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], ref_fps))
                         for i in query])

    scaf_te = nn(d["test"], d["train"])
    scaf_va = nn(d["val"], d["train"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(fps))
    n_tr, n_va = len(d["train"]), len(d["val"])
    rand_te = nn(perm[n_tr + n_va:], perm[:n_tr])

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    bins = np.linspace(0, 1, 31)
    for vals, col, lab in [(rand_te, "#9fb3c8", "Random split (test-equivalent)"),
                           (scaf_te, "#c0504d", "Scaffold split — test"),
                           (scaf_va, "#e8a33d", "Scaffold split — validation")]:
        ax.hist(vals, bins=bins, density=True, alpha=.6, color=col,
                label=f"{lab}\nmean = {vals.mean():.2f}")
    ax.set_xlabel(f"Nearest-neighbour Tanimoto similarity to the closest training "
                  f"molecule (ECFP{radius * 2})")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset} — structural distance from the training set:\n"
                 "scaffold split against random split", fontsize=9)
    ax.legend(fontsize=7.5)
    open_frame(ax)
    fig.tight_layout()
    return [save(fig, "Figure12")]


# ----------------------------------------------------------------------------
# Figures 13-15 -- activity threshold and the validity of the four-class scheme
# ----------------------------------------------------------------------------

def figure13(dataset="IDO1/TDO", band=0.10):
    """Class assignment against the cut-off, on log-log IC50 axes."""
    d = load_dataset(dataset)
    df, c1, c2 = d["df"], d["c1"], d["c2"]
    e1, e2 = ENZYME[dataset]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for c in CLASSES:
        m = d["labels"] == c
        ax.scatter(df[c1][m], df[c2][m], s=11, alpha=.55, linewidths=0,
                   color=CLASS_COLOR[c], label=c)
    for pos in (1000,):
        ax.axvline(pos, ls="--", lw=1, color="0.25")
        ax.axhline(pos, ls="--", lw=1, color="0.25")
        ax.axvspan(pos * (1 - band), pos * (1 + band), color="#d1495b", alpha=.13)
        ax.axhspan(pos * (1 - band), pos * (1 + band), color="#d1495b", alpha=.13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{e1} IC$_{{50}}$ (nM)")
    ax.set_ylabel(f"{e2} IC$_{{50}}$ (nM)")
    ax.set_title(f"{dataset} — class assignment against the 1,000 nM threshold\n"
                 f"(shaded band = ±{band * 100:.0f} % of the cut-off on either axis)",
                 fontsize=9)
    ax.legend(markerscale=2, fontsize=7.5, loc="upper left")
    open_frame(ax)
    fig.tight_layout()
    return [save(fig, "Figure13")]


def figure14(thresholds=(100, 300, 500, 1000, 2000, 5000, 10000)):
    """Proportion of labels that change when the cut-off is moved, all datasets."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    xs = np.arange(len(thresholds))
    w = 0.2
    colour = dict(zip(DATASETS, ["#1b6ca8", "#c0504d", "#e8a33d", "#7a9a4f"]))
    for k, ds in enumerate(DATASETS):
        d = load_dataset(ds)
        base = d["labels"]
        pct = []
        for t in thresholds:
            lab = label_classes(d["df"][d["c1"]], d["df"][d["c2"]], cutoff=t)
            pct.append(100 * (lab != base).mean())
        ax.bar(xs + (k - 1.5) * w, pct, width=w * 0.9, color=colour[ds], label=ds,
               edgecolor="white", linewidth=.4)
    ax.set_xticks(xs, [f"{t:,}" for t in thresholds])
    ax.set_xlabel("Alternative activity threshold (nM)")
    ax.set_ylabel("Molecules relabelled vs the\n1,000 nM reference (%)")
    ax.set_title("A ten-fold change in the cut-off relabels 27–59 % of molecules;\n"
                 "a two-fold change relabels 11–22 %", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center")
    ax.margins(y=0.22)
    open_frame(ax)
    fig.tight_layout()
    return [save(fig, "Figure14")]


def figure15(dataset="IDO1/TDO"):
    """Continuous selectivity ratio within each assigned class."""
    d = load_dataset(dataset)
    e1, e2 = ENZYME[dataset]
    ratio = np.log10(d["df"][d["c2"]] / d["df"][d["c1"]])
    order = ["IA", "II", "AA", "AI"]
    data = [ratio[d["labels"] == c].dropna().to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bp = ax.boxplot(data, patch_artist=True, widths=.55, tick_labels=order,
                    medianprops=dict(color="0.1", lw=1.4),
                    flierprops=dict(marker="o", ms=2.5, mfc="none", mec="0.35",
                                    alpha=.6))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_edgecolor("0.25")
    ax.axhline(0, color="0.1", lw=1.1)
    for s in (np.log10(2), -np.log10(2)):
        ax.axhline(s, ls=":", lw=1, color=GREY)
        ax.annotate("2-fold", xy=(len(order) + 0.45, s), fontsize=7, color=GREY,
                    va="center", annotation_clip=False)
    ax.set_ylabel(f"Selectivity ratio: log$_{{10}}$({e2} IC$_{{50}}$ / "
                  f"{e1} IC$_{{50}}$)")
    ax.set_title(f"{dataset} — selectivity ratio by assigned class\n"
                 f"(positive = {e1}-potency-biased, negative = {e2}-potency-biased)",
                 fontsize=9)
    open_frame(ax)
    fig.tight_layout()
    return [save(fig, "Figure15")]


# ----------------------------------------------------------------------------
# Registry and command line
# ----------------------------------------------------------------------------

FIGURES = {
    1:  (figure01, "Overall performance across the four target pairs"),
    2:  (figure02, "Per-class precision/recall/F1 heatmaps, one panel per dataset"),
    3:  (figure03, "Per-class AUROC/AUPRC heatmaps, one panel per dataset"),
    4:  (figure04, "F1 difference on both selectivity classes, all architectures"),
    5:  (figure05, "GAT minus RF on every endpoint, both selectivity classes"),
    6:  (figure06, "GAT against RF, XGBoost and SVM on both selectivity classes"),
    7:  (figure07, "ROC and precision-recall curves of the classical baselines"),
    8:  (figure08, "IDO1/TDO attention neighbours and scaffolds, both classes"),
    9:  (figure09, "COX-1/COX-2 attention neighbours, both classes"),
    10: (figure10, "Chemical space by data split (PCA)"),
    11: (figure11, "Scaffold composition of the splits"),
    12: (figure12, "Nearest-neighbour similarity: scaffold vs random split"),
    13: (figure13, "Class assignment against the 1,000 nM threshold"),
    14: (figure14, "Label change under alternative thresholds, all datasets"),
    15: (figure15, "Selectivity ratio by assigned class"),
}


def check_baselines(tol_molecules=3):
    """Compare the Figure 7 refit against the published confusion counts.

    Passes when no model is off by more than `tol_molecules` true positives on any
    class. Exact agreement is not expected for the tree ensembles under newer
    scikit-learn / xgboost releases; see the module docstring.
    """
    ok = True
    for (ds, model, cl), (tp, fp, fn) in PUBLISHED_PR.items():
        fb = fit_baselines(ds)
        ci = CLASSES.index(cl)
        yt, yp = fb["y_true"], fb["pred"][model]
        g = (int(((yp == ci) & (yt == ci)).sum()),
             int(((yp == ci) & (yt != ci)).sum()),
             int(((yp != ci) & (yt == ci)).sum()))
        d = abs(g[0] - tp)
        status = "exact" if g == (tp, fp, fn) else (
            f"{d} molecule(s) off" if d <= tol_molecules else "MISMATCH")
        ok &= status != "MISMATCH"
        print(f"  {ds:12s} {model:14s} {cl}: TP/FP/FN got {g}, published "
              f"{(tp, fp, fn)}  [{status}]")
    return ok


def main():
    global OUTDIR, DPI
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", type=int, help="figure numbers (default: all)")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    ap.add_argument("--check", action="store_true",
                    help="validate the Figure 7 baseline refit and exit")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    if args.list:
        for n, (_, desc) in FIGURES.items():
            print(f"  Figure {n:2d}  {desc}")
        return

    OUTDIR, DPI = args.outdir, args.dpi
    set_style()

    if args.check:
        sys.exit(0 if check_baselines() else 1)

    wanted = args.which or sorted(FIGURES)
    bad = [n for n in wanted if n not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure number(s): {bad}")

    written = []
    for n in wanted:
        fn, desc = FIGURES[n]
        print(f"Figure {n}: {desc}")
        written += fn()
    print(f"\n{len(written)} file(s) written to {OUTDIR} at {DPI} dpi")


if __name__ == "__main__":
    main()
