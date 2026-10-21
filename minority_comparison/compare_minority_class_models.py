"""
Compare several fitted classifiers (RF, XGBoost, SVC, ... or precomputed outputs of a GNN) on the classes that matter for drug discovery
(default: the selective classes AI and IA): per-class F1-score, AUROC and AUPRC, with uncertainty and a significance test.

Why a bootstrap?  Each model was trained once, so there are no seeds to vary.  What does vary is WHICH molecules are in the test set:
with a handful of AI or IA molecules a model can look better by chance.  The test set is therefore resampled (paired: every model is
scored on the same resample; stratified: every class keeps its size) and the metrics are recomputed B times.  From that we get
  * a 95% percentile confidence interval for every model and metric,
  * a 95% interval and a two-sided p-value for every model-versus-model difference (Holm-corrected within a family of tests),
  * a verdict per endpoint: clear winner / best but not separated / no difference.
The intervals describe test-set sampling uncertainty only (not training randomness or hyper-parameter selection).
Known limitation (checked by simulation): with only ~25 molecules in a class the percentile bootstrap is slightly optimistic - the bootstrap
standard error was ~12% too small and the nominal 95% interval covered the true value in ~82% of fresh test sets.  For a class with about 10
molecules treat the intervals and p-values as approximate and lean towards the conservative reading.
Note: a bootstrap p-value cannot be smaller than 2/(B+1) and Holm multiplies the smallest one by the number of tests in the family, so
B must exceed about 40 x (tests per family).  With 3 models and 2 classes of interest that is 18 tests -> B >= 720; the default 2000 is safe
up to 4 models (36 tests); a warning is raised if B is too small.

Usage
    from compare_minority_class_models import compare_models
    models = {"RF": best_model_rf, "XGBoost": best_model_xgb, "SVC": best_model_svm}      # anything with predict_proba / predict,
    # or a tuple (proba [n, K], predicted_labels [n]) for models you have already evaluated (e.g. a GNN: np.exp(log_softmax))
    res = compare_models(models, X_test, y_test, class_names=["AA", "AI", "IA", "II"], classes_of_interest=("AI", "IA"), out_dir="minority_comparison")
"""
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

METRICS = ("F1", "AUROC", "AUPRC")
MEAN_LABEL = "mean of classes of interest"


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float); out = np.empty_like(p); order = np.argsort(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def get_outputs(model, X, labels_from="predict"):
    """-> (probabilities [n, K], predicted labels [n]).  `model` may be a fitted estimator or a tuple (proba, labels)."""
    if isinstance(model, (tuple, list)):
        proba, pred = np.asarray(model[0], float), np.asarray(model[1])
    else:
        proba = np.asarray(model.predict_proba(X), float)
        pred = np.asarray(model.predict(X)) if labels_from == "predict" else proba.argmax(1)
    return proba, pred


def class_metrics(y, proba, pred, k):
    """F1 (from the predicted labels), AUROC and AUPRC (from the probabilities) of class k against the rest."""
    pos = (y == k)
    tp = np.sum(pos & (pred == k)); fp = np.sum(~pos & (pred == k)); fn = np.sum(pos & (pred != k))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return f1, roc_auc_score(pos, proba[:, k]), average_precision_score(pos, proba[:, k])


def stratified_resample(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == k), size=int(np.sum(y == k)), replace=True) for k in np.unique(y)])


# ----------------------------------------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------------------------------------
def compare_models(models, X_test, y_test, class_names=("AA", "AI", "IA", "II"), classes_of_interest=("AI", "IA"), n_boot=2000, seed=42,
                   labels_from="predict", out_dir="minority_comparison", make_figures=True, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    y = np.asarray(y_test).astype(int); names = list(models); M = len(names)
    cidx = [list(class_names).index(c) for c in classes_of_interest]; C = len(cidx)
    outs = {n: get_outputs(m, X_test, labels_from) for n, m in models.items()}
    for n, (pr, pd_) in outs.items():
        assert pr.shape == (len(y), len(class_names)), f"{n}: probabilities have shape {pr.shape}, expected {(len(y), len(class_names))}"

    # ---- point estimates on the real test set -> est[m, c, j] ----
    est = np.array([[class_metrics(y, outs[n][0], outs[n][1], k) for k in cidx] for n in names])
    # ---- paired, stratified bootstrap -> boot[b, m, c, j] ----
    rng = np.random.default_rng(seed); boot = np.empty((n_boot, M, C, 3))
    for b in range(n_boot):
        idx = stratified_resample(y, rng); yb = y[idx]
        for mi, n in enumerate(names):
            pr, pdl = outs[n][0][idx], outs[n][1][idx]
            for ci, k in enumerate(cidx):
                boot[b, mi, ci] = class_metrics(yb, pr, pdl, k)
    # add the mean over the classes of interest as one more "class"
    est = np.concatenate([est, est.mean(1, keepdims=True)], axis=1); boot = np.concatenate([boot, boot.mean(2, keepdims=True)], axis=2)
    labels = list(classes_of_interest) + [MEAN_LABEL]
    n_k = {c: int(np.sum(y == list(class_names).index(c))) for c in classes_of_interest}
    prev = {c: n_k[c] / len(y) for c in classes_of_interest}

    # ---- table 1: every model, class and metric with its 95% CI ----
    rows = []
    for mi, n in enumerate(names):
        for ci, lab in enumerate(labels):
            for j, met in enumerate(METRICS):
                lo, hi = np.percentile(boot[:, mi, ci, j], [2.5, 97.5])
                row = dict(Model=n, Class=lab, Metric=met, Estimate=est[mi, ci, j], CI_low=lo, CI_high=hi,
                           n_test_in_class=n_k.get(lab, np.nan), chance_level=(prev.get(lab, np.nan) if met == "AUPRC" else 0.5 if met == "AUROC" else np.nan))
                if met == "AUPRC" and lab in prev: row["AUPRC_lift_over_chance"] = est[mi, ci, j] / prev[lab]
                rows.append(row)
    table = pd.DataFrame(rows)

    # ---- table 2: pairwise differences (A - B) with bootstrap p-values, Holm within two families ----
    prow = []
    for (a, b_) in itertools.combinations(range(M), 2):
        for ci, lab in enumerate(labels):
            for j, met in enumerate(METRICS):
                d = boot[:, a, ci, j] - boot[:, b_, ci, j]
                p = min(1.0, 2 * min((1 + np.sum(d <= 0)) / (n_boot + 1), (1 + np.sum(d >= 0)) / (n_boot + 1)))
                lo, hi = np.percentile(d, [2.5, 97.5])
                prow.append(dict(A=names[a], B=names[b_], Class=lab, Metric=met, Difference_A_minus_B=est[a, ci, j] - est[b_, ci, j],
                                 CI_low=lo, CI_high=hi, p_raw=p, family="per-class" if lab != MEAN_LABEL else "mean of classes of interest"))
    pw = pd.DataFrame(prow, columns=["A", "B", "Class", "Metric", "Difference_A_minus_B", "CI_low", "CI_high", "p_raw", "family"])
    pw["p_holm"] = np.nan
    for fam, g in pw.groupby("family"):
        pw.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    pw["significant_holm"] = pw.p_holm < 0.05
    # a bootstrap p-value cannot be smaller than 2/(B+1); Holm multiplies the smallest one by the family size -> B must be large enough
    biggest = int(pw.groupby("family").size().max()) if len(pw) else 0; need = int(np.ceil(40 * biggest))
    if biggest and n_boot < need:
        warnings.warn(f"n_boot={n_boot} is too small for {biggest} tests per family: the smallest attainable Holm-adjusted p-value is "
                      f"{min(1.0, 2 * biggest / (n_boot + 1)):.3f}, so no difference can be called significant. Use n_boot >= {need}.", stacklevel=2)

    # ---- table 3: verdict per endpoint ----
    vrows = []
    for ci, lab in enumerate(labels):
        for j, met in enumerate(METRICS):
            best = int(np.argmax(est[:, ci, j])); beaten, tied = [], []
            for o in range(M):
                if o == best: continue
                r = pw[(pw.Class == lab) & (pw.Metric == met) & (((pw.A == names[best]) & (pw.B == names[o])) | ((pw.A == names[o]) & (pw.B == names[best])))].iloc[0]
                diff_best = r.Difference_A_minus_B if r.A == names[best] else -r.Difference_A_minus_B
                (beaten if (r.significant_holm and diff_best > 0) else tied).append(names[o])
            if M == 1: verdict = "only one model"
            elif not tied: verdict = f"clear winner: {names[best]}"
            elif beaten: verdict = f"{names[best]} > {', '.join(beaten)}; not separated from {', '.join(tied)}"
            else: verdict = f"no significant difference (highest: {names[best]})"
            vrows.append(dict(Class=lab, Metric=met, **{f"{n}": est[i, ci, j] for i, n in enumerate(names)}, best=names[best], verdict=verdict))
    verdict = pd.DataFrame(vrows)

    # ---- table 4: point estimates of precision / recall at the default decision rule (no CI) ----
    pr_rows = []
    for n in names:
        for k, lab in zip(cidx, classes_of_interest):
            pos = y == k; pred = outs[n][1]
            tp, fp, fn = np.sum(pos & (pred == k)), np.sum(~pos & (pred == k)), np.sum(pos & (pred != k))
            pr_rows.append(dict(Model=n, Class=lab, TP=int(tp), FP=int(fp), FN=int(fn), Precision=tp / (tp + fp) if tp + fp else np.nan, Recall=tp / (tp + fn) if tp + fn else np.nan))
    prtab = pd.DataFrame(pr_rows)

    table.to_csv(os.path.join(out_dir, "minority_metrics_with_CI.csv"), index=False); pw.to_csv(os.path.join(out_dir, "minority_pairwise_differences.csv"), index=False)
    verdict.to_csv(os.path.join(out_dir, "minority_verdict.csv"), index=False); prtab.to_csv(os.path.join(out_dir, "minority_precision_recall.csv"), index=False)
    if make_figures:
        plot_intervals(table, names, labels, os.path.join(out_dir, "Figure_minority_metrics_CI.png"))
        plot_curves(outs, names, y, cidx, classes_of_interest, os.path.join(out_dir, "Figure_minority_ROC_PR_curves.png"))
    if verbose:
        pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
        print(f"test molecules: {len(y)} | per class of interest: {n_k} | bootstrap: {n_boot} stratified paired resamples (seed {seed})")
        t = table.assign(txt=lambda d: d.Estimate.round(3).astype(str) + " [" + d.CI_low.round(3).astype(str) + ", " + d.CI_high.round(3).astype(str) + "]")
        print("\nmetric [95% CI] per model:"); print(t.pivot_table(index=["Class", "Metric"], columns="Model", values="txt", aggfunc="first")[names].to_string())
        print("\nverdict per endpoint (Holm-corrected pairwise tests):"); print(verdict[["Class", "Metric", "best", "verdict"]].to_string(index=False))
        print("\nprecision / recall at the default decision rule:"); print(prtab.round(3).to_string(index=False))
    return dict(metrics=table, pairwise=pw, verdict=verdict, precision_recall=prtab, boot=boot, estimates=est, model_names=names, labels=labels)


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
COLORS = ["#2563eb", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#64748b"]


def plot_intervals(table, names, labels, path):
    fig, axes = plt.subplots(len(labels), 3, figsize=(13, 2.0 * len(labels) + 1.5), squeeze=False)
    for r, lab in enumerate(labels):
        for c, met in enumerate(METRICS):
            ax = axes[r][c]; sub = table[(table.Class == lab) & (table.Metric == met)].set_index("Model").loc[names]
            for i, n in enumerate(names):
                ax.errorbar(sub.Estimate[n], i, xerr=[[sub.Estimate[n] - sub.CI_low[n]], [sub.CI_high[n] - sub.Estimate[n]]], fmt="o", color=COLORS[i % 6], capsize=4, markersize=7, lw=2)
            chance = sub.chance_level.iloc[0]
            if not np.isnan(chance): ax.axvline(chance, color="grey", ls="--", lw=1)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names if c == 0 else [], fontsize=10); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(left=max(0, min(sub.CI_low.min() - 0.05, chance - 0.05 if not np.isnan(chance) else 1)), right=min(1.0, sub.CI_high.max() + 0.05))
            if r == 0: ax.set_title(met, fontweight="bold")
            if c == 0: ax.set_ylabel(lab, fontweight="bold", rotation=0, ha="right", va="center", labelpad=10, fontsize=9)
    fig.suptitle("Test-set metrics with 95% bootstrap CIs (dashed: chance level for AUROC = 0.5, AUPRC = class prevalence)", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_curves(outs, names, y, cidx, cnames, path):
    fig, axes = plt.subplots(len(cidx), 2, figsize=(10, 4.2 * len(cidx)), squeeze=False)
    for r, (k, cn) in enumerate(zip(cidx, cnames)):
        pos = (y == k)
        for i, n in enumerate(names):
            fpr, tpr, _ = roc_curve(pos, outs[n][0][:, k]); pre, rec, _ = precision_recall_curve(pos, outs[n][0][:, k])
            axes[r][0].plot(fpr, tpr, color=COLORS[i % 6], lw=2, label=f"{n} (AUROC {roc_auc_score(pos, outs[n][0][:, k]):.2f})")
            axes[r][1].plot(rec, pre, color=COLORS[i % 6], lw=2, label=f"{n} (AUPRC {average_precision_score(pos, outs[n][0][:, k]):.2f})")
        axes[r][0].plot([0, 1], [0, 1], "k--", lw=1); axes[r][1].axhline(pos.mean(), color="k", ls="--", lw=1)
        axes[r][0].set_title(f"{cn}: ROC (n = {int(pos.sum())})", fontweight="bold"); axes[r][1].set_title(f"{cn}: precision-recall", fontweight="bold")
        axes[r][0].set_xlabel("false-positive rate"); axes[r][0].set_ylabel("true-positive rate"); axes[r][1].set_xlabel("recall"); axes[r][1].set_ylabel("precision")
        for a in axes[r]: a.legend(fontsize=8, loc="lower right" if a is axes[r][0] else "upper right"); a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
