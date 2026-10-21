# Effect-size analysis of GNN performance relative to Random Forest, including AUROC and AUPRC

## Name of the analysis

**Multi-seed effect-size analysis with confidence intervals and family-wise multiplicity control**
(one-sample comparison of seeded GNN runs against a deterministic, class-balanced Random Forest reference, covering classification metrics and threshold-free AUROC/AUPRC, plus between-variant comparisons among GAT configurations).

Short form for the text: *multi-seed effect-size analysis*.

---

## Methods

### Experimental design

Each GNN architecture (GCN, GAT, GINE, GraphSAGE) was trained ten independent times with fixed random seeds (0–9). All runs used the same scaffold-based split (675/225/226 training/validation/test molecules, with test scaffolds absent from the training and validation sets); the test set contained 58 AA, 82 AI, 11 IA and 75 II molecules. Networks were trained with a class-weighted negative log-likelihood loss, and the checkpoint was selected by validation macro-F1 (early stopping, patience 50, at most 200 epochs). The same procedure was repeated for GATv2 and for GAT with graph structure learning (GSL) in place of the GAT layer.

The Random Forest baseline was trained with `class_weight='balanced'`, the counterpart of the class-weighted GNN loss, with hyperparameters tuned by grid search with 5-fold cross-validation [state the scoring criterion]. With a fixed random seed and fixed split it is deterministic, so each of its metrics is a single fixed reference value. Its classification metrics were available to two decimals; its AUROC and AUPRC at full precision.

### Endpoints

For every architecture we analysed 24 endpoints: accuracy, Matthews correlation coefficient (MCC), precision, recall and F1-score for each class (12), macro AUROC and macro AUPRC, and per-class AUROC and AUPRC (8). This gives 96 comparisons per GAT configuration, split into a **primary family** (56 classification-metric comparisons) and a **secondary family** (40 AUROC/AUPRC comparisons).

### AUROC and AUPRC

AUROC (area under the receiver operating characteristic curve) and AUPRC (area under the precision–recall curve, computed as average precision) were calculated one-versus-rest for each class from predicted class probabilities on the test set: for the GNNs, the exponentiated log-softmax output at the test nodes from each seed's best-validation checkpoint; for Random Forest, `predict_proba`. The macro value is the mean of the four per-class values. Both metrics depend only on how each model ranks molecules within a class, not on a decision threshold. Their chance levels differ: 0.5 for AUROC, but the class prevalence for AUPRC (0.05 for IA; 0.25 for the macro average). Per-seed values were summarized by their mean and standard deviation across the ten runs.

### Difference from the Random Forest baseline

Because the reference is a fixed constant and each GNN contributes n = 10 independent training outcomes, each comparison is a one-sample design. For each endpoint:

- difference Δ = mean(GNN) − value(RF);
- sample standard deviation s = s_pop · √(n/(n − 1)), converting the stored population SD;
- standard error SE = s/√n and test statistic t = Δ/SE with n − 1 = 9 degrees of freedom (two-sided p-value);
- 95% confidence interval Δ ± t(0.975, 9) · SE;
- effect size Hedges' g = (Δ/s) · [1 − 3/(4·df − 1)], the small-sample-corrected one-sample standardized difference.

### Multiple-comparison control

The primary correction was the **Holm step-down procedure applied within each family** (56 tests; 40 tests), controlling the family-wise error rate at 0.05, so that adding AUROC/AUPRC does not change the primary results. As sensitivity analyses we report Benjamini–Hochberg q-values (false discovery rate 0.05) within each family, a Holm correction over all 96 tests together, and, for continuity with earlier drafts, Bonferroni correction within architecture × metric groups. Each GAT configuration was treated as its own family, because the GCN, GINE and GraphSAGE results are identical across configurations and should not be counted repeatedly.

### Practical significance

Statistical significance does not imply practical relevance, because a small difference can be significant when seed-to-seed variability is small. We therefore defined a minimal practically relevant difference (MID) of 0.03 in absolute score [justify this value; for accuracy it corresponds to about 7 of 226 test molecules]. A comparison was labelled *not significant* (Holm-adjusted p ≥ 0.05), *significant but small* (|Δ| < MID) or *significant and practically meaningful* (|Δ| ≥ MID); labels were recomputed at MID = 0.02 and 0.05 to show their dependence on this choice. Differences are also expressed as numbers of test molecules (accuracy × 226; recall × class support). Because AUPRC is scaled by class prevalence, one MID may suit it less well than AUROC.

### Robustness to the precision of the Random Forest reference

The Random Forest classification metrics were available to two decimals, so each true value may differ by up to 0.005. A significant result was called robust if its confidence interval still excluded zero after shifting the reference by ±0.005. No such shift was needed for AUROC/AUPRC, which were available at full precision.

### Comparison among GAT configurations

GATv2 and GAT with GSL were each compared with plain GAT using Welch's unequal-variance t-test computed from the mean and SD summaries (10 versus 10 runs, Welch–Satterthwaite degrees of freedom), with a 95% confidence interval for the difference and Hedges' g based on the pooled SD. Eight primary endpoints were tested (accuracy, MCC, macro AUROC, macro AUPRC and the four class F1-scores), 16 comparisons in total, with Holm correction across all 16.

### Implementation and verification

The analysis was implemented in Python (NumPy, SciPy). The one-sample statistics, the Welch statistics and the Holm and Benjamini–Hochberg adjustments were checked against `scipy.stats` and `statsmodels`, and the implementation reproduced all 288 t-statistics and p-values stored in the result files.

---

## Results

Against the class-balanced Random Forest, plain GAT showed the following (selected endpoints; all 96 comparisons are in the supplementary table):

| Endpoint | RF | GAT (mean ± SD, 10 runs) | Δ [95% CI] | Hedges' g | Holm p | Reading |
|---|---|---|---|---|---|---|
| Accuracy | 0.60 | 0.570 ± 0.013 | -0.030 [-0.039, -0.020] | -1.99 | 0.003 | significant, small |
| MCC | 0.45 | 0.417 ± 0.019 | -0.033 [-0.048, -0.018] | -1.47 | 0.017 | significant, meaningful |
| Macro AUROC | 0.81 | 0.801 ± 0.018 | -0.012 [-0.025, +0.001] | -0.58 | 1.000 | n.s. |
| Macro AUPRC | 0.57 | 0.587 ± 0.021 | +0.017 [+0.001, +0.033] | +0.71 | 1.000 | n.s. |
| F1, AA | 0.57 | 0.585 ± 0.005 | +0.015 [+0.011, +0.019] | +2.66 | <0.001 | significant, small |
| F1, AI | 0.50 | 0.616 ± 0.032 | +0.116 [+0.091, +0.140] | +3.09 | <0.001 | significant, meaningful |
| F1, IA | 0.15 | 0.280 ± 0.031 | +0.130 [+0.107, +0.153] | +3.67 | <0.001 | significant, meaningful |
| F1, II | 0.73 | 0.582 ± 0.022 | -0.148 [-0.165, -0.132] | -5.84 | <0.001 | significant, meaningful |
| Recall, IA | 0.09 | 0.427 ± 0.042 | +0.337 [+0.306, +0.369] | +7.02 | <0.001 | significant, meaningful |
| Precision, IA | 0.50 | 0.209 ± 0.030 | -0.291 [-0.314, -0.268] | -8.27 | <0.001 | significant, meaningful |
| AUROC, IA | 0.74 | 0.760 ± 0.040 | +0.017 [-0.013, +0.048] | +0.38 | 1.000 | n.s. |
| AUPRC, IA | 0.21 | 0.297 ± 0.055 | +0.084 [+0.042, +0.125] | +1.32 | 0.046 | significant, meaningful |

**Overall performance.** No GNN was significantly better than Random Forest on any overall metric (accuracy, MCC, macro AUROC or macro AUPRC). GAT's accuracy and MCC were lower by about 0.03 (Holm p = 0.003 and 0.017), both at the 0.03 practical-relevance threshold; the other three architectures showed similar-sized point deficits with wider intervals that were not significant after correction. Macro AUROC was significantly lower only for GINE (−0.028), and macro AUPRC did not differ for any architecture.

**Per-class F1.** The architectures differed from Random Forest in opposite directions across classes. F1 for AI was higher by about 0.12 for GAT, GCN and GraphSAGE (not significant for GINE); F1 for IA was higher for all four architectures (+0.090 to +0.145); F1 for II was lower for all four (−0.075 to −0.155); AA was essentially unchanged (GAT +0.015, significant but small). For IA, GAT recovered on average 4.7 of the 11 IA test molecules (recall 0.43) against 1 of 11 for Random Forest (recall 0.09), but at a precision of 0.21 against 0.50.

**Threshold-free metrics.** Per-class AUROC did not differ significantly from Random Forest for AI for any architecture. For IA, only GCN was significantly higher (+0.049, 95% CI +0.031 to +0.067); GAT's IA AUROC difference was +0.017 (not significant). GAT's IA AUPRC was higher by +0.084 (95% CI +0.042 to +0.125), which was significant within the AUROC/AUPRC family (Holm p = 0.046) but not when all 96 tests were corrected together (p = 0.079). AUROC for II was lower for GAT (−0.055), GCN and GINE.

**Corrections and robustness.** For plain GAT, 44 of 56 primary comparisons had raw p < 0.05 and 31 remained significant after Holm correction (43 by Benjamini–Hochberg); in the secondary family 14 of 40 were nominally significant and 7 remained after correction. Only one result (GAT IA AUPRC) changed under the all-96 correction, and no Holm-significant result was sensitive to the rounding of the Random Forest reference.

**GAT configurations.** GATv2 did not differ significantly from plain GAT on any of the eight endpoints (for example accuracy: Δ = +0.002, 95% CI -0.017 to +0.021, Holm p = 1.000). GAT with GSL was significantly worse on accuracy (Δ = -0.027, 95% CI -0.040 to -0.013, Holm p = 0.013), MCC (Δ = -0.031, 95% CI -0.050 to -0.012, Holm p = 0.046), macro AUROC (Δ = -0.027, 95% CI -0.043 to -0.011, Holm p = 0.031) and AI F1 (Δ = -0.064, 95% CI -0.101 to -0.028, Holm p = 0.025).

---

## Discussion

Against a Random Forest given the same class weighting, the GNNs show no overall advantage. They shift performance between classes: more AI and IA molecules are recovered, at the cost of II and of a small loss in accuracy and MCC. The threshold-free results suggest that much of this reflects a different decision threshold rather than better ranking: AI F1 is about 0.12 higher while AI AUROC does not differ, while macro AUROC is equal or lower and macro AUPRC is unchanged. This is a different operating point, which may be favourable when finding selective compounds matters more than confirming inactives, but it should be presented as a trade-off, not as general superiority.

**The IA class.** For IA, GAT identifies more molecules at the default decision threshold than the class-balanced Random Forest (recall 0.43 versus 0.09; F1 0.28 versus 0.15), and this holds across training seeds and for all four architectures. This is not sufficient to conclude that GAT is clearly or certainly superior for IA, for four reasons. (i) The gain in recall comes with a much lower precision (0.21 versus 0.50), so it is partly a change of threshold. (ii) Ranking-based evidence is weak: GAT's IA AUROC difference is +0.017 (95% CI -0.013 to +0.048), and its IA AUPRC advantage is borderline after correction. (iii) The test set holds only 11 IA molecules, so a single molecule changes recall by about nine percentage points; the confidence intervals here reflect training-seed variability only. As an approximate, unpaired illustration of test-set uncertainty, recovering about 5 versus 1 of 11 molecules gives overlapping 95% Wilson intervals (0.21–0.72 versus 0.02–0.38) and a Fisher exact p of about 0.15. (iv) Random Forest is a single fixed run, so its own variability is not represented, and its IA recall of 0.09 corresponds to a single correctly predicted molecule (1 of 11). We therefore describe the finding as higher sensitivity for IA at the default threshold, with limited evidence of better ranking, and as preliminary given the sample size. Stronger support would require evaluating all IA molecules (for example by scaffold-based cross-validation), a paired per-molecule comparison (McNemar's test or a paired bootstrap), and repeating Random Forest across seeds.

**Statistical versus practical significance.** The effect sizes and intervals separate large from negligible differences: the AI, IA and II F1 differences are large relative to the 0.03 threshold, whereas GAT's AA F1 difference (+0.015) is significant yet negligible, and GAT's accuracy and MCC deficits sit at the threshold. Hedges' g reaches 5 or more only because it is expressed in units of the small run-to-run SD, so conventional benchmarks (0.2/0.5/0.8) should not be applied; the absolute differences and their intervals are the interpretable quantities.

**GAT configurations.** GATv2 matched plain GAT, and GSL reduced accuracy, MCC, macro AUROC and AI F1, so neither variant improved on plain GAT in this dataset.

**Limitations.** (1) The intervals and p-values capture training-seed variability only, not uncertainty from the finite test set, and the IA class has 11 test molecules. (2) Random Forest is treated as a fixed constant, which omits any variability in its training. (3) Per-class results are interdependent, since gains in one class mirror losses in another, and the endpoints within a family are correlated, so Holm correction is conservative. (4) The 0.03 threshold is a domain judgement and AUPRC is scaled by prevalence. (5) Results on one scaffold split may not generalize to other splits.
