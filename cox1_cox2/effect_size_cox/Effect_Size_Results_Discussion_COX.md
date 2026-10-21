# Results and discussion: COX-1/COX-2 inhibitor dataset

## Dataset-specific details

The multi-seed effect-size analysis was applied unchanged to the COX-1/COX-2 dataset (methods as described for the IDO1/TDO dataset: ten seeded runs per architecture, one-sample comparison with the fixed Random Forest reference, 95% confidence intervals, Hedges' g, Holm correction within the primary family of 56 classification-metric comparisons and the secondary family of 40 AUROC/AUPRC comparisons, minimal practically relevant difference of 0.03, and Welch comparisons among GAT configurations). Dataset-specific values: the test set contained 523 molecules (AA 64, AI 20, IA 167, II 272; prevalences 0.12, 0.04, 0.32, 0.52), so AI is the smallest class, and one AI molecule changes AI recall by five percentage points. The class-balanced Random Forest reference had accuracy 0.49, MCC 0.23, macro AUROC 0.635 and macro AUPRC 0.347; its classification metrics were available to two decimals, its AUROC and AUPRC at full precision. A difference of 0.03 in accuracy corresponds to about 16 test molecules.

---

## Results

Plain GAT compared with the Random Forest (selected endpoints; all 96 comparisons are in the supplementary table):

| Endpoint | RF | GAT (mean ± SD, 10 runs) | Δ [95% CI] | Hedges' g | Holm p | Reading |
|---|---|---|---|---|---|---|
| Accuracy | 0.49 | 0.494 ± 0.028 | +0.004 [−0.018, +0.025] | +0.113 | 1.000 | n.s. |
| MCC | 0.23 | 0.238 ± 0.037 | +0.008 [−0.020, +0.035] | +0.180 | 1.000 | n.s. |
| Macro AUROC | 0.63 | 0.687 ± 0.016 | +0.052 [+0.039, +0.064] | +2.734 | <0.001 | significant, meaningful |
| Macro AUPRC | 0.35 | 0.400 ± 0.014 | +0.054 [+0.043, +0.064] | +3.403 | <0.001 | significant, meaningful |
| F1, AA | 0.30 | 0.249 ± 0.028 | −0.051 [−0.072, −0.030] | −1.585 | 0.017 | significant, meaningful |
| F1, AI | 0.06 | 0.169 ± 0.058 | +0.109 [+0.066, +0.153] | +1.638 | 0.014 | significant, meaningful |
| F1, IA | 0.58 | 0.548 ± 0.027 | −0.032 [−0.052, −0.012] | −1.027 | 0.174 | n.s. |
| F1, II | 0.51 | 0.564 ± 0.047 | +0.054 [+0.019, +0.090] | +1.002 | 0.191 | n.s. |
| Recall, AI | 0.05 | 0.255 ± 0.104 | +0.205 [+0.127, +0.283] | +1.717 | 0.010 | significant, meaningful |
| Precision, AI | 0.07 | 0.132 ± 0.046 | +0.062 [+0.027, +0.096] | +1.161 | 0.103 | n.s. |
| AUROC, AI | 0.68 | 0.732 ± 0.033 | +0.051 [+0.026, +0.077] | +1.335 | 0.026 | significant, meaningful |
| AUPRC, AI | 0.08 | 0.119 ± 0.022 | +0.036 [+0.020, +0.052] | +1.439 | 0.018 | significant, meaningful |
| AUROC, II | 0.53 | 0.668 ± 0.021 | +0.141 [+0.125, +0.156] | +5.863 | <0.001 | significant, meaningful |
| AUPRC, II | 0.56 | 0.695 ± 0.019 | +0.134 [+0.120, +0.148] | +6.134 | <0.001 | significant, meaningful |

**Overall performance.** Accuracy and MCC did not differ from the Random Forest for GAT (accuracy +0.004, 95% CI −0.018 to +0.025; MCC +0.008, 95% CI −0.020 to +0.035; Holm p = 1.000 for both). The other architectures showed no MCC difference either, and only GCN's accuracy gain (+0.034) reached nominal significance (Holm p = 0.050), which did not survive the correction over all 96 tests. In contrast, the threshold-free macro metrics were higher than the Random Forest's for every architecture. Macro AUROC was higher by +0.052 for GAT (95% CI +0.039 to +0.064), +0.042 for GINE and +0.030 for GCN (all significant after Holm correction), and by +0.016 for GraphSAGE (not significant after correction, Holm p = 0.050). Macro AUPRC was significantly higher for all four architectures (GAT +0.054, 95% CI +0.043 to +0.064; GINE +0.037; GCN +0.032; GraphSAGE +0.028). GAT's gains of about 0.05 clearly exceed the 0.03 practical-relevance threshold; for GCN and GraphSAGE the gains lie close to it.

**Per-class F1.** F1 for II was higher for GCN (+0.107) and GraphSAGE (+0.099) but not significantly for GAT (+0.054) or GINE. F1 for AA was lower for all four architectures (−0.092 to −0.051), significantly so for GCN, GAT and GINE; GAT's AA recall fell by 0.109, about 7 of the 64 AA molecules. F1 for IA was also lower for all four (−0.080 to −0.027), but significant after correction only for GraphSAGE, where the difference was small.

**The minority class AI.** Only GAT improved AI significantly. Its F1 was 0.169 against 0.06 (+0.109, 95% CI +0.066 to +0.153, Holm p = 0.014) and its recall 0.255 against 0.05 (+0.205, 95% CI +0.127 to +0.283; about 5 versus 1 of the 20 AI molecules, Holm p = 0.010), while its precision (0.132 against 0.07) did not differ significantly. GAT's AI AUROC was 0.732 against 0.681 (+0.051, 95% CI +0.026 to +0.077) and its AI AUPRC 0.119 against 0.083 (+0.036, 95% CI +0.020 to +0.052), about 3.1 times the AI prevalence of 0.038 compared with 2.2 times for the Random Forest. GINE also recovered more AI molecules (recall 0.280) but at a precision of 0.075 and without a significant F1, AUROC or AUPRC gain; GCN and GraphSAGE almost never predicted AI (recall 0.025 and 0.030) and showed no improvement.

**Threshold-free ranking per class.** The largest gains were for II, where the Random Forest's AUROC was 0.527, close to chance: all four architectures ranked II better (+0.117 to +0.141; GAT 0.668, 95% CI of the difference +0.125 to +0.156), and II AUPRC was also significantly higher for all four (+0.101 to +0.134). AUROC for AA did not decrease significantly for any architecture and was higher for GINE (+0.053), so the lower AA F1 reflects the decision threshold and not poorer ranking. For IA, AUROC was lower only for GCN (−0.026, small), and AUPRC was higher for GAT and GraphSAGE (+0.041 and +0.041).

**Corrections and robustness.** For plain GAT, 33 of 56 primary comparisons had raw p < 0.05 and 17 remained significant after Holm correction (32 by Benjamini–Hochberg); in the secondary family 29 of 40 were nominally significant and 21 remained after correction (26 by Benjamini–Hochberg). Eight results lost significance when all 96 tests were corrected together, among them GAT's AI AUROC (Holm p = 0.076) and AI AUPRC (p = 0.050); GAT's AI F1 and recall and its macro AUROC and AUPRC remained significant. No Holm-significant result was sensitive to the rounding of the Random Forest's classification metrics. The number of practically meaningful results was 38, 34 and 27 at thresholds of 0.02, 0.03 and 0.05.

**GAT configurations.** GATv2 and GAT with graph structure learning did not differ significantly from plain GAT on any of the eight primary endpoints (largest absolute differences 0.012 and 0.032; all Holm p ≥ 1.00). Compared with the Random Forest, the three GAT variants showed the same pattern: macro AUROC +0.052 / +0.052 / +0.065 and AI F1 +0.109 / +0.100 / +0.077 for GAT / GATv2 / GAT with graph structure learning.

---

## Discussion

On the COX dataset the GNNs improve the ranking of molecules relative to the Random Forest, while performance at the default decision threshold is unchanged: accuracy and MCC do not differ, but macro AUROC and macro AUPRC are higher by about 0.03 to 0.05. The gain is concentrated in class II, where the Random Forest ranks close to chance (AUROC 0.53), and, for GAT, in the minority class AI. One possible explanation is that neighbourhood information from structurally similar labelled molecules carries signal that the fingerprint features alone do not, but this was not tested here. The comparison also depends on a weak baseline (Random Forest accuracy 0.49), and a stronger or differently tuned baseline might narrow the gap.

Reading threshold-based and threshold-free metrics together clarifies what changed. The lower AA F1 of all four architectures is accompanied by unchanged or higher AA AUROC, so it reflects where each model places its decision threshold. By contrast, the II and AI differences appear in AUROC and AUPRC as well as in F1 or recall, so they reflect better ranking.

**The minority class AI.** For AI, GAT recovers more molecules than the class-balanced Random Forest (recall 0.26 versus 0.05), without a loss of precision, and ranks them better (AUROC +0.051, AUPRC +0.036). This evidence is more consistent than for the IA class of the IDO1/TDO dataset, but it does not justify calling GAT clearly or certainly superior for AI, for four reasons. (i) Absolute performance remains low: precision is 0.13, about 6.6 wrong AI calls per correct one (against 13 for the Random Forest), and AUPRC is about three times the class prevalence. (ii) The test set holds only 20 AI molecules, and the intervals reported here reflect training-seed variability only; as an approximate, unpaired illustration, recovering 5 versus 1 of 20 molecules gives overlapping 95% Wilson intervals (0.11–0.47 versus 0.01–0.24) and a Fisher exact p of about 0.18. (iii) Under the strictest correction (all 96 tests together) the AI AUROC and AUPRC differences become borderline, although AI F1 and recall remain significant. (iv) Only GAT shows the improvement, and the Random Forest is a single fixed run. We therefore describe the finding as improved detection and ranking of AI by GAT that is preliminary given 20 test molecules; scaffold-based cross-validation, a paired per-molecule comparison and repetition of the Random Forest across seeds would strengthen it.

**Statistical versus practical significance.** The effect sizes separate negligible from material differences: GAT's accuracy and MCC differences are close to zero, whereas its macro AUROC and AUPRC gains (about 0.05) exceed the 0.03 threshold and are consistent across architectures. Hedges' g reaches 3 or more only because it is expressed in units of the small run-to-run standard deviation, so conventional benchmarks should not be applied. The 0.03 threshold is a domain judgement and AUPRC is scaled by prevalence, so it may suit AUPRC less well than AUROC.

**Comparison across datasets.** The two datasets differ in what the GNNs add (GAT compared with the class-balanced Random Forest):

| GAT − RF | IDO1/TDO | COX |
|---|---|---|
| Accuracy | −0.030 [−0.039, −0.020] (p 0.003) | +0.004 [−0.018, +0.025] (p 1.000) |
| MCC | −0.033 [−0.048, −0.018] (p 0.017) | +0.008 [−0.020, +0.035] (p 1.000) |
| Macro AUROC | −0.012 [−0.025, +0.001] (p 1.000) | +0.052 [+0.039, +0.064] (p <0.001) |
| Macro AUPRC | +0.017 [+0.001, +0.033] (p 1.000) | +0.054 [+0.043, +0.064] (p <0.001) |
| Minority-class F1 (IA / AI) | +0.130 [+0.107, +0.153] (p <0.001) | +0.109 [+0.066, +0.153] (p 0.014) |
| Minority-class recall | +0.337 [+0.306, +0.369] (p <0.001) | +0.205 [+0.127, +0.283] (p 0.010) |
| Minority-class precision | −0.291 [−0.314, −0.268] (p <0.001) | +0.062 [+0.027, +0.096] (p 0.103) |
| Minority-class AUROC | +0.017 [−0.013, +0.048] (p 1.000) | +0.051 [+0.026, +0.077] (p 0.026) |
| Minority-class AUPRC | +0.084 [+0.042, +0.125] (p 0.046) | +0.036 [+0.020, +0.052] (p 0.018) |

On IDO1/TDO the GNNs mainly shifted the balance between classes, with slightly lower accuracy and MCC and no clear ranking gain, whereas on COX they rank better overall with unchanged accuracy and MCC. Two features are shared: GAT is the only architecture that improved the minority class in both datasets, and neither GATv2 nor graph structure learning improved on plain GAT (the latter reduced accuracy, MCC, macro AUROC and AI F1 on IDO1/TDO and was neutral on COX). The benefit of graph-based models over fingerprint-based Random Forest is therefore dataset-dependent and not a uniform superiority, which is a more cautious and better-supported conclusion for the generalizability question than a general claim.

**Limitations.** (1) The intervals and p-values capture training-seed variability only, not uncertainty from the finite test set, and the AI class has 20 test molecules. (2) The Random Forest is treated as a fixed constant and is weak on this dataset. (3) Per-class results are interdependent and the endpoints within a family are correlated, so Holm correction is conservative; several conclusions change under the stricter all-96 correction. (4) The 0.03 threshold is a domain judgement. (5) Results come from a single split per dataset and two datasets in total, and tuning effort may differ between them.
