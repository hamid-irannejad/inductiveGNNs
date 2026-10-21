# GAT Attention-Based Neighbor Identification and Structural Motif Analysis

*(Methods and Results/Discussion text describing the attention-extraction pipeline and
the resulting chemical structures — for inclusion in the manuscript.)*

---

## Methods

To move beyond aggregate performance metrics and provide structural insight into the
model's decisions, we used the trained GAT model's own attention mechanism to
identify which molecules in the similarity graph most influenced its classification
of individual test compounds as IDO-selective (AI) or TDO-selective (IA).

**Attention extraction.** For each `GATConv` layer, the graph attention operation
computes, for every edge in the graph, a learned coefficient (the attention weight)
that determines how much the source node's features contribute to updating the
destination node's representation. Because the model's forward pass ordinarily
discards these coefficients after using them, we re-ran the trained model's exact
forward computation — reproducing its three-layer architecture (each `GATConv` layer
followed by a linear skip connection, batch normalization, and an ELU activation,
identical to the trained model) — while explicitly requesting the attention
coefficients at each layer. This was verified to reproduce the model's original
hidden-state computation exactly (zero numerical deviation from the trained model's
own forward pass), confirming that the extracted attention weights are precisely
those the model used to arrive at its actual predictions, not an approximation.
Attention weights from the final (third) `GATConv` layer — the representation that
feeds directly into the classification layer — were used for all subsequent
analysis, averaged across attention heads. Self-loop edges (automatically added by
the `GATConv` implementation) were excluded, so that only genuine molecule-to-molecule
edges were considered.

**Identifying influential neighbors.** Analysis was restricted to test-set molecules
whose predicted class matched their true label (i.e., correctly classified AI and IA
compounds), so that the resulting structural signal reflects genuine, correct
decisions rather than the model's errors. For each such target molecule, its
incoming graph edges (i.e., edges from other molecules *into* the target, which
directly inform the target's updated representation) were ranked by attention
weight, and the top three highest-attention neighbor molecules were retained as its
most influential neighbors. This was repeated across all correctly classified AI
molecules and, separately, all correctly classified IA molecules, yielding two
pooled sets of "high-attention neighbor" molecules — one per class.

**Structural motif analysis.** The pooled neighbor sets were analyzed for shared
structure in two complementary ways: (1) Bemis-Murcko scaffold extraction and
frequency counting, to identify ring-system cores that recur across multiple,
independently sampled high-attention neighbors; and (2) maximum common substructure
(MCS) analysis across the most frequently occurring neighbors, to obtain a single
consensus substructure per class. The highest-attention individual neighbor
molecules were also rendered directly as 2D structures, annotated with their
attention weight, for visual inspection (Figures X, Y).

---

## Results and Discussion

Two complementary views of the pooled high-attention neighbor set are presented:
(1) the individual top-ranked neighbor molecules for six example target compounds,
rendered as full 2D structures and annotated with attention weight (Figures X, Y);
and (2) Bemis-Murcko scaffold frequency across the *entire* pooled set of
high-attention neighbors — 148 neighbor mentions across all correctly classified AI
test molecules, and 15 across all correctly classified IA test molecules (Table Z).
Because Murcko scaffolds strip terminal, non-ring substituents to isolate the core
ring topology, Table Z and Figures X/Y are complementary rather than redundant: the
figures show real, fully decorated compounds, while the table reveals which ring
system recurs across a much larger, independently sampled set of the model's
attention-weighted decisions. As an internal consistency check, the number of
neighbor mentions per class (148 for AI, 15 for IA) implies roughly 49 and 5
correctly classified target molecules, respectively (148/3 and 15/3, given the
top-3-neighbors-per-target design) — closely matching GAT's actual multi-run mean
recall on these classes (0.769 for AI, 0.570 for IA; Table 1), confirming the pooled
neighbor set behaves as expected.

### AI (IDO-selective) neighbors — Table Z, Figure X

Pooled across all correctly classified AI molecules, the single most recurring
scaffold is an **N-arylfuropyridin-3-amine** (a furopyridine core bearing an
anilino substituent), accounting for 23 of 148 neighbor mentions (16%); a closely
related pyridinyl-substituted variant of the same furopyridine-aniline core adds a
further 14 mentions (9%), so this one chemotype accounts for roughly a quarter of
all high-attention AI neighbors. The compound illustrated in Figure X with the
single highest attention weight in this study (attn = 0.515) is an example of this
exact scaffold. Beyond this dominant motif, the remaining top-10 scaffolds are
chemically diverse rather than variations on a single core: a quinolinyl-piperidine
linked through an oxalamide to an anilide (11 mentions; illustrated in Figure X at
attn = 0.499); a phenyl-substituted fused pyrazolo-oxa tricyclic system (8
mentions); a benzyl-substituted benzo[c][1,2,5]oxadiazole (benzofurazan) amidine (8
mentions); a cyclopropyl-substituted imidazothiazole (6 mentions; the family
illustrated in Figure X at attn = 0.268 and 0.261); a phenyl-1,2,5-oxadiazole
carboximidamide (5 mentions); a cyclohexylmethyl/cyclopropyl-substituted variant of
the same imidazothiazole family (4 mentions); a phenylimidazole linked to an indole
(3 mentions); and, notably, a **biphenyl-tetrazole** linked through a urea (3
mentions) — a substructure recognizable from established medicinal chemistry as a
common carboxylic-acid bioisostere (most familiar from angiotensin-II-receptor
antagonists), interesting to see recur among the model's independently
high-attention neighbors despite no such prior having been provided to it.
Reflecting this diversity, the maximum common substructure across the full pooled
AI neighbor set reduces to a single non-ring carbon (1 atom, 0 bonds) — i.e., no
single fragment is common to *all* high-attention AI neighbors; the signal instead
comes from several structurally distinct, independently recurring chemotypes.

### IA (TDO-selective) neighbors — Table Z, Figure Y

The pooled IA neighbor set is both much smaller (15 vs. 148 mentions, tracking the
class's small size and lower recall) and structurally more concentrated. A single
scaffold — an unsubstituted-piperazine-bearing **imidazo[1,2-a]pyridine** —
accounts for 9 of 15 mentions (60%). This core is not the same compound as those
shown in Figure Y (whose fully decorated structures carry additional
trifluoromethyl and N-acyl groups absent from the bare Murcko scaffold), but
represents the ring system underlying that structural family once peripheral
substituents are stripped away. The remaining 6 mentions form a second, distinct
cluster around a **naphtho-triazole-dione** (a 1,2,3-triazole fused to a
1,4-naphthoquinone) linked through a benzylic methylene to a carboxamide-substituted
cyclic amine — piperazine (3 mentions, illustrated in Figure Y at attn = 0.102,
0.102), morpholine (2 mentions, attn = 0.093), and a regioisomeric
piperazine-amide (1 mention) — the same scaffold family visually identified in
Figure Y, now confirmed exactly from the underlying SMILES. As with AI, the maximum
common substructure across the *full* pooled IA set is trivial (a single ring-to-
chain carbon-carbon bond) — the two IA clusters (imidazopyridine-piperazine and
naphthoquinone-triazole-amide) are structurally unrelated to one another, so a
meaningful consensus substructure exists only *within* each cluster, not across the
class as a whole.

### Comparing AI and IA

Three quantitative patterns distinguish the two classes' attention-derived
structural signatures. First, attention magnitude: the highest AI attention weights
(0.26–0.52) are roughly five-fold higher than the highest IA weights (0.08–0.10).
Second, concentration: AI's top scaffold family accounts for ~25% of pooled
neighbor mentions, versus IA's dominant scaffold at 60% — despite pooling from a
much larger sample, AI's attention is more evenly distributed across distinct
chemotypes, while IA's concentrates heavily on one recurring core plus one
secondary cluster. Third, MCS across each full pooled set is trivial for both
classes, confirming that neither class is explained by one universal substructure —
rather, each class's high-attention neighbors form several internally consistent
but mutually distinct chemotype clusters. Together, these patterns suggest the
model has learned a comparatively sharp, multi-chemotype signal for
IDO-selectivity, while its signal for TDO-selectivity, though real (GAT's recall
advantage on IA was itself statistically significant; Table 3), is built from a
much narrower structural precedent — consistent with, and offering a model-grounded
structural explanation for, IA's small labeled sample size (n = 10) and the
correspondingly higher run-to-run variance observed for this class throughout the
statistical analyses in this study.

### Methodological note: cluster-wise MCS

Because both classes' pooled neighbor sets resolve into multiple, internally
consistent but mutually distinct scaffold clusters, a global MCS across the entire
pooled set is, by construction, uninformative (as observed above). A natural
extension — not performed here, but straightforward to add — is to compute MCS
*separately within* each dominant scaffold cluster (e.g., across the three
naphthoquinone-triazole analogs identified for IA, or across the two
furopyridine-aniline variants for AI) rather than across the full heterogeneous
pool, yielding a sharper consensus substructure per chemotype rather than a single
trivial fragment across all chemotypes combined.

### Caveats

These structural groupings are hypotheses generated from the trained model's
learned attention patterns on this dataset and split, not independently validated
pharmacological findings. Attention weight reflects what the model found useful for
its classification decision; it is not a direct measure of binding affinity,
selectivity mechanism, or biological relevance, and should not be over-interpreted
as such without further validation — for example, comparison against published
IDO1/TDO co-crystal structures or site-directed mutagenesis data, or prospective
testing of analogs sharing the identified scaffolds. Ring-system names above were
assigned from direct inspection of the canonical SMILES underlying each scaffold
(rather than from the rendered 2D images alone, correcting an earlier, less
reliable image-only reading of these same structures); a final nomenclature check
with structure-to-name software, and review by a medicinal chemist, is recommended
before submission.
