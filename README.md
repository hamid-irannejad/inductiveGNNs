# Molecular Similarity Graph Neural Networks for IDO/TDO Inhibitor Classification

Graph neural network (GNN) architectures compared against a Random Forest (RF) baseline for classifying IDO1/TDO inhibitors into four dual-selectivity classes, using a molecule-as-node similarity graph and rigorous multi-seed statistical validation.

## Overview

Indoleamine 2,3-dioxygenase (IDO1) and tryptophan 2,3-dioxygenase (TDO) are drug targets in cancer immunotherapy, with therapeutic interest in both selective and dual inhibitors. This project builds a similarity graph in which **each molecule is a node** and **edges are defined by pairwise molecular similarity** (selected via mutual k-NN to control graph density), then trains and evaluates several GNN architectures for **4-class node classification**:

| Class | Meaning |
|---|---|
| **AA** | Active against both IDO1 and TDO (dual inhibitor) |
| **AI** | Active against IDO1, inactive against TDO (IDO-selective) |
| **IA** | Inactive against IDO1, active against TDO (TDO-selective) |
| **II** | Inactive against both |

**Headline finding:** Graph Attention Networks (GAT) significantly outperform Random Forest on overall accuracy and MCC, and on F1-score specifically for the two single-target-selectivity classes (AI, IA) — confirmed via 10-seed multi-run statistical testing, not a single training run.

## Dataset

- Source: **ChEMBL36** (IDO1 and TDO inhibitor bioactivity data)
- 4,420 IDO1 + 1,297 TDO inhibitors retrieved and deduplicated (PostgreSQL 17)
- **1,126 molecules** with dual IC50 values (IDO1 *and* TDO) → labeled data (4-class)
- **3,465 molecules** with IC50 for only one target → unlabeled data
- **4,591 molecules total**, used as nodes in the similarity graph

Activity threshold: IC50 < 1000 nM = active ("A"); otherwise inactive ("I").

## Methodology

### Two-stage learning design

1. **Transductive learning** (all 4,591 nodes) — used to select the best-performing combination of node features and edge weights.
   - Node features compared: Morgan fingerprints, MACCS keys, Mordred descriptors, Mol2Vec embeddings
   - Edge weights compared: Tanimoto, cosine, and Dice similarity
   - **Best combination found: Morgan fingerprints (node features) + Tanimoto similarity (edge weight)**

2. **Inductive learning** (1,126 labeled nodes only, held-out test split) — used to compare GNN architectures against Random Forest.

### Architectures compared

- Graph Convolutional Network (**GCN**)
- Graph Attention Network (**GAT**)
- Graph Isomorphism Network with edge features (**GINE**)
- Graph Sample and Aggregate (**GraphSAGE**)
- **Random Forest** (baseline; deterministic given the fixed train/test split)

### Statistical validation (two complementary approaches)

1. **Paired bootstrap resampling** (single training run per architecture) — 5,000 resamples of the fixed test set, used as an initial screening analysis. Precision/recall/F1 compared via percentile confidence intervals and empirical p-values, Bonferroni-corrected across classes.
2. **Multi-seed one-sample t-test** (primary, confirmatory analysis) — each GNN architecture retrained **10 times** with independently fixed random seeds (0–9); the resulting 10-run distribution is compared against RF's single deterministic value via a one-sample t-test, Bonferroni-corrected across classes.

The multi-seed approach was adopted after the single-run bootstrap analysis proved sensitive to an unfixed random seed in GNN training/initialization.

## 

## Key results

| Model | Accuracy | MCC | Significant vs. RF? |
|---|---|---|---|
| Random Forest | 0.726 (fixed) | 0.602 (fixed) | — |
| GCN | 0.729 ± 0.015 | 0.606 ± 0.023 | Not significant |
| **GAT** | **0.741 ± 0.011** | **0.626 ± 0.015** | **Significant, higher** (p < 0.005) |
| GINE | 0.704 ± 0.017 | 0.567 ± 0.025 | Significant, lower |
| GraphSAGE | 0.705 ± 0.009 | 0.568 ± 0.012 | Significant, lower |

*(mean ± std across 10 independently seeded training runs; RF is deterministic)*

GAT additionally shows significant F1 gains over RF specifically on the **AI** (IDO-selective) and **IA** (TDO-selective) classes, with no significant loss on any class — see `results/FINAL_f1_summary_all_architectures.csv` and `figures/Figure3_F1_heatmap_key.png` for the full breakdown.

## Requirements

```
python >= 3.12
numpy == 1.26.4
pandas == 2.2.2
scikit-learn
torch
torch_geometric
rdkit
scipy
matplotlib
seaborn
```

## How to reproduce

1. Run the notebooks inside the 'chemical_space' folder to retrieve and clean the ChEMBL36 dataset (requires database access) and analysis of the chemical space.
2. Run the notebooks inside the 'transductive' folder to build the similarity graph and confirm the best node feature / edge weight combination (Morgan fingerprints + Tanimoto).
3. Run `RF_GCN.ipynb`, `RF_GAT.ipynb`, `RF_GINE.ipynb`, and `RF_GSAGE.ipynb` in the 'inductive' folder— each trains Random Forest and one GNN architecture, then runs the 10-seed multi-run cells appended at the end (this step is the most compute-intensive; each notebook trains its GNN 10 times).
4. Run `ttest_analysis_multirun_vs_RF.ipynb`, inside the 'inductive' folder pointing it at the `*_overall_multirun.csv` / `*_per_class_multirun.csv` files produced in step 3, to reproduce the final statistical comparison.
5. Run `RF_vs_GNN_Figures.ipynb` inside the 'figures' folder to regenerate the final figures from the t-test results.
6. All the output results as CSV files are in the 'results' folder. 

## Citation

If you use this code or these results, please cite:

> [Author names]. "[Manuscript title]." *[Journal name]*, [year]. DOI: [pending]

## License

None.

## Contact

[Hamid Irannejad / irannejadhamid@gmail.com / Drug Design and Development Research Center, Tehran, Iran]
