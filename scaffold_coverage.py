"""Reproduce the notebooks' scaffold split for all four datasets and quantify
structural separation between train and test (reviewer 1, comment 6)."""
import numpy as np, pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

ROOT = '/home/linu/claude_workdir/GNN_RF_manuscript_2'
RAW = {'IDO1/TDO':      ('IDO_TDO/ido1_tdo_common_1126.csv',      'ido1_ic50', 'tdo_ic50'),
       'COX-1/COX-2':   ('cox1_cox2/cox1_cox2_common_20260312.csv', 'cox1_ic50', 'cox2_ic50'),
       'MAO-A/MAO-B':   ('MAOA_MAOB/maoa_maob_common.csv',        'maoa_ic50', 'maob_ic50'),
       'AChE/BChE':     ('AchE_BchE/ache_bche_common.csv',        'ache_ic50', 'bche_ic50')}
CLASSES = ['AA', 'AI', 'IA', 'II']


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split_indices(indices, smiles_list, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Byte-for-byte reproduction of the notebooks' splitter."""
    indices = np.asarray(indices)
    s2i = defaultdict(list)
    for idx, smi in zip(indices, np.asarray(smiles_list)):
        sc = generate_scaffold(smi)
        if sc is None:
            sc = f'__invalid_{idx}__'
        elif sc == '':
            sc = f'__acyclic_{idx}__'
        s2i[sc].append(idx)
    rng = np.random.default_rng(seed)
    groups = list(s2i.values())
    groups = [groups[i] for i in rng.permutation(len(groups))]
    groups.sort(key=len, reverse=True)
    n = len(indices)
    n_tr, n_va = int(np.floor(train_ratio * n)), int(np.floor(val_ratio * n))
    tr, va, te = [], [], []
    for g in groups:
        if len(tr) + len(g) <= n_tr:
            tr.extend(g)
        elif len(va) + len(g) <= n_va:
            va.extend(g)
        else:
            te.extend(g)
    return np.array(tr), np.array(va), np.array(te), s2i


def fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out


def nn_to_train(fp_test, fp_train):
    vals = []
    ft = [f for f in fp_train if f is not None]
    for f in fp_test:
        if f is None:
            continue
        vals.append(max(DataStructs.BulkTanimotoSimilarity(f, ft)))
    return np.array(vals)


rows, splits_out = [], {}
for ds, (f, c1, c2) in RAW.items():
    df = pd.read_csv(f'{ROOT}/{f}', na_values=['na']).reset_index(drop=True)
    lab = np.where(df[c1] <= 1000, 'A', 'I') + np.where(df[c2] <= 1000, 'A', 'I')
    tr, va, te, s2i = scaffold_split_indices(df.index.to_numpy(), df['smiles'].to_numpy())
    splits_out[ds] = dict(train=tr, val=va, test=te, labels=lab, smiles=df['smiles'].to_numpy())

    grp_of = {i: k for k, v in s2i.items() for i in v}
    spl_of = {}
    for name, idx in (('train', tr), ('val', va), ('test', te)):
        for i in idx:
            spl_of[i] = name
    spanning = sum(1 for k, v in s2i.items() if len({spl_of[i] for i in v}) > 1)

    F = fps(df['smiles'].to_numpy())
    nn_sc = nn_to_train([F[i] for i in te], [F[i] for i in tr])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    r_tr, r_te = perm[:len(tr)], perm[len(tr) + len(va):]
    nn_rd = nn_to_train([F[i] for i in r_te], [F[i] for i in r_tr])

    rows.append(dict(
        Dataset=ds, N=len(df),
        train_groups=len({grp_of[i] for i in tr}), val_groups=len({grp_of[i] for i in va}),
        test_groups=len({grp_of[i] for i in te}),
        train_max_group=max(len(v) for v in s2i.values() if spl_of[v[0]] == 'train'),
        train_mean_group=round(len(tr) / len({grp_of[i] for i in tr}), 2),
        test_singleton_frac=round(np.mean([len(s2i[grp_of[i]]) == 1 for i in te]), 3),
        groups_spanning_splits=spanning,
        nn_scaffold_mean=round(float(nn_sc.mean()), 3),
        nn_random_mean=round(float(nn_rd.mean()), 3),
        nn_scaffold_median=round(float(np.median(nn_sc)), 3),
        nn_frac_above_0p4=round(float((nn_sc > 0.4).mean()), 3),
        nn_max=round(float(nn_sc.max()), 3),
        class_counts={c: int((lab == c).sum()) for c in CLASSES},
        test_class_counts={c: int((lab[te] == c).sum()) for c in CLASSES},
    ))

SCAF = pd.DataFrame(rows)
SCAF.to_csv('out/scaffold_coverage_all_datasets.csv', index=False)
print(SCAF.drop(columns=['class_counts', 'test_class_counts']).to_string(index=False))
print()
for r in rows:
    print(r['Dataset'], '| all:', r['class_counts'], '| test:', r['test_class_counts'])
