"""What the "marginal" AUC differences of Table 2 are worth on a screen.

AUC-ROC moves very little between the ablation rows. On a library with 1,147
positives among 450,430 peptides, AUC-ROC is the wrong scale to judge that
margin on: it averages over every threshold, and almost all of them are
thresholds nobody would ever screen at.

This script reports the quantity an experimentalist actually faces -- how many
real CPPs are in the top-k ranked peptides -- for every row of Table 2, using
the same pooled test predictions as
`analyze_ablation_v2_partition.py`.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from analyze_ablation_v2_partition import (  # noqa: E402
    VARIANTS, load_ensemble, pooled_and_per_fold)

TOP_K = [100, 500, 1000, 5000]


def main():
    rows = []
    for label, (repo, ens_rel, _) in VARIANTS.items():
        per_fold, labels = load_ensemble(repo, ens_rel)
        if per_fold is None:
            continue
        mean, lab, auc, _ = pooled_and_per_fold(per_fold, labels)
        order = np.argsort(-mean)
        ranked = lab[order]
        hits = [int(ranked[:k].sum()) for k in TOP_K]
        # Shortlist length needed to recover half of the positives.
        cum = np.cumsum(ranked)
        half = int(np.searchsorted(cum, lab.sum() / 2.0) + 1)
        rows.append((label, auc, hits, half))

    n_pos = int(lab.sum())
    n_tot = int(lab.size)
    print(f'library: {n_tot} peptides, {n_pos} positives '
          f'({100.0 * n_pos / n_tot:.3f}% prevalence)')
    print()
    header = f"{'Variant':32s} {'AUC':>7s}" + ''.join(f'{"top " + str(k):>12s}' for k in TOP_K)
    print(header + f"{'half@':>10s}")
    for label, auc, hits, half in rows:
        line = f'{label:32s} {auc:7.4f}'
        for k, h in zip(TOP_K, hits):
            line += f'{h:7d} ({100.0 * h / k:3.0f}%)'[:12].rjust(12)
        print(line + f'{half:10d}')

    print()
    print('read as: of the k top-ranked peptides, how many are known CPPs; '
          'half@ is the shortlist length that recovers half the positives.')


if __name__ == '__main__':
    main()
