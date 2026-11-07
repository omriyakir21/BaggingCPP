"""All pairwise DeLong tests between the Table 2 rows, ensembles and singles.

`analyze_ablation_v2_partition.py` only tests every row against LoRA+LA. To
isolate the specific contribution of LA pooling we need the two main effects
of the 2x2 as well -- frozen+MLP against frozen+LA (LA on a frozen backbone)
and frozen+MLP against LoRA+MLP (LoRA with an MLP head) -- so this prints the
whole matrix. The single models get the same
matrix, because LA does not help them in the same way it helps the ensembles.
"""

import os
import sys
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from analyze_ablation_v2_partition import (  # noqa: E402
    VARIANTS, delong_test, load_ensemble, load_single, pooled_and_per_fold)


def print_matrix(kind):
    pooled = {}
    for label, (repo, ens_rel, single_rel) in VARIANTS.items():
        if kind == 'single' and label == 'LoRA + LA (sequence sampling)':
            continue  # same single model as LoRA + LA
        if kind == 'ensemble':
            per_fold, labels = load_ensemble(repo, ens_rel)
        else:
            per_fold, labels = load_single(repo, single_rel)
        if per_fold is None:
            continue
        mean, lab, auc, folds = pooled_and_per_fold(per_fold, labels)
        pooled[label] = (mean, lab, auc, np.array(folds))

    print(f'\n{kind}')
    print(f"{'A':32s} {'B':32s} {'AUC A':>8s} {'AUC B':>8s} {'dAUC':>9s} {'p':>12s} {'folds A>B':>10s}")
    for a, b in itertools.combinations(pooled, 2):
        mean_a, lab_a, auc_a, folds_a = pooled[a]
        mean_b, lab_b, auc_b, folds_b = pooled[b]
        if not np.array_equal(lab_a, lab_b):
            print(f'{a:32s} {b:32s}  label order differs, skipped')
            continue
        r = delong_test(mean_a, mean_b, lab_a)
        print(f'{a:32s} {b:32s} {auc_a:8.4f} {auc_b:8.4f} '
              f'{auc_a - auc_b:+9.4f} {r["p_value"]:12.3g} {int((folds_a > folds_b).sum()):>8d}/5')


def main():
    for kind in ('ensemble', 'single'):
        print_matrix(kind)


if __name__ == '__main__':
    main()
