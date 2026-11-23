"""Significance of every comparison in Table 2 that answers a question.

`delong_matrix.py` prints all pairs within one type (single or ensemble). This
script instead picks the comparisons that each answer a question -- does bagging
help, does LA help, does LoRA help, does pretraining help, is the final model
the best, does cluster sampling help -- including single against ensemble, which
the matrix never tests. It also tests two interactions, which a pairwise test
cannot check:

  * LA x LoRA: is LA worth less once LoRA is present ("LA and LoRA are
    substitutes")?
  * LoRA x bagging: does a LoRA model gain more from bagging than a frozen one
    ("fine-tuning buys about four times more from bagging")?

All are DeLong tests on the predictions pooled over the five test folds. Every
model is scored on the same 450,430 sequences, so one DeLong covariance matrix
over all eleven prediction vectors covers every test, interactions included
(as a contrast over four AUCs). p-values are Holm-adjusted over the whole family.
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from analyze_ablation_v2_partition import (  # noqa: E402
    BAGGING, OUT, N_FOLDS, VARIANTS, latest_with, load_ensemble, load_single, pooled_and_per_fold)

sys.path.append(BAGGING)
from evaluation.utils_evaluation import delong_structural_components  # noqa: E402

SHORT = {'CNN': 'CNN', 'frozen + MLP': 'frozen+MLP', 'frozen + LA': 'frozen+LA',
         'LoRA + MLP': 'LoRA+MLP', 'LoRA + LA': 'LoRA+LA',
         'LoRA + LA (sequence sampling)': 'LoRA+LA seq'}

# (question, description, {model: weight}); the AUC contrast is sum(weight * AUC).
# Model keys are '<short name> single' or '<short name> ens'.
def pair(a, b):
    return {a: 1.0, b: -1.0}


COMPARISONS = [
    ('Does bagging help?', 'CNN: ensemble - single', pair('CNN ens', 'CNN single')),
    ('Does bagging help?', 'frozen+MLP: ensemble - single', pair('frozen+MLP ens', 'frozen+MLP single')),
    ('Does bagging help?', 'frozen+LA: ensemble - single', pair('frozen+LA ens', 'frozen+LA single')),
    ('Does bagging help?', 'LoRA+MLP: ensemble - single', pair('LoRA+MLP ens', 'LoRA+MLP single')),
    ('Does bagging help?', 'LoRA+LA: ensemble - single', pair('LoRA+LA ens', 'LoRA+LA single')),

    ('Does LA help?', 'frozen, single: LA - MLP', pair('frozen+LA single', 'frozen+MLP single')),
    ('Does LA help?', 'frozen, ensemble: LA - MLP', pair('frozen+LA ens', 'frozen+MLP ens')),
    ('Does LA help?', 'LoRA, single: LA - MLP', pair('LoRA+LA single', 'LoRA+MLP single')),
    ('Does LA help?', 'LoRA, ensemble: LA - MLP', pair('LoRA+LA ens', 'LoRA+MLP ens')),

    ('Does LoRA help?', 'MLP, single: LoRA - frozen', pair('LoRA+MLP single', 'frozen+MLP single')),
    ('Does LoRA help?', 'MLP, ensemble: LoRA - frozen', pair('LoRA+MLP ens', 'frozen+MLP ens')),
    ('Does LoRA help?', 'LA, single: LoRA - frozen', pair('LoRA+LA single', 'frozen+LA single')),
    ('Does LoRA help?', 'LA, ensemble: LoRA - frozen', pair('LoRA+LA ens', 'frozen+LA ens')),

    ('Does pretraining help?', 'single: LoRA+MLP - CNN', pair('LoRA+MLP single', 'CNN single')),
    ('Does pretraining help?', 'ensemble: LoRA+MLP - CNN', pair('LoRA+MLP ens', 'CNN ens')),

    ('Is the final model the best?', 'LoRA+LA ens - CNN ens', pair('LoRA+LA ens', 'CNN ens')),
    ('Is the final model the best?', 'LoRA+LA ens - frozen+MLP ens', pair('LoRA+LA ens', 'frozen+MLP ens')),
    ('Is the final model the best?', 'LoRA+LA ens - best single (frozen+LA)',
     pair('LoRA+LA ens', 'frozen+LA single')),

    ('Does cluster sampling help?', 'LoRA+LA ens: cluster - sequence', pair('LoRA+LA ens', 'LoRA+LA seq ens')),

    ('Are LA and LoRA substitutes?', 'single: LA gain on frozen - LA gain on LoRA',
     {'frozen+LA single': 1.0, 'frozen+MLP single': -1.0, 'LoRA+LA single': -1.0, 'LoRA+MLP single': 1.0}),
    ('Are LA and LoRA substitutes?', 'ensemble: LA gain on frozen - LA gain on LoRA',
     {'frozen+LA ens': 1.0, 'frozen+MLP ens': -1.0, 'LoRA+LA ens': -1.0, 'LoRA+MLP ens': 1.0}),

    ('Does LoRA gain more from bagging?', 'MLP head: bagging gain LoRA - frozen',
     {'LoRA+MLP ens': 1.0, 'LoRA+MLP single': -1.0, 'frozen+MLP ens': -1.0, 'frozen+MLP single': 1.0}),
    ('Does LoRA gain more from bagging?', 'LA head: bagging gain LoRA - frozen',
     {'LoRA+LA ens': 1.0, 'LoRA+LA single': -1.0, 'frozen+LA ens': -1.0, 'frozen+LA single': 1.0}),
]


def test_sequences(repo, rel, ensemble):
    """Test sequences per fold, in saved order (submodel 0 for an ensemble)."""
    root = os.path.join(repo, rel, 'submodel_0') if ensemble else os.path.join(repo, rel)
    folds = []
    for fold in range(N_FOLDS):
        path = latest_with(os.path.join(root, '*', f'fold_{fold}', 'test', 'sequences.pkl'))
        with open(path, 'rb') as handle:
            folds.append(list(pickle.load(handle)))
    return folds


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    predictions, per_fold_aucs, labels, reference = {}, {}, None, None
    for label, (repo, ens_rel, single_rel) in VARIANTS.items():
        kinds = [('ens', ens_rel, True)]
        if label != 'LoRA + LA (sequence sampling)':  # its single model is LoRA + LA's
            kinds.append(('single', single_rel, False))
        for kind, rel, ensemble in kinds:
            key = f'{SHORT[label]} {kind}'
            per_fold, fold_labels = (load_ensemble if ensemble else load_single)(repo, rel)
            mean, lab, auc, folds = pooled_and_per_fold(per_fold, fold_labels)

            sequences = test_sequences(repo, rel, ensemble)
            if reference is None:
                reference, labels = sequences, lab
            if sequences != reference or not np.array_equal(lab, labels):
                raise RuntimeError(f'{key}: test sequences are not in the reference order')

            predictions[key] = mean
            per_fold_aucs[key] = np.array(folds)
            print(f'{key:20s} AUC {auc:.4f}', flush=True)

    keys = list(predictions)
    aucs, V10, V01 = delong_structural_components(np.vstack([predictions[k] for k in keys]), labels)
    covariance = np.cov(V10) / V10.shape[1] + np.cov(V01) / V01.shape[1]
    index = {k: i for i, k in enumerate(keys)}

    rows = []
    for question, description, weights in COMPARISONS:
        contrast = np.zeros(len(keys))
        for key, weight in weights.items():
            contrast[index[key]] = weight
        difference = float(contrast @ aucs)
        standard_error = float(np.sqrt(contrast @ covariance @ contrast))
        z = difference / standard_error
        fold_differences = sum(w * per_fold_aucs[k] for k, w in weights.items())
        rows.append({'question': question, 'comparison': description,
                     'dAUC': difference,
                     'ci_low': difference - 1.959963985 * standard_error,
                     'ci_high': difference + 1.959963985 * standard_error,
                     'z': z, 'p': float(2 * stats.norm.sf(abs(z))),
                     'folds_positive': int((fold_differences > 0).sum())})

    table = pd.DataFrame(rows)
    table['p_holm'] = holm(table['p'].values)
    table['significant_holm_0.05'] = table['p_holm'] < 0.05
    table.to_csv(os.path.join(OUT, 'significance_all_pairs.csv'), index=False)

    with pd.option_context('display.width', 200, 'display.max_colwidth', 50):
        print(table.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'\nwrote {OUT}/significance_all_pairs.csv')


if __name__ == '__main__':
    main()
