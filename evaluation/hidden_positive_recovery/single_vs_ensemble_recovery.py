"""Does bagging, or only PU learning, recover the hidden positives?

The hidden-positive recovery experiment scores the ensemble only, so it shows
that the pipeline as a whole can find positives hidden in the unlabeled set,
but not whether PU learning or bagging does the work.

This runs the identical measurement on a single model and on the ensemble, both
trained on the same partition with the same architecture and LoRA setting, so
the only difference between them is bagging.

The 28 hidden positives are the CPPs that are new in CPPsite 3.0 and present in
our unlabeled library. They carry label 0 here, exactly as they did during
training.
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

BAGGING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECOVERY = os.path.join(BAGGING, 'assets/hidden_positive_recovery')
HIDDEN = os.path.join(RECOVERY, 'hidden_positives_scores.csv')
NOVELTY = os.path.join(RECOVERY, 'hidden_positives_novelty.csv')
OUT = os.path.join(RECOVERY, 'single_vs_ensemble')

ABL = 'results/hypothesis/ensemble_ablation_v2_partition'
ENSEMBLE = 'results/hypothesis/ensemble_inductive_pu_learning_negative_fractions/fraction_0.3'
SINGLE = f'{ABL}/single_lora_LA'
N_FOLDS, N_SUBMODELS = 5, 50


def latest(path_glob):
    found = None
    for p in sorted(glob.glob(path_glob)):
        found = p
    return found


def pooled_single(rel):
    """sequence -> prediction, pooled over the five out-of-fold test sets."""
    seqs, preds, labels = [], [], []
    for fold in range(N_FOLDS):
        p = latest(os.path.join(BAGGING, rel, '*', f'fold_{fold}', 'test', 'predictions.npy'))
        d = os.path.dirname(p)
        preds.append(np.load(p).reshape(-1))
        labels.append(np.load(os.path.join(d, 'labels.npy')).reshape(-1))
        seqs.append(pd.read_pickle(os.path.join(d, 'sequences.pkl')))
    return (np.concatenate([np.asarray(s, dtype=object) for s in seqs]),
            np.concatenate(preds), np.concatenate(labels))


def pooled_ensemble(rel):
    seqs, preds, labels = [], [], []
    for fold in range(N_FOLDS):
        members = []
        for i in range(N_SUBMODELS):
            p = latest(os.path.join(BAGGING, rel, f'submodel_{i}', '*',
                                    f'fold_{fold}', 'test', 'predictions.npy'))
            members.append(np.load(p).reshape(-1))
            if i == 0:
                d = os.path.dirname(p)
                labels.append(np.load(os.path.join(d, 'labels.npy')).reshape(-1))
                seqs.append(pd.read_pickle(os.path.join(d, 'sequences.pkl')))
        preds.append(np.stack(members).mean(axis=0))
    return (np.concatenate([np.asarray(s, dtype=object) for s in seqs]),
            np.concatenate(preds), np.concatenate(labels))


def recovery(seqs, preds, labels, hidden, tag):
    """The 28 hidden positives ranked against the rest of the unlabeled library."""
    unlabeled = labels == 0
    is_hidden = np.isin(seqs, list(hidden)) & unlabeled

    scores = preds[unlabeled]
    y = is_hidden[unlabeled].astype(int)
    auc = roc_auc_score(y, scores)
    u = mannwhitneyu(scores[y == 1], scores[y == 0], alternative='greater')

    hid, rest = scores[y == 1], scores[y == 0]
    above_h = float((hid > 0.5).mean())
    above_r = float((rest > 0.5).mean())
    pct = [float((rest < s).mean()) for s in hid]

    row = {
        'model': tag,
        'n_hidden_found': int(y.sum()),
        'n_background': int((y == 0).sum()),
        'auc': float(auc),
        'mannwhitney_p': float(u.pvalue),
        'frac_hidden_above_0.5': above_h,
        'frac_background_above_0.5': above_r,
        'enrichment': float(above_h / above_r) if above_r else float('nan'),
        'median_percentile': float(np.median(pct)),
        'median_score_hidden': float(np.median(hid)),
        'median_score_background': float(np.median(rest)),
    }
    per_seq = pd.DataFrame({'sequence': seqs[is_hidden], f'{tag}_score': hid,
                            f'{tag}_percentile': pct})
    return row, per_seq


def main():
    os.makedirs(OUT, exist_ok=True)
    hidden = set(pd.read_csv(HIDDEN)['sequence'])
    print(f'{len(hidden)} hidden positives loaded')

    rows, frames = [], []
    for tag, loader, rel in [('single', pooled_single, SINGLE),
                             ('ensemble', pooled_ensemble, ENSEMBLE)]:
        seqs, preds, labels = loader(rel)
        print(f'{tag:9s} pooled {len(seqs)} sequences, {int((labels == 1).sum())} labelled positives')
        row, per_seq = recovery(seqs, preds, labels, hidden, tag)
        rows.append(row)
        frames.append(per_seq.set_index('sequence'))

    summary = pd.DataFrame(rows)
    print()
    print(summary.to_string(index=False))

    joined = frames[0].join(frames[1], how='outer')
    if os.path.exists(NOVELTY):
        nov = pd.read_csv(NOVELTY)
        col = [c for c in nov.columns if 'ident' in c.lower()]
        if col:
            joined = joined.join(nov.set_index('sequence')[col[0]].rename('identity_to_training'))

    # Stratify by distance to the training positives.
    if 'identity_to_training' in joined:
        print()
        print('stratified by global identity to the nearest training positive:')
        for label, mask in [('all', joined.index == joined.index),
                            ('identity >= 0.6', joined['identity_to_training'] >= 0.6),
                            ('identity < 0.6', joined['identity_to_training'] < 0.6)]:
            sub = joined[mask]
            print(f"  {label:18s} n={len(sub):3d}  "
                  f"single median pct {sub['single_percentile'].median():.4f}   "
                  f"ensemble median pct {sub['ensemble_percentile'].median():.4f}")

    joined.sort_values('ensemble_score', ascending=False).to_csv(
        os.path.join(OUT, 'per_peptide.csv'))
    summary.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nwrote {OUT}/')


if __name__ == '__main__':
    main()
