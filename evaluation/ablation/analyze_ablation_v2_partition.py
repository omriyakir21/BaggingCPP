"""Table 2 (the ablation) on the shipped partition, with the cluster sampler.

Every row here is trained on the same data (the public repo partition, 450,430
sequences, 1,147 positives), with the same negative sampling (whole-cluster
groups drawn from unlabeled sequences only, at fraction 0.3) and the same LoRA
settings where LoRA applies (r=10, alpha=8), as described in Methods 4.3.

Reports, for every variant:
  * pooled AUC over the five test folds, as Table 2 does
  * per-fold AUCs -- these matter. Two models can have identical per-fold AUCs
    and still differ when pooled, in which case the whole gap is cross-fold
    calibration rather than better ranking within a fold. That is invisible
    unless per-fold numbers are printed, so they are printed here.
  * the spread across submodels binned by mean prediction, which is the
    evidence for the ensemble-diversity argument
  * DeLong against the reference model

The CNN rows are read from the cpp repository, because BaggingCPP has no
convolution training function; everything else is local. Set CPP_REPO to its
location (default: a `cpp` folder next to this repository).
"""

import os
import sys
import glob
import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

BAGGING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CPP = os.environ.get('CPP_REPO', os.path.join(os.path.dirname(BAGGING), 'cpp'))
sys.path.append(BAGGING)
from evaluation.utils_evaluation import delong_test  # noqa: E402

N_FOLDS, N_SUBMODELS = 5, 50
OUT = os.path.join(BAGGING, 'assets/ablation_v2_partition')

ABL = 'results/hypothesis/ensemble_ablation_v2_partition'
# label -> (repo, ensemble dir, single dir)
VARIANTS = {
    'CNN':                 (CPP, 'results/hypothesis/ensemble_cnn_v2_partition/cnn',
                            'results/hypothesis/ensemble_cnn_v2_partition/single_cnn'),
    'frozen + MLP':        (BAGGING, f'{ABL}/frozen_MLP', f'{ABL}/single_frozen_MLP'),
    'frozen + LA':         (BAGGING, f'{ABL}/frozen_LA', f'{ABL}/single_frozen_LA'),
    'LoRA + MLP':          (BAGGING, f'{ABL}/lora_MLP', f'{ABL}/single_lora_MLP'),
    'LoRA + LA':           (BAGGING,
                            'results/hypothesis/ensemble_inductive_pu_learning_negative_fractions/fraction_0.3',
                            f'{ABL}/single_lora_LA'),
    'LoRA + LA (sequence sampling)': (BAGGING, f'{ABL}/lora_LA_sequence', f'{ABL}/single_lora_LA'),
}
REFERENCE = 'LoRA + LA'


def latest_with(path_glob):
    found = None
    for p in sorted(glob.glob(path_glob)):
        found = p
    return found


def load_ensemble(repo, rel):
    """(n_submodels, n_seq) of predictions per fold, plus labels per fold."""
    per_fold, labels = [], []
    for fold in range(N_FOLDS):
        preds, lab = [], None
        for i in range(N_SUBMODELS):
            p = latest_with(os.path.join(repo, rel, f'submodel_{i}', '*',
                                         f'fold_{fold}', 'test', 'predictions.npy'))
            if p is None:
                return None, None
            preds.append(np.asarray(np.load(p)).reshape(-1))
            lab = np.load(os.path.join(os.path.dirname(p), 'labels.npy')).reshape(-1)
        per_fold.append(np.stack(preds))
        labels.append(lab)
    return per_fold, labels


def load_single(repo, rel):
    preds, labels = [], []
    for fold in range(N_FOLDS):
        p = latest_with(os.path.join(repo, rel, '*', f'fold_{fold}', 'test', 'predictions.npy'))
        if p is None:
            return None, None
        preds.append(np.asarray(np.load(p)).reshape(-1))
        labels.append(np.load(os.path.join(os.path.dirname(p), 'labels.npy')).reshape(-1))
    return preds, labels


def pooled_and_per_fold(per_fold_preds, labels):
    mean = np.concatenate([p.mean(axis=0) if p.ndim == 2 else p for p in per_fold_preds])
    lab = np.concatenate(labels)
    per_fold = [roc_auc_score(l, p.mean(axis=0) if p.ndim == 2 else p)
                for p, l in zip(per_fold_preds, labels)]
    return mean, lab, float(roc_auc_score(lab, mean)), [float(x) for x in per_fold]


def binned_spread(per_fold, n_bins=20, min_count=200):
    mean = np.concatenate([p.mean(axis=0) for p in per_fold])
    std = np.concatenate([p.std(axis=0) for p in per_fold])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(mean, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_count:
            rows.append((float(0.5 * (edges[b] + edges[b + 1])), float(std[m].mean()), int(m.sum())))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    results, ens_cache = {}, {}

    for label, (repo, ens_rel, single_rel) in VARIANTS.items():
        row = {'label': label}
        per_fold, labels = load_ensemble(repo, ens_rel)
        if per_fold is None:
            row['ensemble'] = None
            print(f'{label:32s} ensemble INCOMPLETE')
        else:
            mean, lab, auc, folds = pooled_and_per_fold(per_fold, labels)
            row['ensemble'] = {'auc': auc, 'per_fold': folds,
                               'per_fold_mean': float(np.mean(folds)),
                               'pr_auc': float(average_precision_score(lab, mean))}
            ens_cache[label] = (mean, lab, per_fold)
            row['spread'] = binned_spread(per_fold)

        sp, sl = load_single(repo, single_rel)
        if sp is None:
            row['single'] = None
        else:
            _, _, sauc, sfolds = pooled_and_per_fold(sp, sl)
            row['single'] = {'auc': sauc, 'per_fold': sfolds,
                             'per_fold_mean': float(np.mean(sfolds))}
        results[label] = row

    print()
    print('=' * 100)
    print('TABLE 2  (public partition, cluster sampler, LoRA r=10 alpha=8)')
    print('=' * 100)
    print(f"{'Variant':32s} {'Single':>8s} {'Ensemble':>9s} {'Gain':>8s} {'PR-AUC':>8s}  per-fold ensemble AUC")
    for label, row in results.items():
        s = row['single']['auc'] if row['single'] else None
        e = row['ensemble']['auc'] if row['ensemble'] else None
        gain = f'{e - s:+.4f}' if (s is not None and e is not None) else '   -'
        folds = ' '.join(f'{x:.4f}' for x in row['ensemble']['per_fold']) if row['ensemble'] else ''
        print(f"{label:32s} {('%.4f' % s) if s else '    -':>8s} {('%.4f' % e) if e else '    -':>9s} "
              f"{gain:>8s} {('%.3f' % row['ensemble']['pr_auc']) if row['ensemble'] else '  -':>8s}  {folds}")

    print()
    print('per-fold MEAN vs POOLED (a gap here means cross-fold calibration, not ranking):')
    for label, row in results.items():
        if row['ensemble']:
            pf, po = row['ensemble']['per_fold_mean'], row['ensemble']['auc']
            print(f'  {label:32s} per-fold mean {pf:.5f}   pooled {po:.5f}   diff {po - pf:+.5f}')

    if REFERENCE in ens_cache:
        print()
        print(f'DeLong against {REFERENCE} (pooled predictions):')
        mean_a, lab_a, _ = ens_cache[REFERENCE]
        for label, (mean_b, lab_b, _) in ens_cache.items():
            if label == REFERENCE:
                continue
            if not np.array_equal(lab_a, lab_b):
                print(f'  {label:32s} SKIPPED, label order differs')
                continue
            r = delong_test(mean_a, mean_b, lab_a)
            print(f"  {label:32s} dAUC {r['auc_a'] - r['auc_b']:+.4f}   p = {r.get('p_value', float('nan')):.3g}")
            results[label]['delong_vs_reference'] = {
                k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}

    print()
    print('SPREAD ACROSS SUBMODELS at fixed mean prediction (the diversity evidence):')
    targets = [0.275, 0.525, 0.725]
    labels_with = [l for l in results if results[l].get('spread')]
    print(f"{'bin':>6s}" + ''.join(f'{l[:22]:>24s}' for l in labels_with))
    for t in targets:
        line = f'{t:6.3f}'
        for l in labels_with:
            rows = results[l]['spread']
            v = min(rows, key=lambda r: abs(r[0] - t))
            line += f'{v[1]:24.3f}'
        print(line)
    print(f"{'ceiling':>6s}" + ''.join(f'{np.sqrt(t * (1 - t)):24.3f}' for t in targets[:1]) + '   (at p=0.275)')

    with open(os.path.join(OUT, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nwrote {OUT}/results.json')


if __name__ == '__main__':
    main()
