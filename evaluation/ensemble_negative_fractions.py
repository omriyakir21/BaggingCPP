"""Inductive PU bagging at several negative sampling fractions.

For every fraction, loads the out-of-fold test predictions of the 50 submodels and reports:
  * the 50-model ensemble: pooled AUC-ROC and PR-AUC, F1 / accuracy / MCC at 0.5, per-fold AUC;
  * the ensemble size curve: AUC of random K-model subsets of the 50, for K = 1 .. 50;
  * the training cost: GPU time per submodel, read from the slurm logs;
  * each run against fraction 0.3: DeLong's test on the pooled predictions, and the folds won.

usage: python -m evaluation.ensemble_negative_fractions [--output_dir ...]
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from evaluation.utils_evaluation import (
    calculate_precision_recall_auc,
    calculate_roc_auc,
    delong_test,
    gather_ensemble_experiments_base_paths,
)

HYPOTHESIS = 'ensemble_inductive_pu_learning_negative_fractions'
FRACTIONS = ['0.1', '0.3', '0.6', '1.0']
# The earlier run at 0.3, trained on groups that also contained positives.
LEAKY_REFERENCE = ('ensemble_inductive_pu_learning', 'groups_inductive')
NUM_SUBMODELS = 50
NUM_FOLDS = 5
ENSEMBLE_SIZES = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SUBSETS_PER_SIZE = 30
# Every run is compared against the fraction the paper uses.
REFERENCE_RUN = 'fraction_0.3'


def load_submodel_predictions(hypothesis, experiment, results_root):
    """Return per-fold arrays of shape (num_submodels, num_sequences_in_fold), and the fold labels."""
    base_paths, missing = gather_ensemble_experiments_base_paths(
        hypothesis, [experiment], it=NUM_SUBMODELS, results_root=results_root)
    base_paths = base_paths[experiment]
    if missing or len(base_paths) < NUM_SUBMODELS:
        raise FileNotFoundError(f'{hypothesis}/{experiment}: unfinished submodels {missing}')
    fold_predictions, fold_labels = [], []
    for fold in range(NUM_FOLDS):
        predictions = []
        for base_path in base_paths:
            test_dir = os.path.join(base_path, f'fold_{fold}', 'test')
            predictions.append(np.load(os.path.join(test_dir, 'predictions.npy')).reshape(-1))
            labels = np.load(os.path.join(test_dir, 'labels.npy')).reshape(-1)
        fold_predictions.append(np.stack(predictions))
        fold_labels.append(labels)
    return fold_predictions, fold_labels


def ensemble_metrics(fold_predictions, fold_labels, submodel_indices):
    fold_means = [predictions[submodel_indices].mean(axis=0) for predictions in fold_predictions]
    y_true, y_score = np.concatenate(fold_labels), np.concatenate(fold_means)
    y_pred = (y_score > 0.5).astype(int)
    per_fold_auc = [calculate_roc_auc(y, m) for y, m in zip(fold_labels, fold_means)]
    return {
        'auc_roc': calculate_roc_auc(y_true, y_score),
        'pr_auc': calculate_precision_recall_auc(y_true, y_score),
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'per_fold_auc_mean': float(np.mean(per_fold_auc)),
        'per_fold_auc_sd': float(np.std(per_fold_auc, ddof=1)),
        **{f'fold_{i}_auc': a for i, a in enumerate(per_fold_auc)},
    }


def ensemble_size_curve(fold_predictions, fold_labels, rng):
    rows = []
    for size in ENSEMBLE_SIZES:
        repeats = 1 if size == NUM_SUBMODELS else SUBSETS_PER_SIZE
        aucs = [ensemble_metrics(fold_predictions, fold_labels,
                                 rng.choice(NUM_SUBMODELS, size, replace=False))['auc_roc']
                for _ in range(repeats)]
        rows.append({'ensemble_size': size, 'auc_mean': np.mean(aucs),
                     'auc_sd': np.std(aucs, ddof=1) if repeats > 1 else np.nan,
                     'auc_min': np.min(aucs), 'auc_max': np.max(aucs), 'subsets': repeats})
    return pd.DataFrame(rows)


def training_hours_per_submodel(log_dir):
    """GPU hours per finished submodel, keyed by fraction, from the 'Training time' lines of the job logs."""
    hours_by_experiment = {}
    for log_path in sorted(glob.glob(os.path.join(log_dir, '**', 'logs.out'), recursive=True)):
        with open(log_path) as handle:
            text = handle.read()
        match = re.search(r'(fraction_([\d.]+)_index_\d+)\.json', text)
        seconds = [float(s) for s in re.findall(r'Training time: ([\d.]+) seconds', text)]
        if match and len(seconds) == NUM_FOLDS:
            hours_by_experiment[match.group(1)] = (match.group(2), sum(seconds) / 3600)
    hours = {fraction: [] for fraction in FRACTIONS}
    for fraction, run_hours in hours_by_experiment.values():
        if fraction in hours:
            hours[fraction].append(run_hours)
    return hours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', default='results/hypothesis')
    parser.add_argument('--log_dir', default=f'slurm/train/configurations/data/{HYPOTHESIS}')
    parser.add_argument('--output_dir', default='assets/negative_fractions')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    runs = [(f'fraction_{fraction}', HYPOTHESIS, f'fraction_{fraction}') for fraction in FRACTIONS]
    runs.append(('leaky_fraction_0.3', *LEAKY_REFERENCE))
    hours = training_hours_per_submodel(args.log_dir)

    summary_rows, curves, pooled = [], [], {}
    for name, hypothesis, experiment in runs:
        try:
            fold_predictions, fold_labels = load_submodel_predictions(hypothesis, experiment, args.results_root)
        except FileNotFoundError as error:
            print(f'skipping {name}: {error}')
            continue
        metrics = ensemble_metrics(fold_predictions, fold_labels, np.arange(NUM_SUBMODELS))
        pooled[name] = (np.concatenate(fold_labels),
                        np.concatenate([predictions.mean(axis=0) for predictions in fold_predictions]),
                        [metrics[f'fold_{i}_auc'] for i in range(NUM_FOLDS)])
        fraction = name.split('_')[-1]
        run_hours = hours.get(fraction, []) if not name.startswith('leaky') else []
        summary_rows.append({'run': name, 'hypothesis': hypothesis, 'experiment': experiment, **metrics,
                             'gpu_hours_per_submodel': np.mean(run_hours) if run_hours else np.nan,
                             'submodels_timed': len(run_hours)})
        curve = ensemble_size_curve(fold_predictions, fold_labels, rng)
        curve.insert(0, 'run', name)
        curves.append(curve)
        print(f'{name}: AUC {metrics["auc_roc"]:.4f}, PR-AUC {metrics["pr_auc"]:.4f}, '
              f'MCC {metrics["mcc"]:.3f}, per-fold AUC {metrics["per_fold_auc_mean"]:.4f} '
              f'± {metrics["per_fold_auc_sd"]:.4f}')

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)

    comparisons = []
    if REFERENCE_RUN in pooled:
        y_reference, score_reference, folds_reference = pooled[REFERENCE_RUN]
        for name, (y_true, y_score, fold_aucs) in pooled.items():
            if name == REFERENCE_RUN:
                continue
            assert np.array_equal(y_true, y_reference), f'{name} is not scored on the same sequences'
            test = delong_test(y_score, score_reference, y_true)
            fold_differences = np.asarray(fold_aucs) - np.asarray(folds_reference)
            comparisons.append({'run': name, 'reference': REFERENCE_RUN, **test,
                                'mean_fold_difference': fold_differences.mean(),
                                'folds_won': int((fold_differences > 0).sum())})
            print(f'{name} - {REFERENCE_RUN}: dAUC {test["difference"]:+.4f} '
                  f'[{test["ci_low"]:+.4f}, {test["ci_high"]:+.4f}], DeLong p={test["p_value"]:.2g}, '
                  f'folds won {comparisons[-1]["folds_won"]}/{NUM_FOLDS}')
        pd.DataFrame(comparisons).to_csv(os.path.join(args.output_dir, 'delong_vs_fraction_0.3.csv'), index=False)
    if curves:
        pd.concat(curves).to_csv(os.path.join(args.output_dir, 'ensemble_size_curve.csv'), index=False)
    print(f'saved to {args.output_dir}')


if __name__ == '__main__':
    main()
