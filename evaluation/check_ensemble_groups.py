"""Check saved ensemble groups: each sampled negative must be an unlabeled training sequence.

usage: python -m evaluation.check_ensemble_groups <groups_dir> [<groups_dir> ...]
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import utils

FOLDS_PATH = 'datasets/data_for_training/folds_training_dicts.pkl'

if __name__ == '__main__':
    folds = utils.load_as_pickle(FOLDS_PATH)
    for groups_dir in sys.argv[1:]:
        for i, fold in enumerate(folds):
            labels = np.asarray(fold['labels_train'])
            unlabeled = np.where(labels == 0)[0]
            groups = utils.load_as_pickle(os.path.join(groups_dir, f'fold_{i}', 'groups_for_ensemble.pkl'))
            positives_in_groups = [int((labels[np.asarray(g)] == 1).sum()) for g in groups]
            draws = np.mean([len(g) for g in groups]) / len(unlabeled)
            distinct = np.mean([len(np.unique(g)) for g in groups]) / len(unlabeled)
            print(f'{groups_dir} fold {i}: groups {len(groups)}, positives in groups {max(positives_in_groups)}, '
                  f'draws {draws:.3f} of unlabeled, distinct {distinct:.3f} of unlabeled')
