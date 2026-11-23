"""Two controls on the hidden-positive recovery experiment.

**1. Is the result an artefact of homology?** The transductive ensemble is
trained on the merged library, so for a hidden positive it may have seen
homologous *labelled* positives that an inductive model never gets. This script
measures that exposure directly: for each of the 28, the highest identity to a
labelled positive that was actually in the training split of the fold that
scored it. In the inductive setting the folds come from MMseqs2 clusters, so
homologs should be held out together; this checks whether that worked.

**2. Does a model trained on CPP1708 recover them too?** If it does, the
recovery is evidence that CPPs look like CPPs, not evidence for PU learning on
the library. This reads the CPP1708-trained ensemble's predictions over our
library from CPP1708_PREDICTIONS.

Nothing here needs a GPU.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

BAGGING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BAGGING)
import utils  # noqa: E402

RECOVERY = os.path.join(BAGGING, 'assets/hidden_positive_recovery')
HIDDEN = os.path.join(RECOVERY, 'hidden_positives_scores.csv')
NOVELTY = os.path.join(RECOVERY, 'hidden_positives_novelty.csv')
CPP1708_PREDICTIONS = os.path.join(BAGGING, 'assets/cross_dataset/combined_predictions.csv')
FOLDS = os.path.join(BAGGING, 'datasets/data_for_training/folds_training_dicts.pkl')
OUT = os.path.join(RECOVERY, 'controls')

N_FOLDS = 5


import biotite.sequence as bseq  # noqa: E402
import biotite.sequence.align as balign  # noqa: E402

MATRIX = balign.SubstitutionMatrix.std_protein_matrix()
STD = set('ACDEFGHIKLMNPQRSTVWY')


def identity(a, b):
    """Global identity, matches / alignment length.

    Same definition as the id_aln column of hidden_positives_novelty.csv, which
    this is compared against -- biotite has no Alignment.get_sequence_identity,
    so the trace is walked directly.
    """
    alignment = balign.align_optimal(bseq.ProteinSequence(a), bseq.ProteinSequence(b),
                                     MATRIX, local=False, max_number=1)[0]
    trace = alignment.trace
    both = (trace[:, 0] >= 0) & (trace[:, 1] >= 0)
    s1 = np.frombuffer(str(alignment.sequences[0]).encode(), dtype='S1')
    s2 = np.frombuffer(str(alignment.sequences[1]).encode(), dtype='S1')
    return float((s1[trace[both, 0]] == s2[trace[both, 1]]).sum()) / len(trace)


def homology_exposure(hidden):
    """Highest identity to a labelled positive the scoring model trained on.

    A sequence is scored by the fold in which it is in the test split, so the
    positives it could have learned from are the positives of that same fold's
    *training* split.
    """
    folds = utils.load_as_pickle(FOLDS)
    rows = []
    for seq in hidden:
        for f in range(N_FOLDS):
            test_sequences = folds[f]['sequences_test']
            if seq not in test_sequences:
                continue
            train_sequences = np.asarray(folds[f]['sequences_train'], dtype=object)
            train_labels = np.asarray(folds[f]['labels_train'])
            train_positives = train_sequences[train_labels == 1]
            best, closest = 0.0, None
            for p in train_positives:
                p = str(p).strip().upper()
                if not p or not set(p) <= STD:
                    continue  # biotite's protein alphabet rejects non-standard residues
                v = identity(seq, p)
                if v > best:
                    best, closest = v, p
            rows.append({'sequence': seq, 'scoring_fold': f,
                         'max_identity_to_trained_positives': best,
                         'closest_trained_positive': closest})
            break
    return pd.DataFrame(rows)


def recovery(scores, is_hidden, tag):
    y = is_hidden.astype(int)
    hid, rest = scores[y == 1], scores[y == 0]
    above_r = float((rest > 0.5).mean())
    return {
        'model': tag,
        'n_hidden': int(y.sum()),
        'n_background': int((y == 0).sum()),
        'auc': float(roc_auc_score(y, scores)),
        'mannwhitney_p': float(mannwhitneyu(hid, rest, alternative='greater').pvalue),
        'frac_hidden_above_0.5': float((hid > 0.5).mean()),
        'frac_background_above_0.5': above_r,
        'enrichment': float((hid > 0.5).mean() / above_r) if above_r else float('nan'),
        'median_percentile': float(np.median([(rest < s).mean() for s in hid])),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    hidden = list(pd.read_csv(HIDDEN)['sequence'])
    print(f'{len(hidden)} hidden positives\n')

    print('=' * 78)
    print('CONTROL 2: a model trained on CPP1708, scored on the same 28')
    print('=' * 78)
    cpp1708 = pd.read_csv(CPP1708_PREDICTIONS)
    unlabeled = cpp1708[cpp1708['label'] == 0].reset_index(drop=True)
    is_hidden = unlabeled['sequence'].isin(hidden).to_numpy()
    print(f'{len(unlabeled)} unlabeled rows, {is_hidden.sum()} of the 28 found in them')
    row = recovery(unlabeled['prediction'].to_numpy(), is_hidden, 'CPP1708-trained ensemble')
    print(pd.DataFrame([row]).to_string(index=False))

    # Contamination: is any of the 28 a CPP1708 training positive?
    positives_1708 = set(cpp1708[cpp1708['label'] == 1]['sequence'])
    overlap = [s for s in hidden if s in positives_1708]
    print(f'\nof the 28, present among the labelled positives of this table: {len(overlap)}')

    print()
    print('=' * 78)
    print('CONTROL 1: how much homology did the scoring model actually see?')
    print('=' * 78)
    exposure = homology_exposure(hidden)
    if NOVELTY and os.path.exists(NOVELTY):
        nov = pd.read_csv(NOVELTY)[['sequence', 'id_aln']].rename(
            columns={'id_aln': 'identity_to_any_positive'})
        exposure = exposure.merge(nov, on='sequence', how='left')
    exposure = exposure.sort_values('max_identity_to_trained_positives', ascending=False)
    print(exposure.to_string(index=False))
    print()
    print(f"median identity to positives the scoring fold trained on: "
          f"{exposure['max_identity_to_trained_positives'].median():.3f}")
    print(f"above 0.6: {(exposure['max_identity_to_trained_positives'] > 0.6).sum()} of "
          f"{len(exposure)}")

    pd.DataFrame([row]).to_csv(os.path.join(OUT, 'cpp1708_recovery.csv'), index=False)
    exposure.to_csv(os.path.join(OUT, 'homology_exposure.csv'), index=False)
    print(f'\nwrote {OUT}/')


if __name__ == '__main__':
    main()
