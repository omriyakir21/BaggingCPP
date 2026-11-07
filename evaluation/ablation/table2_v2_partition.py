"""Table 2 of the paper, on the shipped partition with the cluster sampler.

Columns as in the paper: base model, fine-tune, head, type, trainable params,
train time, AUC-ROC.

  * AUC-ROC comes from results.json, written by analyze_ablation_v2_partition.py
    (pooled over the five test folds).
  * Trainable params are counted by building each model from the configuration
    the run actually used. They are per fold; the ensemble is 50 submodels.
  * Train time is the sum of the 'Training time' lines in the slurm job logs,
    over all five folds (and all 50 submodels for an ensemble). These are summed
    GPU hours on mixed cluster nodes, not wall-clock time. Same source as the
    GPU-hour column of the negative-fraction sweep
    (evaluation/ensemble_negative_fractions.py).

The CNN rows come from the cpp repository: set CPP_REPO to its location
(default: a `cpp` folder next to this repository).
"""

import os
import re
import sys
import glob
import json

import pandas as pd
import torch

BAGGING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CPP = os.environ.get('CPP_REPO', os.path.join(os.path.dirname(BAGGING), 'cpp'))
sys.path.insert(0, BAGGING)
from bootstrappers.model import build_model_from_configuration  # noqa: E402
from models.Convolution import CNNModel  # noqa: E402

N_FOLDS, N_SUBMODELS = 5, 50
OUT = os.path.join(BAGGING, 'assets/ablation_v2_partition')

ABL = 'ensemble_ablation_v2_partition'
FRACTIONS = 'ensemble_inductive_pu_learning_negative_fractions'
CNN = 'ensemble_cnn_v2_partition'
# (base, fine-tune, head, type, results.json key, repo, hypothesis, slurm config regex)
ROWS = [
    ('CNN', '–', 'MLP', 'Single', 'CNN', CPP, CNN, r'single_cnn\.json'),
    ('CNN', '–', 'MLP', 'Ensemble', 'CNN', CPP, CNN, r'cnn_index_\d+\.json'),
    ('ESM2', 'Frozen', 'MLP', 'Single', 'frozen + MLP', BAGGING, ABL, r'single_frozen_MLP\.json'),
    ('ESM2', 'Frozen', 'MLP', 'Ensemble', 'frozen + MLP', BAGGING, ABL, r'frozen_MLP_index_\d+\.json'),
    ('ESM2', 'Frozen', 'LA', 'Single', 'frozen + LA', BAGGING, ABL, r'single_frozen_LA\.json'),
    ('ESM2', 'Frozen', 'LA', 'Ensemble', 'frozen + LA', BAGGING, ABL, r'frozen_LA_index_\d+\.json'),
    ('ESM2', 'LoRA', 'MLP', 'Single', 'LoRA + MLP', BAGGING, ABL, r'single_lora_MLP\.json'),
    ('ESM2', 'LoRA', 'MLP', 'Ensemble', 'LoRA + MLP', BAGGING, ABL, r'lora_MLP_index_\d+\.json'),
    ('ESM2', 'LoRA', 'LA', 'Single', 'LoRA + LA', BAGGING, ABL, r'single_lora_LA\.json'),
    ('ESM2', 'LoRA', 'LA', 'Ensemble', 'LoRA + LA', BAGGING, FRACTIONS, r'fraction_0\.3_index_\d+\.json'),
]


def training_seconds(repo, hypothesis, config_regex):
    """Sum of 'Training time' over the job log that finished all five folds, per config."""
    base = os.path.join(repo, 'slurm/train/configurations/data', hypothesis)
    configs = [c for c in sorted(os.listdir(base)) if re.fullmatch(config_regex, c)]
    total = 0.0
    for config in configs:
        finished = None
        for log in sorted(glob.glob(os.path.join(base, config, '**', 'logs.out'), recursive=True)):
            with open(log) as handle:
                seconds = [float(s) for s in re.findall(r'Training time: ([\d.]+) seconds', handle.read())]
            if len(seconds) == N_FOLDS:
                finished = seconds
        if finished is None:
            raise RuntimeError(f'{config}: no job log with {N_FOLDS} finished folds')
        total += sum(finished)
    return total, len(configs)


def trainable_params(repo, hypothesis, config_regex):
    """Build the model from the run's own configuration and count requires_grad params."""
    config_name = sorted(c for c in os.listdir(os.path.join(repo, 'configurations/data', hypothesis))
                         if re.fullmatch(config_regex, c))[0]
    with open(os.path.join(repo, 'configurations/data', hypothesis, config_name)) as handle:
        model_cfg = json.load(handle)['model']
    if model_cfg['name'] == 'convolution':
        kwargs = {k: model_cfg['kwargs'][k] for k in ('filters', 'kernel_size', 'num_layers', 'padding', 'dilation')}
        model = CNNModel(**kwargs)
    else:
        model = build_model_from_configuration(model_cfg['name'], torch.device('cpu'),
                                               torch.nn.BCEWithLogitsLoss(), 0, model_cfg['kwargs'])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    hours = seconds / 3600
    if hours < 1:
        return f'∼{seconds / 60:.0f} min'
    if hours < 10:
        return f'∼{hours:.1f} h'
    return f'∼{hours:.0f} h'


def main():
    os.chdir(BAGGING)
    with open(os.path.join(OUT, 'results.json')) as handle:
        aucs = json.load(handle)

    rows = []
    for base, fine_tune, head, kind, key, repo, hypothesis, regex in ROWS:
        seconds, n_configs = training_seconds(repo, hypothesis, regex)
        assert n_configs == (1 if kind == 'Single' else N_SUBMODELS), (key, kind, n_configs)
        params = trainable_params(repo, hypothesis, regex)
        rows.append({'Base model': base, 'Fine-tune': fine_tune, 'Head': head, 'Type': kind,
                     'Trainable params': params * (N_SUBMODELS if kind == 'Ensemble' else 1),
                     'Train time (GPU h)': seconds / 3600,
                     'Train time': format_time(seconds),
                     'AUC-ROC': aucs[key][kind.lower()]['auc']})
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, 'table2.csv'), index=False)
    printed = table.drop(columns='Train time (GPU h)').assign(
        **{'Trainable params': table['Trainable params'].map('{:,}'.format),
           'AUC-ROC': table['AUC-ROC'].map('{:.3f}'.format)})
    lines = ['| ' + ' | '.join(printed.columns) + ' |',
             '|' + '---|' * len(printed.columns)]
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in printed.itertuples(index=False)]
    with open(os.path.join(OUT, 'table2.md'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwrote {OUT}/table2.csv and table2.md')


if __name__ == '__main__':
    main()
