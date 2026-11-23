"""Figure for the negative-fraction sweep.

A: AUC-ROC of a K-model ensemble against K, one line per negative sampling fraction
   (mean of random K-subsets of the 50 submodels, band = +-1 sd over subsets).
B: AUC-ROC of the 50-model ensemble per fraction: pooled over folds, and the five per-fold
   values; the x labels carry the GPU time of training the 50 submodels.

Reads the CSVs written by evaluation.ensemble_negative_fractions.
usage: python -m evaluation.plot_ensemble_negative_fractions [--input_dir ...]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRACTIONS = ['0.1', '0.3', '0.6', '1.0']
# Categorical slots 1-4 of the reference palette, in fixed order.
COLORS = {'0.1': '#2a78d6', '0.3': '#eb6834', '0.6': '#1baf7a', '1.0': '#eda100'}
LEAKY_COLOR = '#8a8984'
TEXT_PRIMARY, TEXT_SECONDARY, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NUM_SUBMODELS = 50


def style_axis(axis):
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.grid(axis='y', color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='assets/negative_fractions')
    args = parser.parse_args()
    summary = pd.read_csv(os.path.join(args.input_dir, 'summary.csv')).set_index('run')
    curve = pd.read_csv(os.path.join(args.input_dir, 'ensemble_size_curve.csv'))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': TEXT_PRIMARY,
                         'axes.labelcolor': TEXT_SECONDARY})
    figure, (axis_a, axis_b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.5, 1]})

    # A: ensemble size.
    for fraction in FRACTIONS:
        run = curve[curve.run == f'fraction_{fraction}']
        sd = run.auc_sd.fillna(0)
        axis_a.fill_between(run.ensemble_size, run.auc_mean - sd, run.auc_mean + sd,
                            color=COLORS[fraction], alpha=0.12, linewidth=0)
        axis_a.plot(run.ensemble_size, run.auc_mean, color=COLORS[fraction], linewidth=1.8,
                    marker='o', markersize=3.5, label=f'fraction {fraction}')
    # Direct labels at the line ends, pushed apart so close lines stay readable.
    ends = sorted((curve[curve.run == f'fraction_{f}'].auc_mean.iloc[-1], f) for f in FRACTIONS)
    label_y, min_gap = [], 0.002
    for y, _ in ends:
        label_y.append(max(y, label_y[-1] + min_gap) if label_y else y)
    for (_, fraction), y in zip(ends, label_y):
        axis_a.annotate(fraction, (NUM_SUBMODELS + 1, y), va='center', fontsize=9, color=TEXT_PRIMARY)
    leaky = curve[curve.run == 'leaky_fraction_0.3']
    if len(leaky):
        axis_a.plot(leaky.ensemble_size, leaky.auc_mean, color=LEAKY_COLOR, linewidth=1.2,
                    linestyle='--', label='0.3, before the fix')
    axis_a.set_xlabel('Ensemble size (number of submodels)')
    axis_a.set_ylabel('AUC-ROC (5-fold, pooled)')
    axis_a.set_xlim(0, NUM_SUBMODELS + 5)
    axis_a.set_xticks([1, 10, 20, 30, 40, 50])
    axis_a.set_title('A. AUC-ROC by ensemble size', loc='left', fontsize=11, color=TEXT_PRIMARY)
    axis_a.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=TEXT_PRIMARY)
    style_axis(axis_a)

    # B: negative fraction, with training cost.
    positions = np.arange(len(FRACTIONS))
    labels = []
    for position, fraction in zip(positions, FRACTIONS):
        row = summary.loc[f'fraction_{fraction}']
        fold_aucs = [row[f'fold_{i}_auc'] for i in range(5)]
        axis_b.scatter(np.full(5, position) + np.linspace(-0.12, 0.12, 5), fold_aucs, s=14,
                       color=COLORS[fraction], alpha=0.55, linewidths=0, zorder=2)
        axis_b.scatter(position, row.auc_roc, s=70, color=COLORS[fraction], edgecolors='white',
                       linewidths=1.5, zorder=3)
        axis_b.annotate(f'{row.auc_roc:.4f}', (position, row.auc_roc), xytext=(10, 0),
                        textcoords='offset points', va='center', fontsize=8.5, color=TEXT_PRIMARY)
        labels.append(f'{fraction}\n{row.gpu_hours_per_submodel * NUM_SUBMODELS:.0f} GPU-h')
    axis_b.set_xticks(positions)
    axis_b.set_xticklabels(labels)
    axis_b.set_xlim(-0.5, len(FRACTIONS) - 0.3)
    axis_b.set_xlabel('Negative sampling fraction (cost of 50 submodels)')
    axis_b.set_ylabel('AUC-ROC, 50 submodels')
    axis_b.set_title('B. Pooled (large) and per-fold (small) AUC-ROC', loc='left', fontsize=11,
                     color=TEXT_PRIMARY)
    style_axis(axis_b)

    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(os.path.join(args.input_dir, f'negative_fractions.{extension}'), dpi=200,
                       facecolor='white')
    print(f'saved {args.input_dir}/negative_fractions.png and .pdf')


if __name__ == '__main__':
    main()
