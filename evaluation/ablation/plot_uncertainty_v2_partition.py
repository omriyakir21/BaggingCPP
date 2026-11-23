"""Disagreement between ensemble members, against the ensemble's own prediction.

This is the evidence for the ensemble-diversity argument, drawn from the
Table 2 runs (all six ensembles on the shipped partition). It reads the
results.json written by analyze_ablation_v2_partition.py.

For each ensemble: bin the 450,430 test predictions by the mean prediction of
the 50 submodels, and plot the mean standard deviation across submodels inside
each bin. The dashed line is sqrt(p(1-p)), the largest standard deviation any
set of predictions with that mean can have, so the vertical axis is readable as
a fraction of the maximum possible disagreement.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BAGGING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BAGGING, 'assets/ablation_v2_partition')

ORDER = ['CNN', 'LoRA + MLP', 'LoRA + LA', 'LoRA + LA (sequence sampling)',
         'frozen + LA', 'frozen + MLP']
STYLE = {
    'CNN':                           ('#777777', '-.'),
    'LoRA + MLP':                    ('#d95f02', '-'),
    'LoRA + LA':                     ('#1b9e77', '-'),
    'LoRA + LA (sequence sampling)': ('#1b9e77', ':'),
    'frozen + LA':                   ('#7570b3', '--'),
    'frozen + MLP':                  ('#e7298a', '--'),
}


def main():
    results = json.load(open(os.path.join(OUT, 'results.json')))

    # 18 cm wide, 7 cm high. tight_layout below keeps this size; do not save
    # with bbox_inches='tight', which would crop it.
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(18 / 2.54, 7 / 2.54))
    grid = np.linspace(0.001, 0.999, 400)
    ax.plot(grid, np.sqrt(grid * (1 - grid)), color='black', lw=1.0, ls=(0, (1, 2)),
            label=r'ceiling $\sqrt{p(1-p)}$')

    for label in ORDER:
        rows = results.get(label, {}).get('spread')
        if not rows:
            continue
        p = [r[0] for r in rows]
        sd = [r[1] for r in rows]
        color, ls = STYLE[label]
        ax.plot(p, sd, color=color, ls=ls, lw=1.8, marker='o', ms=3, label=label)

    ax.set_xlabel('ensemble mean prediction')
    ax.set_ylabel('standard deviation across\nthe 50 submodels')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.52)
    # Every curve peaks in the middle and falls to zero at both ends, so the
    # band under them is empty and the legend can sit there without covering
    # anything.
    ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 0.005),
              ncol=3, frameon=False, handlelength=2.0, columnspacing=1.4,
              labelspacing=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        path = os.path.join(OUT, f'uncertainty_vs_prediction.{ext}')
        fig.savefig(path, dpi=300)
        print('wrote', path)

    csv = os.path.join(OUT, 'uncertainty_vs_prediction.csv')
    with open(csv, 'w') as fh:
        fh.write('variant,bin_center,mean_std,n\n')
        for label in ORDER:
            for r in results.get(label, {}).get('spread', []):
                fh.write(f'{label},{r[0]:.4f},{r[1]:.6f},{r[2]}\n')
    print('wrote', csv)


if __name__ == '__main__':
    main()
