"""Configs for the full Table 2 ablation on the shipped partition.

Every row uses the public repo's partition, the cluster sampler (whole
sequence-identity clusters drawn from unlabeled sequences only, as Methods 4.3
describes) and LoRA r=10, alpha=8 throughout, so the table matches the method
the paper describes and lines up with the Section 4.5 sampling-fraction sweep.

Four ensembles, 50 submodels each, plus one single model per architecture:

  frozen_LA          ESM2 frozen + LightAttention          cluster groups
  lora_MLP           ESM2 + LoRA + MLP head                cluster groups
  frozen_MLP         ESM2 frozen + MLP head                cluster groups
  lora_LA_sequence   ESM2 + LoRA + LightAttention          SEQUENCE groups

The cluster-sampled LoRA+LA ensemble is not built here: it is the fraction_0.3
arm of `ensemble_inductive_pu_learning_negative_fractions`. The
sequence-sampled arm above is its counterpart, and the pair tests whether
cluster sampling performs better than sequence sampling.

The CNN baseline is not here: the public repo has no convolution training
function (ignite and libauc were removed), so it is run from `cpp` instead.
"""

import copy
import json
import os

HYPOTHESIS = 'ensemble_ablation_v2_partition'
ITERATION = 50
SUBMODELS = 50
NEGATIVE_SIZE = 0.3

CLUSTER_GROUPS = (f'datasets/data_for_training/ensembles/inductive_pu_learning/'
                  f'iteration_{ITERATION}_negative_size_{NEGATIVE_SIZE}')
SEQUENCE_GROUPS = (f'datasets/data_for_training/ensembles/inductive_pu_learning_sequence_sampling/'
                   f'iteration_{ITERATION}_negative_size_{NEGATIVE_SIZE}')

LORA_KWARGS = {
    'r': 10,
    'lora_alpha': 8,
    'lora_dropout': 0.3,
    'modules_to_save': ['classifier', 'light_attention'],
    'target_modules': [
        r'(esm|model)\.encoder\.layer\.\d+\.attention\.self\.query',
        r'(esm|model)\.encoder\.layer\.\d+\.attention\.self\.value',
    ],
}
ESM2 = 'facebook/esm2_t6_8M_UR50D'

VARIANTS = {
    'frozen_LA': {
        'groups': CLUSTER_GROUPS,
        'model': {'name': 'esm2_LA_no_lora',
                  'kwargs': {'model_name': ESM2, 'num_labels': 1, 'dout': 128,
                             'kernel_size': 7, 'use_max': True}},
    },
    'lora_MLP': {
        'groups': CLUSTER_GROUPS,
        'model': {'name': 'esm2_with_classification_head',
                  'kwargs': {'model_name': ESM2, 'num_labels': 1,
                             'lora_kwargs': copy.deepcopy(LORA_KWARGS)}},
    },
    'frozen_MLP': {
        'groups': CLUSTER_GROUPS,
        'model': {'name': 'esm2_classification_head_no_lora',
                  'kwargs': {'model_name': ESM2, 'num_labels': 1}},
    },
    'lora_LA_sequence': {
        'groups': SEQUENCE_GROUPS,
        'model': {'name': 'esm2_with_LA_head',
                  'kwargs': {'model_name': ESM2, 'num_labels': 1, 'dout': 128,
                             'kernel_size': 7, 'use_max': True,
                             'lora_kwargs': copy.deepcopy(LORA_KWARGS)}},
    },
}


def base_configuration():
    return {
        'hypothesis': HYPOTHESIS,
        'experiment': {'experiment_type': 'ensemble_submodel', 'name': None, 'kwargs': {}},
        'data': {
            'name': 'ensemble_submodel',
            'path': 'datasets/data_for_training/folds_training_dicts.pkl',
            'kwargs': {'ensemble_dict': {
                'algorithm_name': 'inductive_pu_learning',
                'iteration': ITERATION,
                'negative_size': NEGATIVE_SIZE,
                'groups_path': None,
                'submodel_index': None,
            }},
        },
        'model': None,
        'compile': {
            'optimizer': {'name': 'adamw',
                          'kwargs': {'learning_rate': 0.001, 'weight_decay': 0.01}},
            'loss': {'name': 'binary_cross_entropy', 'kwargs': {}},
            'trainer': {'name': 'regular', 'kwargs': {}},
            'training_arguments': {'name': 'regular',
                                   'kwargs': {'epochs': 200, 'batch_size': 256,
                                              'metric': 'roc_auc'}},
        },
        'train': {'name': 'LM'},
    }


def single_configuration():
    """A single model: no bagging, so no groups. It trains on the whole training
    split with every unlabeled sequence as a negative, which is what the Single
    column of Table 2 reports."""
    cfg = base_configuration()
    cfg['experiment']['experiment_type'] = 'single_model'
    cfg['data'] = {
        'name': 'single_model',
        'path': 'datasets/data_for_training/folds_training_dicts.pkl',
        'kwargs': {},
    }
    return cfg


# The Single column needs one run per architecture. LoRA+LA is included here
# because the existing fraction_0.3 run only covers the ensemble.
SINGLE_VARIANTS = {
    'single_frozen_LA': VARIANTS['frozen_LA']['model'],
    'single_lora_MLP': VARIANTS['lora_MLP']['model'],
    'single_frozen_MLP': VARIANTS['frozen_MLP']['model'],
    'single_lora_LA': VARIANTS['lora_LA_sequence']['model'],
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, '..', 'data', HYPOTHESIS)
    os.makedirs(out, exist_ok=True)

    written = 0
    for variant, spec in VARIANTS.items():
        for index in range(SUBMODELS):
            name = f'{variant}_index_{index}'
            cfg = base_configuration()
            cfg['experiment']['name'] = name
            cfg['model'] = copy.deepcopy(spec['model'])
            ed = cfg['data']['kwargs']['ensemble_dict']
            ed['groups_path'] = spec['groups']
            ed['submodel_index'] = index
            with open(os.path.join(out, f'{name}.json'), 'w') as fh:
                json.dump(cfg, fh)
            written += 1

    singles = 0
    for name, model in SINGLE_VARIANTS.items():
        cfg = single_configuration()
        cfg['experiment']['name'] = name
        cfg['model'] = copy.deepcopy(model)
        with open(os.path.join(out, f'{name}.json'), 'w') as fh:
            json.dump(cfg, fh)
        singles += 1

    print(f'wrote {written} ensemble + {singles} single configs to {os.path.normpath(out)}')
    for variant, spec in VARIANTS.items():
        print(f"  {variant:18s} model={spec['model']['name']:34s} groups={os.path.basename(os.path.dirname(spec['groups']))}")
    for name in SINGLE_VARIANTS:
        print(f"  {name:18s} single model, no groups")


if __name__ == '__main__':
    main()
