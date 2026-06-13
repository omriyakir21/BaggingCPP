import argparse
import os
from typing import Dict, List, Tuple
import numpy as np
import utils
from evaluation.utils_evaluation import gather_ensemble_experiments_base_paths


def create_experiment_commands_file(
    hypothesis: str,
    submodels_to_run: List[str],
    output_file: str,
) -> None:
    with open(output_file, 'w') as f:
        f.write('#!/bin/bash\n\n')
        for submodel in submodels_to_run:
            command = (
                f'bash scripts/train/submit_job_experiment.sh '
                f'configurations/data/{hypothesis}/{submodel}.json'
            )
            f.write(command + '\n')
            print(f'Added command for missing submodel experiment: {submodel}')
    print(f'Commands written to {output_file}')


def get_experiments_from_configurations(hypothesis: str, configurations_root: str) -> List[str]:
    hypothesis_config_dir = os.path.join(configurations_root, hypothesis)
    if not os.path.isdir(hypothesis_config_dir):
        raise FileNotFoundError(
            f"Configuration directory not found for hypothesis '{hypothesis}': {hypothesis_config_dir}"
        )

    config_files = [
        name for name in os.listdir(hypothesis_config_dir) if name.endswith('.json')
    ]
    experiments = sorted({name.split('_index_')[0] for name in config_files})
    return experiments


def get_num_submodels(experiment: str, default_it: int) -> int:
    if 'it_' not in experiment:
        return default_it
    return int(experiment.split('it_')[1].split('_')[0])


def validate_submodel_payload(payload: dict, payload_path: str) -> None:
    required_keys = ['fold_index', 'sample_ids', 'sample_sequences', 'w', 'prediction']
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Missing key '{key}' in payload: {payload_path}")

    sample_ids = payload['sample_ids']
    sample_sequences = payload['sample_sequences']
    w = np.asarray(payload['w'], dtype=np.float32)
    prediction = np.asarray(payload['prediction'], dtype=np.float32)

    if not (len(sample_ids) == len(sample_sequences) == w.shape[0] == prediction.shape[0]):
        raise ValueError(
            f"Inconsistent payload lengths in {payload_path}: "
            f"ids={len(sample_ids)}, sequences={len(sample_sequences)}, w={w.shape[0]}, "
            f"prediction={prediction.shape[0]}"
        )


def aggregate_payloads_across_folds(payloads: List[dict]) -> dict:
    merged: Dict[Tuple[str, str], Dict[str, float]] = {}
    fold_indices = set()

    for payload in payloads:
        fold_indices.add(payload.get('fold_index'))
        sample_ids = payload['sample_ids']
        sample_sequences = payload['sample_sequences']
        w = np.asarray(payload['w'], dtype=np.float32)
        prediction = np.asarray(payload['prediction'], dtype=np.float32)

        for sample_id, sample_sequence, w_i, pred_i in zip(sample_ids, sample_sequences, w, prediction):
            key = (sample_id, sample_sequence)
            if key not in merged:
                merged[key] = {
                    'sample_id': sample_id,
                    'sample_sequence': sample_sequence,
                    'w': 0.0,
                    'av_f': 0.0,
                    'av_f2': 0.0,
                }

            w_i = float(w_i)
            pred_i = float(pred_i) if w_i > 0 else 0.0

            merged[key]['w'] += w_i
            merged[key]['av_f'] += pred_i
            merged[key]['av_f2'] += pred_i * pred_i

    merged_items = list(merged.values())
    sample_ids = [item['sample_id'] for item in merged_items]
    sample_sequences = [item['sample_sequence'] for item in merged_items]
    w = np.asarray([item['w'] for item in merged_items], dtype=np.float32)
    av_f = np.asarray([item['av_f'] for item in merged_items], dtype=np.float32)
    av_f2 = np.asarray([item['av_f2'] for item in merged_items], dtype=np.float32)

    mean_pred = np.zeros_like(av_f)
    non_zero_mask = w > 0
    mean_pred[non_zero_mask] = av_f[non_zero_mask] / w[non_zero_mask]

    var_pred = np.zeros_like(av_f2)
    var_pred[non_zero_mask] = (
        av_f2[non_zero_mask] / w[non_zero_mask] - np.square(mean_pred[non_zero_mask])
    )
    var_pred = np.maximum(var_pred, 0.0)

    return {
        'fold_indices': sorted(idx for idx in fold_indices if idx is not None),
        'sample_ids': sample_ids,
        'sample_sequences': sample_sequences,
        'w': w,
        'av_f': av_f,
        'av_f2': av_f2,
        'mean_pred': mean_pred,
        'var_pred': var_pred,
    }


def aggregate_experiment(
    hypothesis: str,
    experiment: str,
    submodel_base_paths: List[str],
    num_submodels: int,
    folds: int,
    stats_filename: str,
    results_root: str,
) -> bool:
    if len(submodel_base_paths) < num_submodels:
        print(
            f"Skipping {experiment}: found {len(submodel_base_paths)} completed submodel runs, "
            f"expected {num_submodels}."
        )
        return False

    selected_base_paths = submodel_base_paths[:num_submodels]

    missing_payloads = []
    for submodel_index, submodel_base_path in enumerate(selected_base_paths):
        for fold_index in range(folds):
            payload_path = os.path.join(submodel_base_path, f'fold_{fold_index}', stats_filename)
            if not os.path.exists(payload_path):
                missing_payloads.append((submodel_index, fold_index, payload_path))

    if missing_payloads:
        print(f"Skipping {experiment}: missing {len(missing_payloads)} payload files.")
        for submodel_index, fold_index, payload_path in missing_payloads[:10]:
            print(
                f"  missing submodel={submodel_index}, fold={fold_index}, path={payload_path}"
            )
        if len(missing_payloads) > 10:
            print(f"  ... and {len(missing_payloads) - 10} more")
        return False

    aggregated_payload_dir = os.path.join(
        results_root,
        hypothesis,
        experiment,
        'aggregated_payload',
    )
    os.makedirs(aggregated_payload_dir, exist_ok=True)

    all_payloads = []
    for fold_index in range(folds):
        for submodel_base_path in selected_base_paths:
            payload_path = os.path.join(submodel_base_path, f'fold_{fold_index}', stats_filename)
            payload = utils.load_as_pickle(payload_path)
            validate_submodel_payload(payload, payload_path)
            all_payloads.append(payload)

    aggregated_payload = aggregate_payloads_across_folds(all_payloads)
    output_path = os.path.join(aggregated_payload_dir, 'all_folds_aggregated.pkl')
    utils.save_as_pickle(aggregated_payload, output_path)
    print(f"Saved aggregated payload across folds: {output_path}")

    return True

def create_predictions_csv_from_aggregated_payload(
    aggregated_payload_path: str,
    output_csv_path: str,
) -> None:
    aggregated_payload = utils.load_as_pickle(aggregated_payload_path)
    sample_ids = aggregated_payload['sample_ids']
    sample_sequences = aggregated_payload['sample_sequences']
    mean_pred = aggregated_payload['mean_pred']
    var_pred = aggregated_payload['var_pred']

    with open(output_csv_path, 'w') as f:
        f.write('sample_id,sample_sequence,mean_prediction,var_prediction\n')
        for sample_id, sample_sequence, mean_p, var_p in zip(sample_ids, sample_sequences, mean_pred, var_pred):
            f.write(f'{sample_id},{sample_sequence},{mean_p:.6f},{var_p:.6f}\n')
    print(f"Saved predictions CSV: {output_csv_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate transductive per-submodel payloads after ensemble training.'
    )
    parser.add_argument('--hypothesis', required=True, default="ensemble_transductive_pu_learning",help='Hypothesis name under configurations/data and results/hypothesis.')
    parser.add_argument('--it', type=int, default=50, help='Default number of submodels for experiments without it_ in the name.')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds to aggregate.')
    parser.add_argument('--stats-filename', default='transductive_stats.pkl', help='Per-fold payload filename produced by training.')
    parser.add_argument(
        '--configurations-root',
        default=os.path.join('configurations', 'data'),
        help='Root directory containing per-hypothesis configuration folders.',
    )
    parser.add_argument(
        '--results-root',
        default=os.path.join('results', 'hypothesis'),
        help='Root directory containing per-hypothesis result folders.',
    )
    parser.add_argument(
        '--commands-output',
        default=os.path.join('submit_ensemble_experiments.sh'),
        help='Output path for the generated missing-experiments shell commands file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiments = get_experiments_from_configurations(
        hypothesis=args.hypothesis,
        configurations_root=args.configurations_root,
    )
    if not experiments:
        print(f"No experiments found for hypothesis: {args.hypothesis}")
        return

    print(f"Found experiments for hypothesis '{args.hypothesis}': {experiments}")

    base_paths, submodels_to_run = gather_ensemble_experiments_base_paths(
        hypothesis=args.hypothesis,
        experiments=experiments,
        it=args.it,
        results_root=args.results_root,
        required_fold=f'fold_{args.folds - 1}',
    )

    if submodels_to_run:
        print(f"Detected incomplete submodels from run scan ({len(submodels_to_run)}):")
        print(submodels_to_run)

    create_experiment_commands_file(
        hypothesis=args.hypothesis,
        submodels_to_run=submodels_to_run,
        output_file=args.commands_output,
    )

    aggregated_experiments = []
    skipped_experiments = []

    for experiment in experiments:
        num_submodels = get_num_submodels(experiment, default_it=args.it)
        did_aggregate = aggregate_experiment(
            hypothesis=args.hypothesis,
            experiment=experiment,
            submodel_base_paths=base_paths.get(experiment, []),
            num_submodels=num_submodels,
            folds=args.folds,
            stats_filename=args.stats_filename,
            results_root=args.results_root,
        )
        if did_aggregate:
            aggregated_experiments.append(experiment)
        else:
            skipped_experiments.append(experiment)



    print('\nAggregation summary')
    print(f'Aggregated experiments ({len(aggregated_experiments)}): {aggregated_experiments}')
    print(f'Skipped experiments ({len(skipped_experiments)}): {skipped_experiments}')

    create_predictions_csv_from_aggregated_payload(
        aggregated_payload_path=os.path.join(
            args.results_root,
            args.hypothesis,
            aggregated_experiments[0],
            'aggregated_payload',
            'all_folds_aggregated.pkl',
        ),
        output_csv_path=os.path.join(
            args.results_root,
            args.hypothesis,
            aggregated_experiments[0],
            'aggregated_payload',
            'predictions.csv',
        ),
    )
