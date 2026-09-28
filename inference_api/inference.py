import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch

from inference_api.ensemble import DEFAULT_WORK_DICT, LoRAEnsemble, get_device


def read_fasta(fasta_path:str)->Tuple[List[str],List[str]]:
    """
    Reads a FASTA file and returns a list of sequences and a list of their corresponding labels.
    Assumes that the label is encoded in the header line of the FASTA file, separated by a space.
    For example, a header line might look like: >sequence_id label
    """
    sequences = []
    keys = []
    with open(fasta_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # This is a header line, extract the label
                label = line[1:].strip()
                keys.append(label)
            else:
                # This is a sequence line
                sequences.append(line)
    return sequences, keys


def get_indexes_dict(bagging_cpp_dataset_path: str, sequences: list, no_cross_predictions: bool = False) -> dict:
    """
    Reads the bagging_cpp_dataset.csv file and creates a dictionary mapping fold indices to lists of sequence indices.
    """
    # Initialized with integer keys to prevent KeyError later in the script
    fold_to_indices = {0: [], 1: [], 2: [], 3: [], 4: [], -1: []}
    if no_cross_predictions:
        # If no_cross_predictions is True, we will only use the -1 fold for all sequences
        fold_to_indices[-1] = list(range(len(sequences)))
        return fold_to_indices

    # Only these two columns are used. Reading just them avoids the DtypeWarning raised by the
    # free-text 'description' column and keeps the 450k-row load cheap.
    try:
        df = pd.read_csv(bagging_cpp_dataset_path, usecols=['sequence', 'test_fold_index'])
    except ValueError as e:
        raise ValueError("The input CSV must contain 'sequence' and 'test_fold_index' columns.") from e

    sequences_to_test_folds = dict( zip( df['sequence'], df['test_fold_index']  ) )

    # Using enumerate avoids the O(N^2) complexity of sequences.index(sequence)
    for idx, sequence in enumerate(sequences):
        fold_index = sequences_to_test_folds.get(sequence, -1)
        if fold_index in fold_to_indices:
            fold_to_indices[fold_index].append(idx)
        else:
            fold_to_indices[-1].append(idx)

    return fold_to_indices


def load_ensemble(use_custom_model: bool = False,
                  num_submodels_max: int = 50,
                  device: torch.device = None,
                  work_dict: Optional[dict] = None,
                  verbose: bool = True) -> LoRAEnsemble:
    """
    Build the ensemble once so it can be reused across many prediction calls
    (a design loop should hold on to this object instead of calling `predict`).
    """
    return LoRAEnsemble(work_dict=work_dict,
                        device=device,
                        num_submodels_max=num_submodels_max,
                        use_custom_model=use_custom_model,
                        verbose=verbose)


def predict_helper(sequences_fasta: str,
                   use_custom_model: bool = False,
                   no_cross_predictions: bool = False,
                   batch_size: int = 64,
                   num_submodels_max: int = 50,
                   device: torch.device = None,
                   ensemble: Optional[LoRAEnsemble] = None) -> pd.DataFrame:
    """
    Run the LoRA LA ensemble prediction on the sequences in `sequences_fasta`
    and return the predictions DataFrame.

    Pass an already built `ensemble` to skip loading the adapters from disk.
    """
    sequences, keys = read_fasta(sequences_fasta)
    if ensemble is None:
        ensemble = load_ensemble(use_custom_model=use_custom_model,
                                 num_submodels_max=num_submodels_max,
                                 device=device)
    return ensemble.predict(sequences=sequences,
                            keys=keys,
                            no_cross_predictions=no_cross_predictions,
                            batch_size=batch_size)


def predict(sequences_fasta: str,
            output_csv: str,
            use_custom_model: bool = False,
            no_cross_predictions: bool = False,
            batch_size: int = 64,
            num_submodels_max: int = 50,
            device: torch.device = None,
            ensemble: Optional[LoRAEnsemble] = None) -> pd.DataFrame:
    """
    Run the LoRA LA ensemble prediction on the sequences in `sequences_fasta`,
    save the results to `output_csv` and return the predictions DataFrame.
    """
    predictions_df = predict_helper(sequences_fasta=sequences_fasta,
                                    use_custom_model=use_custom_model,
                                    no_cross_predictions=no_cross_predictions,
                                    batch_size=batch_size,
                                    num_submodels_max=num_submodels_max,
                                    device=device,
                                    ensemble=ensemble)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # i want 4 digits after the decimal point in the predictions and std
    predictions_df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f'Predictions saved to {output_csv}')
    return predictions_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using LoRA LA ensemble model.')
    parser.add_argument('--sequences_fasta', required=True, help='Path to the input sequences in FASTA format.')
    parser.add_argument('--output_csv', required=True, help='Path to save the output predictions CSV file.')
    parser.add_argument('--use_custom_model', action='store_true', help='Flag to indicate if a custom model should be used.')
    parser.add_argument('--no_cross_predictions', action='store_true', help='Flag to indicate if cross-predictions should be disabled.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction.')
    parser.add_argument('--num_submodels_max', type=int, default=50, help='Maximum number of submodels to use (set e.g. to 10 to speed-up inference)')
    args = parser.parse_args()

    predict(sequences_fasta=args.sequences_fasta,
            output_csv=args.output_csv,
            use_custom_model=args.use_custom_model,
            no_cross_predictions=args.no_cross_predictions,
            batch_size=args.batch_size,
            num_submodels_max=args.num_submodels_max)
