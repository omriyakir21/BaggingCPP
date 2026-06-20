import os
import torch
from models.inference_LM import load_esm2_with_LA_lora_model,load_tokenizer,predict_binary_probs_from_sequences
from evaluation.utils_evaluation import get_experiment_base_paths_for_ensemble
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        
    df = pd.read_csv(bagging_cpp_dataset_path)
    if 'test_fold_index' not in df.columns:
        raise ValueError("The input CSV must contain a 'test_fold_index' column.")
    
    sequences_to_test_folds = dict( zip( df['sequence'], df['test_fold_index']  ) )

    # Using enumerate avoids the O(N^2) complexity of sequences.index(sequence)
    for idx, sequence in enumerate(sequences):
        fold_index = sequences_to_test_folds.get(sequence, -1)
        if fold_index in fold_to_indices:
            fold_to_indices[fold_index].append(idx)
        else:
            fold_to_indices[-1].append(idx)
            
    return fold_to_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using LoRA LA ensemble model.')
    parser.add_argument('--sequences_fasta', required=True, help='Path to the input sequences in FASTA format.')
    parser.add_argument('--output_csv', required=True, help='Path to save the output predictions CSV file.')
    parser.add_argument('--use_custom_model', action='store_true', help='Flag to indicate if a custom model should be used.')
    parser.add_argument('--no_cross_predictions', action='store_true', help='Flag to indicate if cross-predictions should be disabled.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction.')
    parser.add_argument('--num_submodels_max', type=int, default=50, help='Maximum number of submodels to use (set e.g. to 10 to speed-up inference)')
    args = parser.parse_args()

    work_dict = {'hypothesis':'ensemble_inductive_pu_learning', 
                         'experiment': 'groups_inductive',
                         'model_name': 'facebook/esm2_t6_8M_UR50D', 
                         'num_submodels': min(50 , args.num_submodels_max), 
                         'num_folds': 5,
                         'huggingface_model_folder_path': 'huggingface_repo/ensemble', 
                         'bagging_cpp_dataset_path': 'datasets/full_datasets/bagging_cpp_dataset.csv',
                         'num_labels': 1, 
                         'dout': 128, 
                         'kernel_size': 7, 
                         'use_max': True
}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    sequences, keys = read_fasta(args.sequences_fasta)

    if args.use_custom_model:
        base_paths, submodels_to_run = get_experiment_base_paths_for_ensemble(experiment=work_dict['experiment'],
                                                        base_paths={},
                                                        hypothesis_path=os.path.join('results', 'hypothesis', work_dict['hypothesis']),
                                                        num_submodels=work_dict['num_submodels'])
        base_paths_list = list(base_paths[work_dict['experiment']])

   
    indexes_dict = get_indexes_dict(bagging_cpp_dataset_path=work_dict['bagging_cpp_dataset_path'],\
                                    sequences=sequences, no_cross_predictions=args.no_cross_predictions)
    for i in range(-1, work_dict['num_folds']):
        print(f'Fold {i} has {len(indexes_dict[i])} sequences to predict on.')
    tokenizer = load_tokenizer(model_name=work_dict['model_name'])
    ordered_predictions = np.empty(len(sequences), dtype=float)
    ordered_std = np.empty(len(sequences), dtype=float)
    non_specific_predictions = []
    non_specific_sequences = [sequences[i] for i in indexes_dict[-1]]
    model = None
    for fold in tqdm(range(work_dict['num_folds']), desc=f"Predicting using ensemble"):
        fold_indices = indexes_dict[fold]
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        fold_specific_predictions = []
        fold_non_specific_predictions = []
        for index in range(work_dict['num_submodels']):
            try:
                if args.use_custom_model:
                    model_path = os.path.join(base_paths_list[index], f'fold_{fold}', 'model')     
                else:
                    model_path = os.path.join(work_dict['huggingface_model_folder_path'] ,f'submodel_{index}',f'fold_{fold}','model')

                model = load_esm2_with_LA_lora_model(model_path=model_path,device=device,model_name=work_dict['model_name'],
                                                    num_labels=work_dict['num_labels'],dout=work_dict['dout'],kernel_size=work_dict['kernel_size'],
                                                    use_max=work_dict['use_max'],model=model)
                if len(specific_fold_sequences) > 0:
                    fold_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                            sequences=specific_fold_sequences,
                                                                            tokenizer=tokenizer,batch_size=args.batch_size,
                                                                            use_tqdm=False))
                else: 
                    fold_specific_predictions.append(np.zeros((len(specific_fold_sequences), 1)))
                
                if len(non_specific_sequences) > 0:
                    fold_non_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                                sequences=non_specific_sequences,
                                                                                tokenizer=tokenizer,batch_size=args.batch_size,
                                                                                use_tqdm=False))
                else:
                    fold_non_specific_predictions.append(np.zeros((len(non_specific_sequences), 1)))

            except Exception as e:
                print(e)
                print(f'Error loading fold {fold} for submodel {index}')
                continue
        fold_specific_std = np.std(fold_specific_predictions, axis=0).reshape(-1)
        ordered_std[fold_indices] = fold_specific_std
        fold_specific_predictions = np.mean(fold_specific_predictions, axis=0).reshape(-1)
        ordered_predictions[fold_indices] = fold_specific_predictions
        fold_non_specific_predictions = np.mean(fold_non_specific_predictions, axis=0)
        non_specific_predictions.append(fold_non_specific_predictions)
    non_specific_predictions = np.mean(non_specific_predictions, axis=0)
    ordered_predictions[indexes_dict[-1]] = non_specific_predictions.reshape(-1)
    ordered_std[indexes_dict[-1]] = np.std(non_specific_predictions, axis=0).reshape(-1)
    print('Finished predictions, saving to CSV...')
    predictions_df = pd.DataFrame({
        'sequence': sequences,
        'label': keys,
        'prediction': ordered_predictions,
        'model_uncertainty': ordered_std
        
    })
    output_dir = os.path.dirname(args.output_csv)
    os.makedirs(output_dir, exist_ok=True)
    # i want 4 digits after the decimal point in the predictions and std
    predictions_df.to_csv(args.output_csv, index=False, float_format='%.4f')
    print(f'Predictions saved to {args.output_csv}')

