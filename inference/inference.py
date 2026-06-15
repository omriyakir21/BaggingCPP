import os

import torch
from models.inference_LM import load_esm2_with_LA_lora_model,load_tokenizer,predict_binary_probs_from_sequences
from evaluation.utils_evaluation import get_fold_indexes,get_experiment_base_paths_for_ensemble
import argparse
from utils import load_as_pickle
from typing import List, Tuple
import numpy as np
import pandas as pd

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using LoRA LA ensemble model.')
    parser.add_argument('--sequences_fasta', required=True, help='Path to the input sequences in FASTA format.')
    parser.add_argument('--output_csv', required=True, help='Path to save the output predictions CSV file.')
    parser.add_argument('--use_huggingface_repo', action='store_true', help='Whether to load the model from the Hugging Face repository.')
    parser.add_argument('--hypothesis', default="ensemble_inductive_pu_learning", help='Hypothesis name under configurations/data and results/hypothesis.')
    parser.add_argument('--experiment', default="groups_inductive", help='Experiment name under results/hypothesis/experiment.')
    parser.add_argument('--model_name', default="facebook/esm2_t6_8M_UR50D", help='Name of the ESM2 model to use.')
    parser.add_argument('--num_submodels', type=int, default=50, help='Number of submodels in the ensemble.')
    parser.add_argument('--huggingface_model_folder_path', default="huggingface_repo/ensemble", help='Path to the Hugging Face model folder containing the ensemble submodel folders.')
    parser.add_argument('--folds_training_dicts_path', help='Path to the folds training dictionaries pickle file, required if --use_huggingface_repo is not set.')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for classification.')
    parser.add_argument('--dout', type=int, default=128, help='Output dimension for the light attention head.')
    parser.add_argument('--kernel_size', type=int, default=7, help='Kernel size for the light attention head.')
    args = parser.parse_args()

    use_max = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    sequences, keys = read_fasta(args.sequences_fasta)
    # folds_training_dicts = load_as_pickle(args.folds_training_dicts_path)
    if not args.use_huggingface_repo:
        base_paths, submodels_to_run = get_experiment_base_paths_for_ensemble(experiment=args.experiment,
                                                        base_paths={},
                                                        hypothesis_path=os.path.join('results', 'hypothesis', args.hypothesis),
                                                        num_submodels=args.num_submodels)
        base_paths_list = list(base_paths[args.experiment])

    # indexes_dict = get_fold_indexes(fold_training_dicts=folds_training_dicts,sequences=sequences)
    indexes_dict = {0:[],1:[],2:[],3:[],4:[],-1: list(range(len(sequences))) }
    tokenizer = load_tokenizer(model_name=args.model_name)
    ordered_predictions = np.empty(len(sequences), dtype=float)
    non_specific_predictions = []
    non_specific_sequences = [sequences[i] for i in indexes_dict[-1]]
    for fold in range(5):
        fold_indices = indexes_dict[fold]
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        fold_specific_predictions = []
        fold_non_specific_predictions = []
        for index in range(args.num_submodels):
            try:
                if args.use_huggingface_repo:
                    model_path = os.path.join(args.huggingface_model_folder_path ,f'submodel_{index}',f'fold_{fold}','model')
                else:
                    model_path = os.path.join(base_paths_list[index], f'fold_{fold}', 'model')

                model = load_esm2_with_LA_lora_model(model_path=model_path,device=device,model_name=args.model_name,
                                                    num_labels=args.num_labels,dout=args.dout,kernel_size=args.kernel_size,use_max=use_max)
                if len(specific_fold_sequences) > 0:
                    fold_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                            sequences=specific_fold_sequences,
                                                                            tokenizer=tokenizer,batch_size=512))
                else: 
                    fold_specific_predictions.append(np.zeros((len(specific_fold_sequences), 1)))
                
                if len(non_specific_sequences) > 0:
                    fold_non_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                                sequences=non_specific_sequences,
                                                                                tokenizer=tokenizer,batch_size=512))
                else:
                    fold_non_specific_predictions.append(np.zeros((len(non_specific_sequences), 1)))

            except Exception as e:
                print(e)
                print(f'Error loading fold {fold} for submodel {index}')
                continue
        fold_specific_predictions = np.mean(fold_specific_predictions, axis=0).reshape(-1)
        ordered_predictions[fold_indices] = fold_specific_predictions
        fold_non_specific_predictions = np.mean(fold_non_specific_predictions, axis=0)
        non_specific_predictions.append(fold_non_specific_predictions)
    non_specific_predictions = np.mean(non_specific_predictions, axis=0)
    ordered_predictions[indexes_dict[-1]] = non_specific_predictions.reshape(-1)
    predictions_df = pd.DataFrame({
        'sequences': sequences,
        'prediction': ordered_predictions,
    })

    predictions_df.to_csv(args.output_csv, index=False)

