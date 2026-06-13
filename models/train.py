import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from argparse import ArgumentParser
import torch
from models.dataset import CrossValidationDataset
from typing import Tuple, List
from bootstrappers.model import build_model_from_configuration
from bootstrappers.optimizer import build_optimizer_from_configuration
from bootstrappers.loss import build_loss_from_configuration
from bootstrappers.results_folder_path import build_results_folder_path_from_configuration
from bootstrappers.dataset import build_dataset_from_configuration
from bootstrappers.training import run_training_function_from_parameters
from bootstrappers.tokenizer import build_tokenizer_from_configuration
from bootstrappers.save_model import save_model_from_parameters
from bootstrappers.infer import infer_from_parameters
from models.dataset import Dataset
from transformers import AutoTokenizer,Trainer,PreTrainedTokenizer
from peft import PeftModel
import sys
from utils import save_as_pickle
import utils
import numpy as np
from models.inference_LM import predict_binary_probs_from_sequences
from models.inference_convolution import predict_binary_probs_from_sequences_convolution

def save_inference_results(inference_results: dict, path: str):
    for set_name, set_dict in inference_results.items():
        set_path = os.path.join(path, set_name)
        os.makedirs(set_path, exist_ok=True)
        predictions = set_dict['predictions']
        labels = set_dict['labels']
        np.save(os.path.join(set_path, 'predictions.npy'), predictions)
        np.save(os.path.join(set_path, 'labels.npy'), labels)
        save_as_pickle(set_dict['sequences'],os.path.join(set_path, 'sequences.pkl'), )


def is_transductive_mode(train_configuration: dict) -> bool:
    data_kwargs = train_configuration.get('data', {}).get('kwargs', {})
    ensemble_dict = data_kwargs.get('ensemble_dict', {})
    return ensemble_dict.get('algorithm_name') == 'transductive_pu_learning'


def predict_sequences_from_training_mode(values_for_training_dict: dict, sequences: List[str]) -> np.ndarray:
    train_name = values_for_training_dict['training_configuration']['train']['name']
    if len(sequences) == 0:
        return np.array([], dtype=np.float32)

    if train_name == 'LM':
        predictions = predict_binary_probs_from_sequences(
            model=values_for_training_dict['model'],
            sequences=sequences,
            tokenizer=values_for_training_dict['tokenizer'],
            device=values_for_training_dict['device'],
        )
    elif train_name == 'convolution':
        predictions = predict_binary_probs_from_sequences_convolution(
            model=values_for_training_dict['model'],
            sequences=sequences,
            device=values_for_training_dict['device'],
        )
    else:
        raise ValueError(f'Unsupported train mode for transductive prediction: {train_name}')

    return np.asarray(predictions).reshape(-1)


def build_transductive_submodel_payload(
    unlabeled_ids: List[str],
    unlabeled_sequences: List[str],
    complement_ids: List[str],
    complement_sequences: List[str],
    predictions: np.ndarray,
    fold_index: int,
) -> dict:
    if predictions.shape[0] != len(unlabeled_ids):
        raise ValueError(
            f'Transductive predictions shape mismatch in fold {fold_index}: '
            f'{predictions.shape[0]} predictions for {len(unlabeled_ids)} unlabeled ids'
        )

    complement_keys = set(zip(complement_ids, complement_sequences))
    w = np.asarray(
        [
            1.0 if (sample_id, sequence) in complement_keys else 0.0
            for sample_id, sequence in zip(unlabeled_ids, unlabeled_sequences)
        ],
        dtype=np.float32,
    )

    masked_predictions = predictions.astype(np.float32) * w

    return {
        'fold_index': fold_index,
        'sample_ids': unlabeled_ids,
        'sample_sequences': unlabeled_sequences,
        'w': w,
        'prediction': masked_predictions,
    }


def save_transductive_payload(path: str, payload: dict) -> None:
    save_as_pickle(payload, path)
    npz_path = path.replace('.pkl', '.npz')
    np.savez(
        npz_path,
        sample_ids=np.array(payload['sample_ids'], dtype=object),
        sample_sequences=np.array(payload['sample_sequences'], dtype=object),
        w=payload['w'],
        prediction=payload['prediction'],
    )


def save_transductive_fold_stats(
    values_for_training_dict: dict,
    fold_path: str,
    fold_index: int,
) -> None:
    fold = values_for_training_dict['dataset'].fold
    if 'transductive_complement_sequences' not in fold:
        return

    unlabeled_sequences = fold['transductive_unlabeled_sequences']
    unlabeled_ids = fold['transductive_unlabeled_ids']
    complement_sequences = fold['transductive_complement_sequences']
    complement_ids = fold['transductive_complement_ids']

    predictions = predict_sequences_from_training_mode(values_for_training_dict, unlabeled_sequences)

    transductive_submodel_payload = build_transductive_submodel_payload(
        unlabeled_ids=unlabeled_ids,
        unlabeled_sequences=unlabeled_sequences,
        complement_ids=complement_ids,
        complement_sequences=complement_sequences,
        predictions=predictions,
        fold_index=fold_index,
    )

    # Keep a single per-fold payload in run results.
    save_transductive_payload(
        os.path.join(fold_path, 'transductive_stats.pkl'),
        transductive_submodel_payload,
    )


def load_configuration() -> dict:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="configuration_path",
                        help="path to configuration file")
    args = parser.parse_args()
    configuration_path = args.configuration_path
    print('experiment configuration file:', configuration_path)
    with open(configuration_path, 'r') as f:
        train_configuration = json.load(f)
    return train_configuration

def bootstrap_train(results_folder_path:str, train_configuration: dict, dataset:Dataset,fold_index:int, device:torch.device) -> Tuple[PeftModel,PreTrainedTokenizer , torch.nn.Module,Trainer]:
    loss = build_loss_from_configuration(
        dataset=dataset,device=device,**train_configuration['compile']['loss'])
    model = build_model_from_configuration(loss_fct=loss,device=device,
                                           fold_index=fold_index,
                                           **train_configuration['model'])
    tokenizer = build_tokenizer_from_configuration(name=train_configuration['train']['name'],
                    model_name=train_configuration['model']['kwargs']['model_name'] if 'model_name' 
                    in train_configuration['model']['kwargs'] else None)
    optimizer = build_optimizer_from_configuration(
        model=model,loss_fct=loss,device=device,**train_configuration['compile']['optimizer']) 
    output_dir = os.path.join(results_folder_path, 'checkpoints')
    logging_dir = os.path.join(results_folder_path, 'logs')
    values_for_training_dict = {
        'output_dir': output_dir,
        'logging_dir': logging_dir,
        'training_configuration':train_configuration,
        'dataset':dataset,
        'optimizer':optimizer,
        'tokenizer':tokenizer,
        'model':model,
        'device':device,
        'loss_fct':loss
    }
    return values_for_training_dict


def run_cross_validation(results_folder_path: str,
                         train_configuration: dict,
                         cross_validation_dataset: CrossValidationDataset,device: torch.device) -> None:    
    
    for i, dataset in enumerate(cross_validation_dataset.fold_datasets):
        fold_path = os.path.join(results_folder_path, f'fold_{i}') 
        os.makedirs(fold_path, exist_ok=True)
        values_for_training_dict = bootstrap_train(results_folder_path=fold_path,
                                                  train_configuration=train_configuration,
                                                  dataset=dataset,
                                                  device=device,
                                                  fold_index=i)
        
        model = run_training_function_from_parameters(name=train_configuration['train']['name'],fold_index=i,kwargs=values_for_training_dict)
        values_for_training_dict['model'] = model
        inference_results = infer_from_parameters(name=train_configuration['train']['name'],values_for_training_dict=values_for_training_dict)
        save_model_from_parameters(name= train_configuration['model']['name'],
                                   model=model,
                                   save_dir = os.path.join(fold_path, 'model'))
        save_inference_results(inference_results, fold_path)
        if is_transductive_mode(train_configuration):
            save_transductive_fold_stats(
                values_for_training_dict=values_for_training_dict,
                fold_path=fold_path,
                fold_index=i,
            )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    train_configuration = load_configuration()
    results_folder_path = build_results_folder_path_from_configuration(
        hypothesis=train_configuration['hypothesis'], experiment=train_configuration['experiment'])
    print('results path:', results_folder_path)

    with open(f'{results_folder_path}/configuration.json', 'w') as f:
        json.dump(train_configuration, f)

    cross_validation_dataset = build_dataset_from_configuration(**train_configuration['data'])

    run_cross_validation(results_folder_path,
                         train_configuration,
                         cross_validation_dataset,device)

