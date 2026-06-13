import os
from typing import Union
import torch
from libauc.losses import AUCMLoss
from models.dataset import CrossValidationDataset
import utils
import numpy as np
import copy


def resolve_groups_root(ensemble_dict: dict) -> str:
    groups_root = ensemble_dict['groups_path']
    if ensemble_dict.get('algorithm_name') != 'transductive_pu_learning':
        return groups_root

    group_creation_algorithm = ensemble_dict.get('transductive_pu_group_creation_algorithm')
    if group_creation_algorithm is None:
        return groups_root

    candidate_root = os.path.join(groups_root, group_creation_algorithm)
    if os.path.isdir(candidate_root):
        return candidate_root
    return groups_root


def create_cross_validation_dataset_for_folds(fold_training_dict: list,submodel_index:int, groups_path:str,fold_index:int) -> CrossValidationDataset:
    labels_fold = fold_training_dict['labels_train']
    positive_indexes_fold = np.where(labels_fold == np.int64(1))[0]
    group_for_train_all_submodels = utils.load_as_pickle(os.path.join(groups_path,f'fold_{fold_index}','groups_for_ensemble.pkl'))
    group_for_train = group_for_train_all_submodels[submodel_index]
    group_indices_fold = np.concatenate([positive_indexes_fold, group_for_train])
    sequences_fold = np.array(fold_training_dict['sequences_train'])[group_indices_fold].tolist()
    train_ids_fold = np.array(fold_training_dict['ids_train'])[group_indices_fold].tolist()
    train_descriptions_fold = np.array(fold_training_dict['descriptions_train'])[group_indices_fold].tolist()
    labels_fold = np.concatenate([np.ones(positive_indexes_fold.shape[0]), np.zeros(len(group_for_train))])
    submodel_fold_training_dict = copy.deepcopy(fold_training_dict)
    submodel_fold_training_dict['sequences_train'] = sequences_fold
    submodel_fold_training_dict['ids_train'] = train_ids_fold
    submodel_fold_training_dict['descriptions_train'] = train_descriptions_fold
    submodel_fold_training_dict['labels_train'] = labels_fold
    return submodel_fold_training_dict


def create_transductive_cross_validation_dataset_for_folds(
    fold_training_dict: dict,
    submodel_index: int,
    groups_path: str,
    fold_index: int,
) -> dict:
    # Transductive mode uses 4 folds for training (train + test from the original split)
    # and keeps the original validation fold for early stopping.
    merged_train_sequences = np.array(
        fold_training_dict['sequences_train'] + fold_training_dict['sequences_test']
    )
    merged_train_ids = np.array(fold_training_dict['ids_train'] + fold_training_dict['ids_test'])
    merged_train_descriptions = np.array(
        fold_training_dict['descriptions_train'] + fold_training_dict['descriptions_test']
    )
    merged_train_labels = np.array(
        fold_training_dict['labels_train'] + fold_training_dict['labels_test']
    )

    positive_indices = np.where(merged_train_labels == np.int64(1))[0]
    unlabeled_indices = np.where(merged_train_labels == np.int64(0))[0]

    group_for_train_all_submodels = utils.load_as_pickle(
        os.path.join(groups_path, f'fold_{fold_index}', 'groups_for_ensemble.pkl')
    )
    sampled_negative_indices = np.array(
        group_for_train_all_submodels[submodel_index],
        dtype=np.int64,
    )

    if sampled_negative_indices.size == 0:
        complement_unlabeled_indices = unlabeled_indices.copy()
    else:
        if not np.all(np.isin(sampled_negative_indices, unlabeled_indices)):
            raise ValueError(
                f'Found sampled indices outside merged unlabeled pool in fold {fold_index}. '
                f'Check groups_path={groups_path} for transductive preprocessing compatibility.'
            )
        complement_unlabeled_indices = np.setdiff1d(
            unlabeled_indices,
            np.unique(sampled_negative_indices),
            assume_unique=False,
        )

    train_indices = np.concatenate([positive_indices, sampled_negative_indices])
    train_labels = np.concatenate(
        [
            np.ones(positive_indices.shape[0], dtype=np.int64),
            np.zeros(sampled_negative_indices.shape[0], dtype=np.int64),
        ]
    )

    submodel_fold_training_dict = copy.deepcopy(fold_training_dict)
    submodel_fold_training_dict['sequences_train'] = merged_train_sequences[train_indices].tolist()
    submodel_fold_training_dict['ids_train'] = merged_train_ids[train_indices].tolist()
    submodel_fold_training_dict['descriptions_train'] = merged_train_descriptions[train_indices].tolist()
    submodel_fold_training_dict['labels_train'] = train_labels

    # In transductive PU mode, original test fold is absorbed into training.
    # Keep validation as-is and clear test split to avoid reporting test metrics.
    submodel_fold_training_dict['sequences_test'] = []
    submodel_fold_training_dict['ids_test'] = []
    submodel_fold_training_dict['descriptions_test'] = []
    submodel_fold_training_dict['labels_test'] = np.array([], dtype=np.int64)

    # Keep full unlabeled pool and complement metadata for Algorithm-2 aggregation.
    submodel_fold_training_dict['transductive_unlabeled_ids'] = merged_train_ids[unlabeled_indices].tolist()
    submodel_fold_training_dict['transductive_unlabeled_sequences'] = merged_train_sequences[unlabeled_indices].tolist()
    submodel_fold_training_dict['transductive_complement_ids'] = merged_train_ids[
        complement_unlabeled_indices
    ].tolist()
    submodel_fold_training_dict['transductive_complement_sequences'] = merged_train_sequences[
        complement_unlabeled_indices
    ].tolist()

    return submodel_fold_training_dict


def bootstrap_ensemble_submodel_dataset(path: str ,ensemble_dict:dict) -> list:
    fold_training_dicts = utils.load_as_pickle(path)
    submodel_fold_training_dicts = []
    resolved_groups_root = resolve_groups_root(ensemble_dict)
    for i in range(len(fold_training_dicts)):
        fold_training_dict = fold_training_dicts[i]
        if ensemble_dict.get('algorithm_name') == 'transductive_pu_learning':
            submodel_fold_training_dict = create_transductive_cross_validation_dataset_for_folds(
                fold_training_dict=fold_training_dict,
                submodel_index=ensemble_dict['submodel_index'],
                groups_path=resolved_groups_root,
                fold_index=i,
            )
        else:
            submodel_fold_training_dict = create_cross_validation_dataset_for_folds(
                fold_training_dict=fold_training_dict,
                submodel_index=ensemble_dict['submodel_index'],
                groups_path=resolved_groups_root,
                fold_index=i,
            )
        submodel_fold_training_dicts.append(submodel_fold_training_dict)
    submodel_cross_validation_datasets = CrossValidationDataset(fold_dicts=submodel_fold_training_dicts)
    return submodel_cross_validation_datasets

def bootstrap_model_dataset(path: str) -> list:
    fold_training_dicts = utils.load_as_pickle(path)
    cross_validation_dataset = CrossValidationDataset(fold_dicts=fold_training_dicts)
    return cross_validation_dataset

dataset_to_bootstrapper = {
    'ensemble_submodel': bootstrap_ensemble_submodel_dataset,
    'single_model': bootstrap_model_dataset
}

def build_dataset_from_configuration(name: str,path:str,kwargs: dict) -> Union[torch.nn.Module, AUCMLoss]:
    print(f"keyword arguments for dataset: {kwargs}")
    supported_losses = list(dataset_to_bootstrapper.keys())
    if name not in dataset_to_bootstrapper.keys():
        raise Exception(f'dataset: {name} not supported. supported losses: {supported_losses}')
    return dataset_to_bootstrapper[name](path = path,**kwargs)