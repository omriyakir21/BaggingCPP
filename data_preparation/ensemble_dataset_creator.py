import os
import sys
import numpy as np
import random
import utils
import itertools
from data_preparation.cross_validation_utils import (
    cluster_sequences,
    create_cluster_participants_indices,
)
import argparse

def sample_training_sequences(iteration: int, negative_size: float, folds_traning_dicts: list):
    training_sequences_folds = [fold['sequences_train'] for fold in folds_traning_dicts]
    label_folds = [fold['labels_train'] for fold in folds_traning_dicts]
    positive_indexes = [np.where(label_fold == np.int64(1))[0] for label_fold in label_folds]
    print(f'positive indices example shape: {positive_indexes[0].shape}')
    unlabeled_indexes = [np.where(label_fold == np.int64(0))[0] for label_fold in label_folds]
    print(f'unlabeled indices example shape: {unlabeled_indexes[0].shape}')
    folds_groups_for_train = []
    for i in range(len(training_sequences_folds)):
        groups_for_train = create_groups(unlabeled_indexes[i], negative_size, iteration)
        folds_groups_for_train.append(groups_for_train)
    subsampled_training_sequences_folds = []
    subsampled_labels_folds = []
    for i in range(len(training_sequences_folds)):
        fold_groups_for_train = folds_groups_for_train[i]
        groups_indices = [np.concatenate([positive_indexes[i], group_for_train]) for group_for_train in fold_groups_for_train]
        sequences = [np.array(training_sequences_folds[i])[indices] for indices in groups_indices]
        labels = [np.concatenate([np.ones(positive_indexes[i].shape[0]), np.zeros(len(group_for_train))]) for group_for_train in fold_groups_for_train]
        subsampled_training_sequences_folds.append(sequences)
        subsampled_labels_folds.append(labels)
    return subsampled_training_sequences_folds,subsampled_labels_folds



def create_groups(clusters_participants_list: list, fraction: float, iterations_number: int):
    groups_for_train = []
    for _ in range(iterations_number):
        selected_clusters_indices = random.choices(clusters_participants_list, k=int(fraction*len(clusters_participants_list)))
        group_indices = np.concatenate([list(cluster) for cluster in selected_clusters_indices])
        groups_for_train.append(group_indices)
    return  groups_for_train


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--algorithm_name", default='inductive_pu_learning') # 'inductive', 'transductive_pu_learning'
    parser.add_argument("--iteration_grid", nargs='+', type=int, default=[50])
    parser.add_argument("--negative_size_grid", nargs='+', type=float, default=[0.3])
    parser.add_argument("--folds_training_dicts_path", default='datasets/data_for_training/folds_training_dicts.pkl')
    args = parser.parse_args()
    
    num_folds = args.num_folds
    alogrithm_name = args.algorithm_name
    iteration_grid = args.iteration_grid
    negative_size_grid = args.negative_size_grid
    folds_training_dicts_path = args.folds_training_dicts_path
    folds_training_dicts = utils.load_as_pickle(folds_training_dicts_path)
    output_dir = os.path.join('datasets','data_for_training','ensembles',alogrithm_name)
    os.makedirs(output_dir, exist_ok=True)
    

    label_folds = [fold['labels_train'] for fold in folds_training_dicts]
    sequences_folds = [fold['sequences_train'] for fold in folds_training_dicts]
    unlabeled_indexes = [np.where(label_fold == np.int64(0))[0] for label_fold in label_folds]

    clusters_participants_list_folds = None
    transductive_unlabeled_global_indices_folds = None
    if alogrithm_name == 'transductive_pu_learning':
        print('clustering merged 4-fold unlabeled pools for transductive_pu_learning')
        transductive_unlabeled_global_indices_folds = []
        clusters_participants_list_folds = []
        for i in range(num_folds):
            merged_sequences = np.array(
                folds_training_dicts[i]['sequences_train'] + folds_training_dicts[i]['sequences_test']
            )
            merged_labels = np.array(
                folds_training_dicts[i]['labels_train'] + folds_training_dicts[i]['labels_test']
            )
            unlabeled_global_indices = np.where(merged_labels == np.int64(0))[0]
            transductive_unlabeled_global_indices_folds.append(unlabeled_global_indices)

            unlabeled_sequences = merged_sequences[unlabeled_global_indices].tolist()
            clustered_indices, _ = cluster_sequences(unlabeled_sequences, seqid=0.5, coverage=0.4)
            clusters_participants_list_folds.append(
                create_cluster_participants_indices(clustered_indices)
            )
    else:
        print('clustering sequences for version v2')
        clustered_indices_folds = [cluster_sequences(sequences_folds[i], seqid=0.5, coverage=0.4)[0] for i in range(num_folds)]
        clusters_participants_list_folds = [create_cluster_participants_indices(clustered_indices_folds[i]) for i in range(num_folds)]

    for iteration, negative_size in itertools.product(iteration_grid,negative_size_grid):
        groups_dir = os.path.join(output_dir,f'iteration_{iteration}_negative_size_{negative_size}')
        os.makedirs(groups_dir, exist_ok=True)
        for i in range(num_folds):
            clusters_participants_list = clusters_participants_list_folds[i]
            groups_for_train = create_groups(clusters_participants_list=clusters_participants_list,
                                                fraction=negative_size,
                                                iterations_number=iteration,
                                                )

            if alogrithm_name == 'transductive_pu_learning':
                unlabeled_global_indices = transductive_unlabeled_global_indices_folds[i]
                groups_for_train = [
                    unlabeled_global_indices[np.array(group_indices, dtype=np.int64)]
                    for group_indices in groups_for_train
                ]
            fold_dir = os.path.join(groups_dir, f'fold_{i}')
            os.makedirs(fold_dir, exist_ok=True)
            utils.save_as_pickle(groups_for_train, os.path.join(fold_dir, 'groups_for_ensemble.pkl'))
