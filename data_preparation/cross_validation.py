
import os

import data_preparation.cross_validation_utils as cv_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_for_training_dir", default="datasets/data_for_training")
parser.add_argument("--full_dataset_path", default="datasets/full_datasets/bagging_cpp_dataset.csv")
parser.add_argument("--num_folds", type=int, default=5)

if __name__ == '__main__':
    args = parser.parse_args()
    data_for_training_dir = args.data_for_training_dir
    full_dataset_path = args.full_dataset_path
    num_folds = args.num_folds
    os.makedirs(data_for_training_dir, exist_ok=True)
    cv_utils.partition_to_folds_and_save(data_for_training_dir=data_for_training_dir, dataset_path = full_dataset_path, num_folds=num_folds) 