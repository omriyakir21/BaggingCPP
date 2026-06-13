import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Union
from utils import load_as_pickle
from datasets import Dataset as HFDataset
import torch

class Dataset:
    def __init__(self, fold: dict):
        self.fold = fold

        self.train_set = self.fold['sequences_train']
        self.validation_set = self.fold['sequences_validation']
        self.test_set = self.fold['sequences_test']

        self.train_labels = self.fold['labels_train']
        self.validation_labels = self.fold['labels_validation']
        self.test_labels = self.fold['labels_test']

        self.train_ids = self.fold['ids_train']
        self.validation_ids = self.fold['ids_validation']
        self.test_ids = self.fold['ids_test']

        self.train_descriptions = self.fold['descriptions_train']
        self.validation_descriptions = self.fold['descriptions_validation']
        self.test_descriptions = self.fold['descriptions_test']

        assert len(self.train_set) == len(self.train_labels)
        assert len(self.validation_set) == len(self.validation_labels)
        assert len(self.test_set) == len(self.test_labels)
    @staticmethod
    def calculate_class_weights(labels):
        # Ensure labels is a PyTorch tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int64)  # Use float32 or another appropriate dtype
        # Calculate the number of positives and negatives
        num_negatives = torch.sum(labels == 0).item()  # Count zeros in the tensor
        num_positives = torch.sum(labels == 1).item()  # Count ones in the tensor
        # Compute class weight ratio
        ratio = num_negatives / num_positives
        class_weights = torch.tensor([ratio], dtype=torch.float32)

        return class_weights

    @staticmethod
    def tokenize_function(tokenizer,sequences, labels):
        tokens = tokenizer(list(sequences), truncation=True, padding="max_length", max_length=50, return_tensors='pt')
        tokens["labels"] = list(labels)
        return tokens

    @staticmethod
    def create_hf_dataset_for_classification(sequences, labels,tokenizer):
        """
        Create a Hugging Face dataset for classification.
        """
        # in your bootstrap_train, immediately after creating val_dataset
        dataset = HFDataset.from_dict(Dataset.tokenize_function(tokenizer,sequences, labels))
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return dataset



class CrossValidationDataset:
    def __init__(self,fold_dicts:dict):
        self.fold_dicts = fold_dicts
        self.fold_datasets = []

        for fold in self.fold_dicts:
            fold_dataset = Dataset(fold)
            self.fold_datasets.append(fold_dataset)

