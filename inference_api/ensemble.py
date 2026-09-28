"""Persistent LoRA ensemble for repeated inference.

The CLI in `inference.py` scores one FASTA and exits, so reloading every adapter from
disk costs nothing it cares about. A design loop calls the ensemble thousands of times,
where those reloads dominate the runtime. `LoRAEnsemble` pays the loading cost once:
the ESM2 base is built a single time and all `num_folds * num_submodels` adapters are
attached to it as named PEFT adapters, so scoring a batch only switches the active
adapter. Predictions are unchanged - `set_adapter` selects exactly the weights that
`PeftModel.from_pretrained` would have loaded.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig

from evaluation.utils_evaluation import get_experiment_base_paths_for_ensemble
from models.Esm2_with_LA import ESMWithLightAttentionHead
from models.inference_LM import load_tokenizer, predict_binary_probs

DEFAULT_WORK_DICT = {
    'hypothesis': 'ensemble_inductive_pu_learning',
    'experiment': 'groups_inductive',
    'model_name': 'facebook/esm2_t6_8M_UR50D',
    'num_submodels': 50,
    'num_folds': 5,
    'model_folder_path': 'results_upload/model_folder/ensemble',
    'bagging_cpp_dataset_path': 'datasets/full_datasets/bagging_cpp_dataset.csv',
    'num_labels': 1,
    'dout': 128,
    'kernel_size': 7,
    'use_max': True,
}

MAX_SEQUENCE_LENGTH = 50


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class LoRAEnsemble:
    """Holds the base model, every adapter and the fold lookup table in memory."""

    def __init__(self,
                 work_dict: Optional[dict] = None,
                 device: Optional[torch.device] = None,
                 num_submodels_max: int = 50,
                 use_custom_model: bool = False,
                 load_fold_map: bool = True,
                 fast_adapter_switch: bool = True,
                 verbose: bool = True):
        self.work_dict = dict(DEFAULT_WORK_DICT)
        if work_dict:
            self.work_dict.update(work_dict)
        self.work_dict['num_submodels'] = min(self.work_dict['num_submodels'], num_submodels_max)
        self.device = device if device is not None else get_device()
        self.fast_adapter_switch = fast_adapter_switch
        self.verbose = verbose
        if self.verbose:
            print(f"Using device: {self.device}")

        self.tokenizer = load_tokenizer(model_name=self.work_dict['model_name'])
        self.model: Optional[PeftModel] = None
        # fold -> [adapter name], ordered by submodel index; only members that loaded.
        self.adapter_names: Dict[int, List[str]] = {}
        self.failed_members: List[Tuple[int, int]] = []
        self._sequence_to_fold: Optional[dict] = None

        self._load_all_adapters(use_custom_model=use_custom_model)
        if load_fold_map:
            self._load_fold_map()

    # ------------------------------------------------------------------ loading

    def _member_paths(self, use_custom_model: bool) -> Dict[Tuple[int, int], str]:
        """(submodel index, fold) -> adapter directory."""
        if use_custom_model:
            base_paths, _ = get_experiment_base_paths_for_ensemble(
                experiment=self.work_dict['experiment'],
                base_paths={},
                hypothesis_path=os.path.join('results', 'hypothesis', self.work_dict['hypothesis']),
                num_submodels=self.work_dict['num_submodels'])
            roots = list(base_paths[self.work_dict['experiment']])
        else:
            roots = [os.path.join(self.work_dict['model_folder_path'], f'submodel_{index}')
                     for index in range(self.work_dict['num_submodels'])]

        paths = {}
        for fold in range(self.work_dict['num_folds']):
            for index, root in enumerate(roots):
                paths[(index, fold)] = os.path.join(root, f'fold_{fold}', 'model')
        return paths

    def _build_base_model(self) -> ESMWithLightAttentionHead:
        config = AutoConfig.from_pretrained(self.work_dict['model_name'])
        config.num_labels = self.work_dict['num_labels']
        config.dout = self.work_dict['dout']
        config.kernel_size = self.work_dict['kernel_size']
        config.use_max = self.work_dict['use_max']
        return ESMWithLightAttentionHead(config=config, device=self.device, loss_fct=None)

    def _load_all_adapters(self, use_custom_model: bool) -> None:
        member_paths = self._member_paths(use_custom_model)
        base_model = self._build_base_model()
        self.adapter_names = {fold: [] for fold in range(self.work_dict['num_folds'])}

        iterator = sorted(member_paths.items(), key=lambda item: (item[0][1], item[0][0]))
        if self.verbose:
            iterator = tqdm(iterator, desc="Loading ensemble adapters")
        for (index, fold), model_path in iterator:
            adapter_name = f'submodel_{index}_fold_{fold}'
            try:
                if self.model is None:
                    self.model = PeftModel.from_pretrained(base_model, model_path,
                                                           adapter_name=adapter_name)
                else:
                    self.model.load_adapter(model_path, adapter_name=adapter_name)
            except Exception as e:
                print(e)
                print(f'Error loading fold {fold} for submodel {index}')
                self.failed_members.append((index, fold))
                continue
            self.adapter_names[fold].append(adapter_name)

        if self.model is None:
            raise RuntimeError('No ensemble adapter could be loaded; check model_folder_path.')
        self.model = self.model.to(self.device)
        self.model.eval()
        self._cache_adapter_switch_targets()
        if self.verbose:
            loaded = sum(len(names) for names in self.adapter_names.values())
            print(f'Loaded {loaded} ensemble members ({len(self.failed_members)} failed).')

    def _load_fold_map(self) -> None:
        """Read the 450k-row dataset once and keep the sequence -> fold lookup."""
        path = self.work_dict['bagging_cpp_dataset_path']
        # Only these two columns are used. Reading just them avoids the DtypeWarning raised by the
        # free-text 'description' column and keeps the 450k-row load cheap.
        try:
            df = pd.read_csv(path, usecols=['sequence', 'test_fold_index'])
        except ValueError as e:
            raise ValueError("The input CSV must contain 'sequence' and 'test_fold_index' columns.") from e
        self._sequence_to_fold = dict(zip(df['sequence'], df['test_fold_index']))

    # ---------------------------------------------------------------- inference

    def get_indexes_dict(self, sequences: Sequence[str], no_cross_predictions: bool = False) -> dict:
        """Map fold index -> positions in `sequences`; -1 collects the unseen sequences."""
        fold_to_indices = {fold: [] for fold in range(self.work_dict['num_folds'])}
        fold_to_indices[-1] = []
        if no_cross_predictions:
            fold_to_indices[-1] = list(range(len(sequences)))
            return fold_to_indices

        if self._sequence_to_fold is None:
            self._load_fold_map()
        for idx, sequence in enumerate(sequences):
            fold_index = self._sequence_to_fold.get(sequence, -1)
            if fold_index in fold_to_indices:
                fold_to_indices[fold_index].append(idx)
            else:
                fold_to_indices[-1].append(idx)
        return fold_to_indices

    def _cache_adapter_switch_targets(self) -> None:
        """Collect the modules whose active adapter has to be flipped on every switch.

        `PeftModel.set_adapter` walks every adapter of every tuner layer to fix up
        `requires_grad`, so its cost grows with the number of loaded adapters - with 250
        members it dominated the runtime. Inference never needs those flags (everything
        runs under `torch.no_grad`), so `_set_active_adapter` writes `_active_adapter`
        directly on the cached modules, which is O(wrapped modules) instead of
        O(wrapped modules * adapters). Both classes read the active adapter from exactly
        this attribute, and `set_adapter` has no other effect here - our adapters are
        never merged, so its `unmerge()` branch is dead.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils.other import ModulesToSaveWrapper

        self._tuner_layers = []
        self._modules_to_save = []
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                if getattr(module, 'merged', False):
                    raise RuntimeError('Adapters are merged; the fast adapter switch expects unmerged layers.')
                self._tuner_layers.append(module)
            elif isinstance(module, ModulesToSaveWrapper):
                self._modules_to_save.append(module)

    def _set_active_adapter(self, adapter_name: str) -> None:
        if not self.fast_adapter_switch:
            self.model.set_adapter(adapter_name)
            return
        for layer in self._tuner_layers:
            layer._active_adapter = [adapter_name]
        for wrapper in self._modules_to_save:
            wrapper._active_adapter = adapter_name
        self.model.active_adapter = adapter_name

    def _tokenize(self, sequences: Sequence[str]) -> dict:
        inputs = self.tokenizer(list(sequences), truncation=True, padding="max_length",
                                max_length=MAX_SEQUENCE_LENGTH, return_tensors='pt')
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self,
                sequences: Sequence[str],
                keys: Optional[Sequence[str]] = None,
                no_cross_predictions: bool = False,
                batch_size: int = 64,
                show_progress: bool = True) -> pd.DataFrame:
        """Score `sequences` with the in-memory ensemble and return the predictions frame."""
        sequences = list(sequences)
        if len(sequences) == 0:
            return pd.DataFrame({'sequence': [], 'label': [], 'prediction': [], 'model_uncertainty': []})

        indexes_dict = self.get_indexes_dict(sequences, no_cross_predictions=no_cross_predictions)
        if self.verbose:
            for fold in range(-1, self.work_dict['num_folds']):
                print(f'Fold {fold} has {len(indexes_dict[fold])} sequences to predict on.')

        # Tokenizing once instead of once per member - the sequences are the same for every
        # adapter, and for 250 members this was 250 redundant tokenizer passes.
        inputs = self._tokenize(sequences)
        non_specific_indices = indexes_dict[-1]

        ordered_predictions = np.empty(len(sequences), dtype=float)
        ordered_std = np.empty(len(sequences), dtype=float)
        non_specific_predictions = []

        folds = range(self.work_dict['num_folds'])
        if show_progress and self.verbose:
            folds = tqdm(folds, desc="Predicting using ensemble")
        for fold in folds:
            fold_indices = indexes_dict[fold]
            # One forward pass per member covers both the fold's own sequences and the unseen
            # ones; they are independent rows, so concatenating them changes nothing numerically.
            row_indices = fold_indices + non_specific_indices
            if len(row_indices) == 0 or len(self.adapter_names[fold]) == 0:
                if len(fold_indices) > 0:
                    ordered_predictions[fold_indices] = np.nan
                    ordered_std[fold_indices] = np.nan
                continue
            index_tensor = torch.as_tensor(row_indices, dtype=torch.long, device=self.device)
            fold_inputs = {k: v.index_select(0, index_tensor) for k, v in inputs.items()}

            fold_specific_predictions = []
            fold_non_specific_predictions = []
            for adapter_name in self.adapter_names[fold]:
                self._set_active_adapter(adapter_name)
                member_predictions = predict_binary_probs(model=self.model, inputs=fold_inputs,
                                                          device=self.device, batch_size=batch_size,
                                                          use_tqdm=False).reshape(len(row_indices), -1)
                if len(fold_indices) > 0:
                    fold_specific_predictions.append(member_predictions[:len(fold_indices)])
                if len(non_specific_indices) > 0:
                    fold_non_specific_predictions.append(member_predictions[len(fold_indices):])

            if len(fold_indices) > 0:
                ordered_std[fold_indices] = np.std(fold_specific_predictions, axis=0).reshape(-1)
                ordered_predictions[fold_indices] = np.mean(fold_specific_predictions, axis=0).reshape(-1)
            # Keep every submodel's prediction rather than averaging per fold, so the spread below is
            # measured across ensemble members - the same quantity ordered_std reports.
            non_specific_predictions.extend(fold_non_specific_predictions)

        if len(non_specific_indices) > 0:
            # (num_folds * num_submodels, num_sequences, 1)
            non_specific_stack = np.stack(non_specific_predictions, axis=0)
            ordered_predictions[non_specific_indices] = np.mean(non_specific_stack, axis=0).reshape(-1)
            ordered_std[non_specific_indices] = np.std(non_specific_stack, axis=0).reshape(-1)

        if self.verbose:
            print('Finished predictions.')
        return pd.DataFrame({
            'sequence': sequences,
            'label': list(keys) if keys is not None else list(range(len(sequences))),
            'prediction': ordered_predictions,
            'model_uncertainty': ordered_std,
        })
