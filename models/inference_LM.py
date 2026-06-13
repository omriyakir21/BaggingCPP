import torch
import os
import sys
sys
from peft import PeftModel
from transformers import PreTrainedTokenizer, AutoConfig
from models.dataset import Dataset
import numpy as np
from utils import save_as_pickle
from typing import Union
from models.Esm2_with_LA import ESMWithLightAttentionHead

def inference(model: Union[PeftModel, ESMWithLightAttentionHead], dataset: Dataset, tokenizer: PreTrainedTokenizer, device: torch.device) -> dict:
    name_to_set = {
        'train': {'data': dataset.train_set, 'labels': dataset.train_labels},
        'validation': {'data': dataset.validation_set, 'labels': dataset.validation_labels},
        'test': {'data': dataset.test_set, 'labels': dataset.test_labels}}
    name_to_yhat = {}
    for name, my_set in name_to_set.items():
        if len(my_set['data']) == 0:
            continue
        name_to_yhat[name] = {
            'predictions': predict_binary_probs(
                model,
                tokenizer(
                    my_set['data'],
                    truncation=True,
                    padding="max_length",
                    max_length=50,
                    return_tensors='pt',
                ),
                device,
            ),
            'labels': my_set['labels'],
            'sequences': my_set['data'],
        }
    return name_to_yhat

def predict_binary_probs(model: PeftModel,
                         inputs: dict,
                         device: torch.device,
                         batch_size: int = 512,
                         return_embeddings: bool = False) -> np.ndarray:
    """
    Perform inference using a LoRA fine-tuned model for binary classification in batches.

    Args:
        model:     The trained PeftModel.
        inputs:    Tokenized inputs (a dict of tensors).
        device:    Device on which to run inference.
        batch_size:Maximum number of examples per forward pass.

    Returns:
        probs: numpy array of shape (N, ...) with predicted probabilities.
    """
    model.eval()
    # ensure all tensors are on the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    N = next(iter(inputs.values())).size(0)
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch_inputs = {k: v[start:start+batch_size] for k, v in inputs.items()}
            if return_embeddings:
                outputs = model(**batch_inputs, return_embeddings=return_embeddings)
            else:
                outputs = model(**batch_inputs)
            if return_embeddings:
                outputs, embeddings = outputs
                all_embeddings.append(embeddings.cpu().numpy())
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_preds = np.concatenate(all_probs, axis=0)
    if return_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_preds, all_embeddings
    return all_preds


def predict_binary_probs_from_sequences(model: PeftModel,
                         sequences: list,
                         tokenizer: PreTrainedTokenizer,
                         device: torch.device,
                         return_embeddings:bool = False,
                         batch_size: int = 512) -> np.ndarray:
    """
    Perform inference using a LoRA fine-tuned model for binary classification in batches.

    Args:
        model:     The trained PeftModel.
        sequences: List of sequences to predict.
        tokenizer: Tokenizer to use for encoding the sequences.
        device:    Device on which to run inference.
        batch_size:Maximum number of examples per forward pass.

    Returns:
        probs: numpy array of shape (N, ...) with predicted probabilities.
    """
    
    inputs = tokenizer(sequences, truncation=True, padding="max_length", max_length=50, return_tensors='pt')
    outputs = predict_binary_probs(model=model, inputs=inputs, device=device, batch_size=batch_size, return_embeddings=return_embeddings)
    return outputs

def load_esm2_with_LA_lora_model(model_path:str, device:torch.device,model_name:str,num_labels:int, dout:int, kernel_size:int,use_max:bool):
    """
    Load a LoRA fine-tuned ESM-2 model with light attention.
    Args:
        model_path: Path to the LoRA fine-tuned model.
        device:     Device on which to load the model.
        model_name: Name of the base ESM-2 model.
        num_labels: Number of labels for classification.
        dout:       Output dimension for the light attention head.
        kernel_size: Kernel size for the light attention head.
    Returns:
        model:      The LoRA fine-tuned ESM-2 model.
    """
    from models.Esm2_with_LA import ESMWithLightAttentionHead
    esm2_config = AutoConfig.from_pretrained(model_name) 
    esm2_config.num_labels = num_labels
    esm2_config.dout = dout
    esm2_config.kernel_size = kernel_size 
    esm2_config.use_max = use_max

    model = ESMWithLightAttentionHead(config=esm2_config,device=device,loss_fct=None)
    lora_model = PeftModel.from_pretrained(model, model_path)
    lora_model = lora_model.to(device)
    return lora_model

def load_tokenizer(model_name:str):
    """
    Load the tokenizer for the specified model.
    Args:
        model_name: Name of the base model.
    Returns:
        tokenizer:  The tokenizer for the specified model.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer