import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.Convolution import CNNModel
import numpy as np
import torch


def predict_binary_probs_from_sequences_convolution(model: CNNModel,
                         sequences: list,
                         device:torch.device,
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

    predictions = model.predict_with_convolution(
            processed_sequences = CNNModel.process_sequences_convolution(sequences=sequences,
                                                                         device=device))
    return predictions


def load_convolution_model(model_path:str,filters:int, kernel_size:int, num_layers:int,padding : str,dilation:int,device:torch.device):
    model = CNNModel(filters=filters, kernel_size=kernel_size, num_layers=num_layers,padding = padding,dilation = dilation).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model 