
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.Convolution import CNNModel
from peft import PeftModel
import torch
from typing import Union
from models.Esm2_with_LA import ESMWithLightAttentionHead

def bootstrap_LM_save_model(model:Union[PeftModel,ESMWithLightAttentionHead],save_dir:str) -> None:  
    model.save_pretrained(save_dir)
    print(f'model save path {save_dir}...')

def bootstrap_convolution_save_model(model:CNNModel,save_dir:str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model.pt'))
    print(f'model save path {save_dir}...')

def bootstrap_LA_no_lora_save_model(model:Union[PeftModel,ESMWithLightAttentionHead],save_dir:str) -> None:  
    model.save_LA_head(save_dir)
    print(f'model save path {save_dir}...')


save_model_to_bootstrapper = {
    'esm2_with_LA_head': bootstrap_LM_save_model,
    'convolution': bootstrap_convolution_save_model,
    'esm2_LA_no_lora': bootstrap_LA_no_lora_save_model,
    'esm2_with_classification_head': bootstrap_LM_save_model,
}

def save_model_from_parameters(name: str,model:Union[CNNModel,PeftModel,ESMWithLightAttentionHead],save_dir:str) -> None:
    supported_training_functions = list(save_model_to_bootstrapper.keys())
    if name not in save_model_to_bootstrapper.keys():
        raise Exception(f'train_function: {name} not supported. supported training functions: {supported_training_functions}')
    save_model_to_bootstrapper[name](model=model,save_dir=save_dir)