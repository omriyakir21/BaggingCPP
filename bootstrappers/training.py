from typing import Union
from models.Convolution import CNNModel
from peft import PeftModel
from models.dataset import Dataset
import torch
from bootstrappers.training_arguments import build_training_arguments_from_configuration
from bootstrappers.trainer import build_trainer_from_configuration
from transformers import PreTrainedTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from models.metrics import calculate_roc_auc
import copy
import time

def bootstrap_LM_train_function(output_dir:str,logging_dir:str,training_configuration:dict,dataset:Dataset,
                                optimizer:torch.optim.Optimizer,model:PeftModel,device:torch.device,
                                loss_fct:Union[torch.nn.Module],tokenizer:PreTrainedTokenizer) -> PeftModel:  
    training_arguments = build_training_arguments_from_configuration(output_dir=output_dir,
        logging_dir=logging_dir,**training_configuration['compile']['training_arguments'])
    train_dataset = Dataset.create_hf_dataset_for_classification(dataset.train_set, dataset.train_labels,tokenizer)
    val_dataset = Dataset.create_hf_dataset_for_classification(dataset.validation_set, dataset.validation_labels,tokenizer)
    trainer = build_trainer_from_configuration(training_arguments=training_arguments,train_dataset=train_dataset,loss_fct=loss_fct,
                                               val_dataset=val_dataset,optimizer=optimizer,model=model,
                                               device=device,**training_configuration['compile']['trainer'])
    trainer.train()
    print(f"Fold done, best metric score: {trainer.state.best_metric:.4f}")
    return trainer.model

    
train_function_to_bootstrapper = {
    'LM': bootstrap_LM_train_function,
}

def run_training_function_from_parameters(name: str,fold_index:int, kwargs: dict) -> Union[CNNModel, PeftModel]:
    supported_training_functions = list(train_function_to_bootstrapper.keys())
    if name not in train_function_to_bootstrapper.keys():
        raise Exception(f'train_function: {name} not supported. supported training functions: {supported_training_functions}')
    print(f'Training fold {fold_index}...')
    start_time = time.time()
    model = train_function_to_bootstrapper[name](**kwargs)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    return model