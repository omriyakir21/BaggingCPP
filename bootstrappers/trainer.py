import os
import sys
import torch
from models.trainer import CustomTrainer
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from transformers.trainer_callback import EarlyStoppingCallback
from models.metrics import metrics_evaluation
from typing import Union
from libauc.losses import AUCMLoss
from libauc.sampler import DualSampler
from peft import PeftModel
from torch.utils.data import DataLoader


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=8,  # Number of epochs to wait for improvement
    early_stopping_threshold=0.0,  # Minimum change to consider as improvement
)


def bootstrap_pesg_trainer(train_dataset:HFDataset,val_dataset:HFDataset,
                          loss_fct:torch.nn.Module,model:Union[PeftModel,torch.nn.Module],
                          training_arguments:TrainingArguments,
                          optimizer:torch.optim.Optimizer,
                          device:torch.device) -> Trainer:
    print(train_dataset.column_names)
    labels = train_dataset['labels']
    print(f" labels size: {len(labels)}")
    train_sampler = DualSampler(dataset=train_dataset,
                                batch_size=training_arguments.per_device_train_batch_size ,
                                sampling_rate=0.5,labels=labels)
    train_loader = DataLoader(train_dataset,batch_size=training_arguments.per_device_train_batch_size, sampler=train_sampler)      
    
    return CustomTrainer(
        custom_train_loader=train_loader,
        device=device,
        loss_fct=loss_fct,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics= metrics_evaluation,
        optimizers = (optimizer, None),
    )

def bootstrap_regular_trainer(train_dataset:HFDataset,val_dataset:HFDataset,
                              loss_fct:torch.nn.Module,model:Union[PeftModel,torch.nn.Module],training_arguments:TrainingArguments,
                              optimizer:torch.optim.Optimizer,device:torch.device) -> Trainer:    
    return CustomTrainer(
        custom_train_loader=None,
        device=device,
        loss_fct=loss_fct,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics= metrics_evaluation,
        optimizers = (optimizer, None),
    )   
     
training_arguments_to_bootstrapper = {
    'regular': bootstrap_regular_trainer,
    'pesg':bootstrap_pesg_trainer
}

def build_trainer_from_configuration(name: str,train_dataset:HFDataset,val_dataset:HFDataset,training_arguments:TrainingArguments,
                                     loss_fct:Union[torch.nn.Module, AUCMLoss],optimizer:torch.optim.Optimizer,
                                     model:Union[PeftModel,torch.nn.Module],device:torch.device, kwargs: dict) -> Trainer:
    supported_losses = list(training_arguments_to_bootstrapper.keys())
    if name not in training_arguments_to_bootstrapper.keys():
        raise Exception(f'loss: {name} not supported. supported losses: {supported_losses}')
    return training_arguments_to_bootstrapper[name](optimizer=optimizer,loss_fct=loss_fct,
                                                    train_dataset=train_dataset,val_dataset=val_dataset,
                                                    training_arguments=training_arguments,model=model,
                                                    device=device,**kwargs)

