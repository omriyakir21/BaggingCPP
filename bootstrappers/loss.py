from typing import Union
import torch
from libauc.losses import AUCMLoss
from models.dataset import Dataset


def bootstrap_bce(dataset:Dataset,device:torch.device) -> torch.nn.Module:    
    train_labels = dataset.train_labels
    class_weights = Dataset.calculate_class_weights(train_labels).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

def bootstrap_auc_margin(margin: float,dataset: Dataset ,device:torch.device) -> AUCMLoss:
    assert 0<=margin<=1, f'margin should be in [0,1], got {margin}'
    positive_ratio = float(sum(dataset.train_labels) / len(dataset.train_labels))
    return AUCMLoss(margin = margin,imratio = positive_ratio,device=device)

loss_to_bootstrapper = {
    'binary_cross_entropy': bootstrap_bce,
    'auc_margin': bootstrap_auc_margin
}

def build_loss_from_configuration(name: str,dataset:Dataset,device:torch.device, kwargs: dict) -> Union[torch.nn.Module, AUCMLoss]:
    supported_losses = list(loss_to_bootstrapper.keys())
    if name not in loss_to_bootstrapper.keys():
        raise Exception(f'loss: {name} not supported. supported losses: {supported_losses}')
    return loss_to_bootstrapper[name](dataset=dataset,device=device,**kwargs)