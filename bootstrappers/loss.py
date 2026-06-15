import torch
from models.dataset import Dataset


def bootstrap_bce(dataset:Dataset,device:torch.device) -> torch.nn.Module:    
    train_labels = dataset.train_labels
    class_weights = Dataset.calculate_class_weights(train_labels).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

loss_to_bootstrapper = {
    'binary_cross_entropy': bootstrap_bce,
}

def build_loss_from_configuration(name: str,dataset:Dataset,device:torch.device, kwargs: dict) -> torch.nn.Module:
    supported_losses = list(loss_to_bootstrapper.keys())
    if name not in loss_to_bootstrapper.keys():
        raise Exception(f'loss: {name} not supported. supported losses: {supported_losses}')
    return loss_to_bootstrapper[name](dataset=dataset,device=device,**kwargs)