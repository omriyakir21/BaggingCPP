import torch 
from torch.optim import Adam, AdamW
from typing import Union
from peft import PeftModel


def bootstrap_adam(learning_rate: float,model:Union[PeftModel,torch.nn.Module],weight_decay: float ) -> torch.optim.Optimizer:
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def bootstrap_adamw(learning_rate: float,model:Union[PeftModel,torch.nn.Module],weight_decay: float) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def bootstrap_sgd(learning_rate: float,model:Union[PeftModel,torch.nn.Module],momentum: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

optimizer_to_bootstrapper = {
    'sgd': bootstrap_sgd,
    'adam': bootstrap_adam,
    'adamw': bootstrap_adamw,    
}
def create_args_for_optimizer(name: str,loss_fct:Union[torch.nn.Module],device:torch.device, kwargs: dict)-> dict:
    args= kwargs.copy()
    if name == 'pesg':
        args['loss_fct'] = loss_fct
        args['device'] = device
    return args

def build_optimizer_from_configuration(name: str,model:Union[PeftModel,torch.nn.Module],loss_fct:Union[torch.nn.Module],device:torch.device, kwargs: dict) -> torch.optim.Optimizer:
    supported_optimizers = list(optimizer_to_bootstrapper.keys())
    if name not in optimizer_to_bootstrapper.keys():
        raise Exception(f'optimizer: {name} not supported. supported optimizers: {supported_optimizers}')
    kwargs['model'] = model
    args = create_args_for_optimizer(name,loss_fct=loss_fct,device=device, kwargs=kwargs)
    return optimizer_to_bootstrapper[name](**args)