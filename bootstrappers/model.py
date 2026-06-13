from typing import Tuple, Union
from transformers import EsmForSequenceClassification, AutoConfig
from models.Esm2_with_LA import ESMWithLightAttentionHead
import torch
from peft import LoraConfig, get_peft_model,PeftModel,TaskType
import re
from libauc.losses import AUCMLoss
import os
from models.Convolution import CNNModel
from torchsummary import summary

def make_lora(model:Union[torch.nn.Module,EsmForSequenceClassification],lora_kwargs:dict,extra_kwargs:dict)-> PeftModel:
    lora_kwargs['fold_index'] = extra_kwargs['fold_index']
    lora_model = get_lora_model(model, lora_kwargs)
    lora_model.print_trainable_parameters()
    return lora_model

def bootstrap_esm2_with_classification_head(model_name: str,num_labels: int,lora_kwargs:dict,extra_kwargs:dict) -> EsmForSequenceClassification:
    model = EsmForSequenceClassification.from_pretrained(model_name, num_labels= num_labels)
    return make_lora(model = model,lora_kwargs=lora_kwargs,extra_kwargs=extra_kwargs)

def bootstrap_esm2_with_LA(model_name: str,  num_labels: int , dout: int , kernel_size: int,use_max:bool,lora_kwargs:dict,extra_kwargs:dict) -> torch.nn.Module:
    esm2_config = AutoConfig.from_pretrained(model_name,trust_remote_code=True) 
    esm2_config.num_labels = num_labels
    esm2_config.dout = dout
    esm2_config.kernel_size = kernel_size
    esm2_config.use_max = use_max
    print(f"esm2_config: {esm2_config}")  
    model = ESMWithLightAttentionHead(config=esm2_config,device=extra_kwargs['device'],loss_fct=extra_kwargs['loss_fct'])
    return make_lora(model = model,lora_kwargs=lora_kwargs,extra_kwargs=extra_kwargs)

def bootstrap_esm2_with_LA_no_lora(model_name: str,  num_labels: int , dout: int , kernel_size: int,use_max:bool,extra_kwargs:dict) -> torch.nn.Module:
    esm2_config = AutoConfig.from_pretrained(model_name) 
    esm2_config.num_labels = num_labels
    esm2_config.dout = dout
    esm2_config.kernel_size = kernel_size
    esm2_config.use_max = use_max
    print(f"esm2_config: {esm2_config}")  
    model = ESMWithLightAttentionHead(config=esm2_config,device=extra_kwargs['device'],loss_fct=extra_kwargs['loss_fct'])

    unfreeze_keys = ("light_attention", "dropout", "classifier")
    for name, param in model.named_parameters():
        if any(name.startswith(key) for key in unfreeze_keys):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def bootstrap_with_convolution(filters:int, kernel_size:int, num_layers:int,padding : str,dilation:int,existing_path:str,extra_kwargs:dict):
    model = CNNModel(filters=filters, kernel_size=kernel_size, num_layers=num_layers,padding = padding,dilation = dilation).to(extra_kwargs['device'])
    if existing_path is not None:
        ckpt_path = os.path.join(
            existing_path,
            f'fold_{extra_kwargs["fold_index"]}',
            'model',
            'model.pt'
        )
        state_dict = torch.load(ckpt_path, map_location=extra_kwargs['device'])
        model.load_state_dict(state_dict)
    summary(model, (20, 50))
    return model 

model_to_bootstrapper = {
    'esm2_with_classification_head': bootstrap_esm2_with_classification_head,
    'esm2_with_LA_head': bootstrap_esm2_with_LA  ,
    'convolution': bootstrap_with_convolution,
    'esm2_LA_no_lora': bootstrap_esm2_with_LA_no_lora
}

def get_target_modules(model: EsmForSequenceClassification,target_modules:list) -> list:
    lora_target_modules_regexs = target_modules
    target_modules = []
    for regex in lora_target_modules_regexs:
        for name, _ in model.named_modules():
            if re.match(regex, name):
                target_modules.append(name)
    return target_modules

def get_lora_config(model: EsmForSequenceClassification, target_modules:list,r:int,lora_alpha:int,lora_dropout:float,modules_to_save:list)-> LoraConfig:
    lora_target_modules = get_target_modules(model,target_modules)
    print(f'lora_target_modules: {lora_target_modules}')
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode= False,
        target_modules= lora_target_modules,
        r= r,
        lora_alpha= lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save = modules_to_save,
    )
    return lora_config

def get_lora_model(model: EsmForSequenceClassification, lora_kwargs:dict) -> PeftModel:
    if 'existing_path' in lora_kwargs:
        lora_model = PeftModel.from_pretrained(model,
                os.path.join(lora_kwargs['existing_path'],f'fold_{lora_kwargs["fold_index"]}','model'))
    else:
        fold_index = lora_kwargs.pop('fold_index')
        lora_config = get_lora_config(model, **lora_kwargs) 
        lora_kwargs['fold_index'] = fold_index
        lora_model = get_peft_model(model, lora_config)
    return lora_model

def build_model_from_configuration(name: str, device:torch.device ,
                                   loss_fct:Union[torch.nn.Module, AUCMLoss],
                                   fold_index:int,kwargs: dict) -> Union[torch.nn.Module, PeftModel]:
    
    supported_models = list(model_to_bootstrapper.keys())
    if name not in model_to_bootstrapper.keys():
        raise Exception(
            f'model: {name} not supported. supported models: {supported_models}')
    extra_kwargs = {
         'loss_fct': loss_fct,
         'device': device,
         'fold_index': fold_index,
        }
    
    model = model_to_bootstrapper[name](extra_kwargs=extra_kwargs,**kwargs) 
    # print the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    return model
 