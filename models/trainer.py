# 
from transformers import Trainer
import torch
# from torch.utils.data import DataLoader
# from transformers.trainer_utils import (
#     EvalLoopOutput,
#     EvalPrediction,
#     denumpify_detensorize,
#     has_length,
#     IterableDatasetShard,
#     find_batch_size,
#     EvalLoopContainer,
# )
from transformers.modeling_outputs import SequenceClassifierOutput
# from transformers.trainer_pt_utils import (
#     deepspeed_init,
# )
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import logging
import time
class CustomTrainer(Trainer):
    def __init__(self, custom_train_loader,loss_fct,device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_loader = custom_train_loader
        self.loss_fct = loss_fct
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.custom_train_loader is not None:
            out: SequenceClassifierOutput = model(**inputs)
            loss = out.loss
            logits = out.logits
            if return_outputs:
                # return a dict so Trainer picks up only your logits
                return loss, {"logits": logits}
            return loss
        else:
            try:
                labels = inputs.pop("labels").to(self.device).float()
                outputs = model(**inputs)
                logits = outputs.get("logits").view(-1).to(self.device)
                loss = self.loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
            except Exception as e:
                print(f"Error in compute_loss: {e}")
                raise e
    
    def get_train_dataloader(self):
        if self.custom_train_loader is not None:
            return self.custom_train_loader
        return super().get_train_dataloader()
    

       # def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
    #     print("in compute loss")
    #     # labels = inputs.pop("labels").to(self.device).float()
    #     sequence_classifier_output = model(**inputs)
    #     logits = sequence_classifier_output.logits
    #     loss = sequence_classifier_output.loss
    #     # print(f"logits shape: {logits.shape}")
    #     # print(f"labels shape : {labels.shape}")
    #     # loss = self.loss_fct(logits, labels)  
    #     return (loss, logits) if return_outputs else loss
    


    
