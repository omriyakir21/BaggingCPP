import torch
from transformers import TrainingArguments

def bootstrap_training_arguments(output_dir : str, logging_dir : str,batch_size : int,epochs: int,metric: str ) -> TrainingArguments:    
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        per_device_eval_batch_size  = batch_size,
        greater_is_better=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        logging_steps=10,
        )

training_arguments_to_bootstrapper = {
    'regular': bootstrap_training_arguments,
}

def build_training_arguments_from_configuration(name: str,output_dir:str,logging_dir:str, kwargs: dict) -> TrainingArguments:
    supported_losses = list(training_arguments_to_bootstrapper.keys())
    if name not in training_arguments_to_bootstrapper.keys():
        raise Exception(f'loss: {name} not supported. supported losses: {supported_losses}')
    return training_arguments_to_bootstrapper[name](output_dir=output_dir,logging_dir=logging_dir,**kwargs)



