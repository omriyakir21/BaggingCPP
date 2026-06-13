from typing import Union
from models.Convolution import CNNModel
from peft import PeftModel
from models.dataset import Dataset
import torch
from libauc.losses import AUCMLoss
from bootstrappers.training_arguments import build_training_arguments_from_configuration
from bootstrappers.trainer import build_trainer_from_configuration
from transformers import PreTrainedTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from models.metrics import calculate_roc_auc
import copy
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from libauc.optimizers import PESG
from libauc.sampler import DualSampler
import time

def bootstrap_LM_train_function(output_dir:str,logging_dir:str,training_configuration:dict,dataset:Dataset,
                                optimizer:torch.optim.Optimizer,model:PeftModel,device:torch.device,
                                loss_fct:Union[torch.nn.Module, AUCMLoss],tokenizer:PreTrainedTokenizer) -> PeftModel:  
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

def bootstrap_convolution_train_function(output_dir:str,logging_dir:str,training_configuration:dict,dataset:Dataset,
                                optimizer:torch.optim.Optimizer,model:CNNModel,device:torch.device,
                                loss_fct:Union[torch.nn.Module, AUCMLoss],tokenizer:PreTrainedTokenizer) -> CNNModel:
    
    writer = SummaryWriter(log_dir=logging_dir)
    
    X_train =CNNModel.process_sequences_convolution(sequences=dataset.train_set,device=device) 
    y_train = torch.tensor(dataset.train_labels, dtype=torch.float32).to(device)
    
    X_val = CNNModel.process_sequences_convolution(sequences=dataset.validation_set,device=device)
    y_val = torch.tensor(dataset.validation_labels, dtype=torch.float32).to(device)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=training_configuration['compile']['training_arguments']['kwargs']['batch_size'],shuffle = True)
    train_dataset = TensorDataset(X_train, y_train)
    if type(optimizer) == PESG:
        train_sampler = DualSampler(train_dataset,
                            batch_size=training_configuration['compile']['training_arguments']['kwargs']['batch_size'],
                            sampling_rate=0.5,
                            labels=y_train.cpu().numpy())
        train_loader = DataLoader(train_dataset,
                                  batch_size=training_configuration['compile']['training_arguments']['kwargs']['batch_size'],
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=training_configuration['compile']['training_arguments']['kwargs']['batch_size'], shuffle=True)  
    
    def roc_score_function(engine):
        print('roc_score_function')
        y_true, y_pred = engine.state.output
        val_roc_auc = calculate_roc_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())
        engine.state.score = val_roc_auc
        return val_roc_auc
                
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        X_batch, y_batch = batch
        outputs = model(X_batch).squeeze(dim=1)
        loss = loss_fct(outputs, y_batch)
        loss.backward()
        optimizer.step()
        engine.state.loss += loss.item()
        return loss.item()

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, y_batch = batch
            y_pred = model(X_batch).squeeze(dim=1)
            return y_batch, y_pred

    def accumulate_outputs(engine, batch):
        y_true, y_pred = batch
        if not hasattr(engine.state, 'all_y_true'):
            engine.state.all_y_true = []
            engine.state.all_y_pred = []
        engine.state.all_y_true.append(y_true)
        engine.state.all_y_pred.append(y_pred)

    def reset_accumulated_outputs(engine):
        print('reset_accumulated_outputs')
        engine.state.all_y_true = []
        engine.state.all_y_pred = []
        engine.state.score = 0

    def reset_best_score(engine):
        predictions = CNNModel.predict_with_convolution(model,X_val)
        engine.state.best_score = calculate_roc_auc(y_val.cpu().numpy(), predictions)
        print(f"Initial best score: {engine.state.best_score:.3f}")
        engine.state.best_model = copy.deepcopy(model.state_dict())


    def reset_loss(engine):
        engine.state.loss = 0

    def get_accumulated_outputs(engine):
        y_true = torch.cat(engine.state.all_y_true, dim=0)
        y_pred = torch.cat(engine.state.all_y_pred, dim=0)
        return y_true, y_pred

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.STARTED, reset_best_score)
    trainer.add_event_handler(Events.EPOCH_STARTED, reset_loss)
    evaluator = Engine(validation_step)
    evaluator.add_event_handler(Events.EPOCH_STARTED, reset_accumulated_outputs)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: accumulate_outputs(engine, engine.state.output))
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: setattr(engine.state, 'output', get_accumulated_outputs(engine)))

    @trainer.on(Events.EPOCH_COMPLETED)
    def calculate_metric_engine(engine):
        model.eval()
        print(f' loss: {engine.state.output:.3f}')
        evaluator.run(val_loader)
        score = evaluator.state.score
        print(f"Epoch {engine.state.epoch} - metric_score: {score:.3f}")
        if score > engine.state.best_score + 0.001:
            engine.state.best_score = score
            engine.state.best_model = copy.deepcopy(model.state_dict())
            print(f" epoch {engine.state.epoch} - load best model")
        writer.add_scalar('Loss/train', engine.state.loss, engine.state.epoch)
        writer.add_scalar('Metric/validation', score, engine.state.epoch)
    
   
    handler = EarlyStopping(patience=5,score_function = roc_score_function , trainer = trainer)
    
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer.run(train_loader, max_epochs=training_configuration['compile']['training_arguments']['kwargs']['epochs'])
                        
    model.load_state_dict(trainer.state.best_model)
    y_pred = CNNModel.predict_with_convolution(model,X_val)
    y_true = y_val.cpu().numpy()
    
    val_metric = calculate_roc_auc(y_true, y_pred)
    print(f"Validation metric: {val_metric:.3f}")
    writer.close()
    return model
    
train_function_to_bootstrapper = {
    'LM': bootstrap_LM_train_function,
    'convolution': bootstrap_convolution_train_function,
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