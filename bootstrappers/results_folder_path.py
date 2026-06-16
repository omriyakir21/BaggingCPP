import os
from typing import Union
from datetime import datetime
import uuid



def create_folder_path(hypothesis:str ,experiment_name:str) -> str:
    random_id = str(datetime.now()).split(' ')[0] + '_' + uuid.uuid4().hex[:10]
    folder_path = os.path.join(
        'results', 'hypothesis', hypothesis, experiment_name, random_id)
    return folder_path

def bootstrap_ensemble_submodel_results_folder_path(hypothesis:str ,experiment_name:str) -> str:
    submodel_index = experiment_name.split('_index')[-1]
    experiment_name = experiment_name.split('_index')[0]
    ensemble_folder_path = create_folder_path(hypothesis, experiment_name)
    end_path = ensemble_folder_path.split('/')[-1]
    submodel_folder_path = os.path.join('/'.join(ensemble_folder_path.split('/')[:-1]), f'submodel{submodel_index}',end_path)
    os.makedirs(submodel_folder_path, exist_ok=True)
    return submodel_folder_path

def bootstrap_model_results_path(hypothesis:str ,experiment_name:str) -> str:
    single_model_folder_path = create_folder_path(hypothesis, experiment_name)
    os.makedirs(single_model_folder_path, exist_ok=True)
    return single_model_folder_path

results_folder_path_to_bootstrapper = {
    'ensemble_submodel': bootstrap_ensemble_submodel_results_folder_path,
    'single_model': bootstrap_model_results_path,
}

def build_results_folder_path_from_configuration(hypothesis: str, experiment: dict) -> str:
    supported_losses = list(results_folder_path_to_bootstrapper.keys())
    print(f"experiment: {experiment}")
    if experiment['experiment_type'] not in results_folder_path_to_bootstrapper.keys():
        raise Exception(f'dataset: {experiment["experiment_type"]} not supported. supported losses: {supported_losses}')
    return results_folder_path_to_bootstrapper[experiment['experiment_type']](
        hypothesis=hypothesis, experiment_name = experiment['name'],**experiment['kwargs'])