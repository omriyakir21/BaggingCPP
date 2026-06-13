import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import pandas as pd
from models.inference_LM import load_esm2_with_LA_lora_model,load_tokenizer,predict_binary_probs_from_sequences
from models.inference_convolution import predict_binary_probs_from_sequences_convolution,load_convolution_model
from typing import Dict, List, Tuple, Union
import torch

def calculate_roc_auc(y_true, y_score):
    """
    Calculate the ROC AUC score.
    Args:
        y_true (list): True binary labels.
        y_score (list): Target scores, can either be probability estimates of the positive class or binary decisions.
    Returns:
        float: ROC AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def calculate_precision_recall_auc(y_true, y_score):
    """
    Calculate the Precision-Recall AUC score.
    Args:
        y_true (list): True binary labels.
        y_score (list): Target scores, can either be probability estimates of the positive class or binary decisions.
    Returns:
        float: Precision-Recall AUC score.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def get_experiment_hyperparams(experiment_name:str)-> dict:
    """
    Extracts hyperparameters from the experiment name.
    The name should be formatted as 'exp_name_param1_value1_param2_value2'.
    """
    print(f"Extracting hyperparameters from experiment name: {experiment_name}")
    parts = experiment_name.split('_')
    hyperparams = {}
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            hyperparams[parts[i]] = parts[i + 1]
    print(f"Extracted hyperparameters: {hyperparams}")
    return hyperparams


def create_pandas_grid_search_summary(multi_experiments_data: dict):
    """
    Create a summary DataFrame from multiple experiments data.
    Each experiment's data should be a tuple of (y_true, y_score).
    """

    # Initialize an empty list to store the results
    from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
    results = []

    for exp_name, exp_data in multi_experiments_data.items():
        if not exp_data:
            print(f"No data for experiment '{exp_name}', skipping.")
            continue

        try:
            y_true, y_score = exp_data
        except Exception:
            raise ValueError(
                f"experiment '{exp_name}' data must be (y_true, y_score), got {type(exp_data)}"
            )

        # Extract hyperparameters from the experiment name
        hyperparams = get_experiment_hyperparams(exp_name)
        hyperparams['AUC'] = calculate_roc_auc(y_true, y_score)
        hyperparams['Precision-Recall AUC'] = calculate_precision_recall_auc(y_true, y_score)
        
        # Threshold scores to compute binary predictions
        pred_labels = [1 if score >= 0.5 else 0 for score in y_score]
        hyperparams['F1'] = f1_score(y_true, pred_labels)
        hyperparams['Accuracy'] = accuracy_score(y_true, pred_labels)
        hyperparams['MCC'] = matthews_corrcoef(y_true, pred_labels)

        results.append(hyperparams)

        # Sort results by AUC in reverse order
        results.sort(key=lambda x: x['AUC'], reverse=True)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns.")
    return df

def multi_exp_data_to_fig(multi_experiments_data: dict, 
                          show_std_in_legend: bool = True, log_scale: bool = True, 
                          resolution: int = 1_000) -> go.Figure:

    # Simulation settings
    num_experiments = len(multi_experiments_data)  # Change this number dynamically

    # Define fixed FPR values for interpolation (log scale)
    if log_scale:
        fpr_interp_values = np.logspace(-3, 0, resolution)  # Avoiding zero
    else:
        fpr_interp_values = np.linspace(0, 1, resolution)

    # Dynamically choose colors from Plotly's palette
    color_palette = px.colors.qualitative.Set1  # Can also use Set2, Plotly, Dark24, etc.
    colors = color_palette * (num_experiments // len(color_palette) + 1)  # Repeat if needed

    fig = go.Figure()

    print('computing rocs')
    exp_idx = 0
    exp_to_auc = dict()
    for exp_name, exp_data in tqdm(multi_experiments_data.items()):
        if not exp_data:
            print(f"No data for experiment '{exp_name}', skipping.")
            continue

        try:
            y_true, y_score = exp_data
        except Exception:
            raise ValueError(
                f"experiment '{exp_name}' data must be (y_true, y_score), got {type(exp_data)}"
            )
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        exp_to_auc[exp_name] = auc_val
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color=colors[exp_idx], width=3),
            name=f'{exp_name} (AUC = {auc_val:.4f})'
        ))
        exp_idx += 1
    # Adapted random guess line (log space)
    print('adding random roc')
    if log_scale:
        random_guess_fpr = np.logspace(-3, 0, resolution)  # Avoiding zero
    else:
        random_guess_fpr = np.linspace(0, 1, resolution)
    random_guess_tpr = random_guess_fpr  # y = x, but now correctly sampled

    fig.add_trace(go.Scatter(x=random_guess_fpr, y=random_guess_tpr, mode='lines',
                            line=dict(dash='dash', color='black'), name='Random Guess'))

    # Layout
    print('fig update')
    if log_scale:
        xaxis_dict = dict(title='False Positive Rate (Log Scale)', type='log')
    else:
        xaxis_dict = dict(title='False Positive Rate')
    fig.update_layout(
        title='ROC Curves Across Multiple Experiments',
        xaxis=xaxis_dict,
        yaxis=dict(title='True Positive Rate'),
        template='plotly_white'
    )

    if not show_std_in_legend:
        for trace in fig['data']: 
            if ('Std Dev' in trace['name']):
                trace['showlegend'] = False

    return fig, exp_to_auc


def get_fold_indexes(fold_training_dicts:list,sequences:list):
    """
    Get the indexes of the sequences in each fold.
    Args:
        fold_training_dicts (list): List of dictionaries containing fold training data.
        sequences (list): List of sequences to be indexed.
    Returns:
        dict: Dictionary with fold numbers as keys and lists of indexes as values.
    """
    folds_test_sets = [set(fold['sequences_test']) for fold in fold_training_dicts]
    fold_num_list = [-1 for _ in range(len(sequences))]
    for i in range(len(sequences)):
        for j in range(len(folds_test_sets)):
            if sequences[i] in folds_test_sets[j]:
                fold_num_list[i] = j
                break

    indexes_dict = {}
    for num in range(-1, 5):
        indexes_dict[num] = []
        for i in range(len(sequences)):
            if fold_num_list[i] == num:
                indexes_dict[num].append(i)
    return indexes_dict


def get_experiment_base_paths_for_ensemble(experiment:str,base_paths:dict,hypothesis_path:str,num_submodels:int):
    submodels_to_run = []
    experiment_base_paths = []
    for index in range(num_submodels):
        experiments_submodel_runs = []
        submodel_path = os.path.join(hypothesis_path, experiment, f'submodel_{index}')
        if not os.path.exists(submodel_path):
            submodels_to_run.append(f'{experiment}_index_{index}')
            continue
        try:
            experiments_submodel_runs = os.listdir(submodel_path)
        except Exception as e:
            print(e)
            continue
        try:
            experiments_submodel_runs = sorted(
            experiments_submodel_runs, 
            key=lambda x: os.path.getmtime(os.path.join(submodel_path, x)), 
            reverse=True)
        except Exception as e:
            print(e)
        if len(experiments_submodel_runs) == 0:
            print(f'submodel_path: {submodel_path} does not have any runs')
            submodels_to_run.append(f'{experiment}_index_{index}')
            continue
        experiment_submodels_valid_run_found = False
        for updated_experiment_run in experiments_submodel_runs:
            run_name = f'{submodel_path}/{updated_experiment_run}'
            if 'fold_4' in os.listdir(run_name):
                experiment_base_paths.append(run_name)
                experiment_submodels_valid_run_found = True
                break
        if not experiment_submodels_valid_run_found:
            print(f':experiment {experiment},submodel {index} didnt finish, skipping')
            submodels_to_run.append(f'{experiment}_index_{index}')
    base_paths[experiment] = experiment_base_paths
    return base_paths, submodels_to_run

def list_runs(hypo_path, experiment):
    try:
        runs = os.listdir(os.path.join(hypo_path, experiment))
        runs = [os.path.join(hypo_path, experiment, run) for run in runs if os.path.isdir(os.path.join(hypo_path, experiment, run))]
        return runs
    except OSError as e:
        print(f"Error listing runs for {experiment}: {e}")
        return []

def sort_runs_by_date(runs):
    try:
        return sorted(
            runs,
            key=lambda x: os.path.getmtime( x), 
            reverse=True)
    except Exception as e:
        print(f"Error sorting runs: {e}")
        return runs

def find_latest_valid_run(hypo_path, experiment, runs, required_fold='fold_4'):
    for run in runs:
        if required_fold in os.listdir(run):
            return run
    return None

def collect_base_path(hypothesis_path, experiment, required_fold='fold_4'):
    runs = list_runs(hypothesis_path, experiment)
    if not runs:
        raise ValueError(f"No runs found for experiment {experiment} in {hypothesis_path}")
    runs = sort_runs_by_date(runs)
    base_path = find_latest_valid_run(hypothesis_path, experiment, runs, required_fold)
    if not base_path:
        raise ValueError(f"No valid runs found for experiment {experiment} in {hypothesis_path}")
    return base_path

def get_experiment_base_paths(experiment:str,base_paths:dict,hypothesis_path:str,ensemble:bool,num_submodels:int):
    """
    Collects the base paths for the given experiment.
    If ensemble is True, it collects paths for all submodels.
    """
    if ensemble:
        return get_experiment_base_paths_for_ensemble(experiment, base_paths, hypothesis_path,num_submodels)
    else:
        base_path = collect_base_path(hypothesis_path, experiment)
        return {experiment: base_path}

def predict_embeddings_for_LA_single_experiment(experiment: str,
                                                base_paths: dict,
                                                model_name: str,
                                                device: str,
                                                num_labels: int,
                                                dout: int,
                                                kernel_size: int,
                                                use_max: bool,
                                                sequences: list,
                                                labels:int,
                                                indexes_dict: dict):
    """
    For a single LA experiment, predict probabilities and extract embeddings.
    Returns a dict whose keys are fold numbers 1 to 5, and values are lists of tuples:
      (sequence, prediction, embeddings)
    Non-specific examples (indexed by -1) are ignored.
    """
    if experiment not in base_paths:
        raise ValueError(f'Experiment {experiment} not found in base paths')
    if len(base_paths[experiment]) == 0:
        raise ValueError(f'No base paths found for experiment {experiment}')
    tokenizer = load_tokenizer(model_name=model_name)
    folds_results = []

    # Process folds 0 through 4 (return keys 1 to 5)
    for fold in range(5):
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        specific_fold_labels = [labels[i] for i in indexes_dict[fold]]
        try:
            model_path = os.path.join(base_paths[experiment], f'fold_{fold}', 'model')
            model = load_esm2_with_LA_lora_model(model_path=model_path,
                                                 device=device,
                                                 model_name=model_name,
                                                 num_labels=num_labels,
                                                 dout=dout,
                                                 kernel_size=kernel_size,
                                                 use_max=use_max)


            predictions, embeddings = predict_binary_probs_from_sequences(model=model,
                                                                        sequences=specific_fold_sequences,
                                                                        tokenizer=tokenizer,
                                                                        device=device,
                                                                        batch_size=512,
                                                                        return_embeddings=True)

            folds_results.append((specific_fold_sequences,specific_fold_labels, predictions, embeddings))
        except Exception as e:
            print(e)
            print(f'Error processing fold {fold} for experiment {experiment}')
            raise(e)
    return folds_results

def predict_for_LA_single_experiment(experiment:str,base_paths:dict,
                                                        model_name:str,device:str,
                                                        num_labels:int,dout:int,kernel_size:int,
                                                        use_max:bool,sequences:list,
                                                        labels:list,indexes_dict:dict):
    if experiment not in base_paths:
        raise ValueError(f'Experiment {experiment} not found in base paths')
    if len(base_paths[experiment]) == 0:
        raise ValueError(f'No base paths found for experiment {experiment}')
    tokenizer = load_tokenizer(model_name=model_name)
    predictions_dict = {}
    non_specific_predictions = []
    non_specific_sequences = [sequences[i] for i in indexes_dict[-1]]
    non_specific_labels = [labels[i] for i in indexes_dict[-1]]
    for fold in range(5):
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        specific_fold_labels = [labels[i] for i in indexes_dict[fold]]
        try:
            model_path = os.path.join(base_paths[experiment], f'fold_{fold}','model')
            model = load_esm2_with_LA_lora_model(model_path=model_path,device=device,model_name=model_name,
                                                num_labels=num_labels,dout=dout,kernel_size=kernel_size,use_max=use_max)
            if len(specific_fold_sequences) > 0:
                fold_specific_predictions = predict_binary_probs_from_sequences(model=model,device=device,
                                                                        sequences=specific_fold_sequences,
                                                                        tokenizer=tokenizer,batch_size=512)
            else: 
                fold_specific_predictions = np.zeros((len(specific_fold_sequences), 1))
            
            if len(non_specific_sequences) > 0:
                fold_non_specific_predictions = predict_binary_probs_from_sequences(model=model,device=device,
                                                                            sequences=non_specific_sequences,
                                                                            tokenizer=tokenizer,batch_size=512)
            else:
                fold_non_specific_predictions=np.zeros((len(non_specific_sequences), 1))
        except Exception as e:
            print(e)
            print(f'Error loading fold {fold} for experiment {experiment}')
            continue
        predictions_dict[fold] = {'prediction':fold_specific_predictions,'sequence':specific_fold_sequences,'label':specific_fold_labels}
        non_specific_predictions.append(fold_non_specific_predictions)
    non_specific_predictions = np.mean(non_specific_predictions, axis=0)
    predictions_dict[-1] = {'prediction':non_specific_predictions,'sequence':non_specific_sequences,'label':non_specific_labels}
    predictions_df = pd.DataFrame()
    for fold in predictions_dict.keys():
        if predictions_dict[fold]['prediction'].shape[0] == 0:
            print(f'fold {fold} has no predictions')
            continue
        fold_df = pd.DataFrame(predictions_dict[fold])
        fold_df['fold'] = fold
        predictions_df = pd.concat([predictions_df, fold_df], axis=0)
    return predictions_df


def predict_LoRA_LA_ensemble_experiment_using_base_paths(experiment:str,base_paths:dict,
                                                        model_name:str,device:str,
                                                        num_labels:int,dout:int,kernel_size:int,
                                                        use_max:bool,sequences:list,
                                                        labels:list,indexes_dict:dict,num_submodels:int):
    """
    Predicts the probabilities of the sequences using the LoRA LA model and the base paths provided.
    """
    if experiment not in base_paths:
        raise ValueError(f'Experiment {experiment} not found in base paths')
    if len(base_paths[experiment]) == 0:
        raise ValueError(f'No base paths found for experiment {experiment}')
    if len(base_paths[experiment]) < num_submodels:
        print(f'didnt finish {experiment}, skipping')
        raise Exception(f'didnt finish {experiment}, skipping')
    tokenizer = load_tokenizer(model_name=model_name)
    predictions_dict = {}
    non_specific_predictions = []
    non_specific_sequences = [sequences[i] for i in indexes_dict[-1]]
    non_specific_labels = [labels[i] for i in indexes_dict[-1]]
    for fold in range(5):
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        specific_fold_labels = [labels[i] for i in indexes_dict[fold]]
        fold_specific_predictions = []
        fold_non_specific_predictions = []
        for index in range(num_submodels):
            try:
                model_path = os.path.join(base_paths[experiment][index], f'fold_{fold}','model')
                model = load_esm2_with_LA_lora_model(model_path=model_path,device=device,model_name=model_name,
                                                    num_labels=num_labels,dout=dout,kernel_size=kernel_size,use_max=use_max)
                if len(specific_fold_sequences) > 0:
                    fold_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                            sequences=specific_fold_sequences,
                                                                            tokenizer=tokenizer,batch_size=512))
                else: 
                    fold_specific_predictions.append(np.zeros((len(specific_fold_sequences), 1)))
                
                if len(non_specific_sequences) > 0:
                    fold_non_specific_predictions.append(predict_binary_probs_from_sequences(model=model,device=device,
                                                                                sequences=non_specific_sequences,
                                                                                tokenizer=tokenizer,batch_size=512))
                else:
                    fold_non_specific_predictions.append(np.zeros((len(non_specific_sequences), 1)))

            except Exception as e:
                print(e)
                print(f'Error loading fold {fold} for experiment {experiment}, submodel {index}')
                continue
        fold_specific_predictions = np.mean(fold_specific_predictions, axis=0)
        predictions_dict[fold] = {'prediction':fold_specific_predictions,'sequence':specific_fold_sequences,'label':specific_fold_labels}
        fold_non_specific_predictions = np.mean(fold_non_specific_predictions, axis=0)
        non_specific_predictions.append(fold_non_specific_predictions)
    non_specific_predictions = np.mean(non_specific_predictions, axis=0)
    predictions_dict[-1] = {'prediction':non_specific_predictions,'sequence':non_specific_sequences,'label':non_specific_labels}
    predictions_df = pd.DataFrame()
    for fold in predictions_dict.keys():
        if predictions_dict[fold]['prediction'].shape[0] == 0:
            print(f'fold {fold} has no predictions')
            continue
        fold_df = pd.DataFrame(predictions_dict[fold])
        fold_df['fold'] = fold
        predictions_df = pd.concat([predictions_df, fold_df], axis=0)
    return predictions_df

def predict_for_LoRA_LA_experiment(experiment:str,base_paths:dict,
                                                        model_name:str,device:str,
                                                        num_labels:int,dout:int,kernel_size:int,
                                                        use_max:bool,sequences:list,
                                                        labels:list,indexes_dict:dict,ensemble:bool,num_submodels:int):
    if ensemble:
        return predict_LoRA_LA_ensemble_experiment_using_base_paths(experiment=experiment, base_paths=base_paths,
                                                                    model_name=model_name, device=device,
                                                                    num_labels=num_labels, dout=dout,
                                                                    kernel_size=kernel_size, use_max=use_max,
                                                                    sequences=sequences, labels=labels,
                                                                    indexes_dict=indexes_dict,num_submodels = num_submodels)
    else:
        return predict_for_LA_single_experiment(experiment=experiment, base_paths=base_paths,
                                                model_name=model_name, device=device,
                                                num_labels=num_labels, dout=dout, kernel_size=kernel_size,
                                                use_max=use_max, sequences=sequences, labels=labels,
                                                indexes_dict=indexes_dict)

def predict_convolution_ensemble_experiment_using_base_paths(experiment:str,base_paths:dict,
                                        device:str,filters:int,dialation:int,kernel_size:int,
                                        num_layers:int,padding:str,sequences:list,
                                        labels:list,indexes_dict:dict):
    """
    Predicts the probabilities of the sequences using the convolution model and the base paths provided.
    """
    """
    Predicts the probabilities of the sequences using the LoRA LA model and the base paths provided.
    """
    num_submodels = int(experiment.split('it_')[1].split('_')[0])
    if experiment not in base_paths:
        raise ValueError(f'Experiment {experiment} not found in base paths')
    if len(base_paths[experiment]) == 0:
        raise ValueError(f'No base paths found for experiment {experiment}')

    if len(base_paths[experiment]) < num_submodels:
        print(f'didnt finish {experiment}, skipping')
        raise Exception(f'didnt finish {experiment}, skipping')
    predictions_dict = {}
    non_specific_predictions = []
    non_specific_sequences = [sequences[i] for i in indexes_dict[-1]]
    non_specific_labels = [labels[i] for i in indexes_dict[-1]]
    for fold in range(5):
        specific_fold_sequences = [sequences[i] for i in indexes_dict[fold]]
        specific_fold_labels = [labels[i] for i in indexes_dict[fold]]
        fold_specific_predictions = []
        fold_non_specific_predictions = []
        for index in range(num_submodels):
            try:
                model_path = os.path.join(base_paths[experiment][index], f'fold_{fold}','model','model.pt')
                model = load_convolution_model(model_path=model_path,filters=filters,device=device,
                                               dilation=dialation,kernel_size=kernel_size,
                                               num_layers=num_layers,padding=padding)
                if len(specific_fold_sequences) > 0:
                    fold_specific_predictions.append(predict_binary_probs_from_sequences_convolution(model=model,device=device,
                                                                            sequences=specific_fold_sequences))
                else:
                    fold_specific_predictions.append(np.zeros((len(specific_fold_sequences), 1)))
                if len(non_specific_sequences) > 0:                            
                    fold_non_specific_predictions.append(predict_binary_probs_from_sequences_convolution(model=model,device=device,
                                                                            sequences=non_specific_sequences))
                else:
                    fold_non_specific_predictions.append(np.zeros((len(non_specific_sequences), 1)))
                                                                            
            except Exception as e:
                print(e)
                print(f'Error loading fold {fold} for experiment {experiment}, submodel {index}')
                continue
        fold_specific_predictions = np.mean(fold_specific_predictions, axis=0)
        predictions_dict[fold] = {'prediction':fold_specific_predictions,'sequence':specific_fold_sequences,'label':specific_fold_labels}
        fold_non_specific_predictions = np.mean(fold_non_specific_predictions, axis=0)
        non_specific_predictions.append(fold_non_specific_predictions)
    non_specific_predictions = np.mean(non_specific_predictions, axis=0)
    predictions_dict[-1] = {'prediction':non_specific_predictions,'sequence':non_specific_sequences,'label':non_specific_labels}
    predictions_df = pd.DataFrame()
    for fold in predictions_dict.keys():
        if predictions_dict[fold]['prediction'].shape[0] == 0:
            print(f'fold {fold} has no predictions')
            continue
        fold_df = pd.DataFrame(predictions_dict[fold])
        fold_df['fold'] = fold
        predictions_df = pd.concat([predictions_df, fold_df], axis=0)
    return predictions_df


def gather_ensemble_experiments_base_paths(hypothesis, experiments, it=50, results_root='../results/hypothesis', required_fold='fold_4'):
    """
    Scan results for each experiment/submodel and return:
      - base_paths: dict[experiment] = list of latest valid run paths (one per submodel)
      - submodels_to_run: list of experiment_index strings that are missing or incomplete
    """
    base_paths = {}
    submodels_to_run = []
    hypothesis_path = os.path.join(results_root, hypothesis)
    for experiment in experiments:
        experiment_base_paths = []
        num_submodels = int(experiment.split('it_')[1].split('_')[0]) if 'it_' in experiment else it
        print(num_submodels)
        for index in range(num_submodels):
            submodel_path = os.path.join(hypothesis_path, experiment, f'submodel_{index}')
            if not os.path.exists(submodel_path):
                submodels_to_run.append(f'{experiment}_index_{index}')
                continue

            try:
                runs = os.listdir(submodel_path)
            except Exception as e:
                print(e)
                submodels_to_run.append(f'{experiment}_index_{index}')
                continue

            # sort runs by modification time (newest first)
            try:
                runs = sorted(
                    runs,
                    key=lambda x: os.path.getmtime(os.path.join(submodel_path, x)),
                    reverse=True
                )
            except Exception as e:
                print(e)

            if not runs:
                print(f'submodel_path: {submodel_path} does not have any runs')
                submodels_to_run.append(f'{experiment}_index_{index}')
                continue

            valid_found = False
            for run_name in runs:
                full_run = os.path.join(submodel_path, run_name)
                if not os.path.isdir(full_run):
                    continue
                try:
                    if required_fold in os.listdir(full_run):
                        experiment_base_paths.append(full_run)
                        valid_found = True
                        break
                except Exception as e:
                    print(e)
                    continue

            if not valid_found:
                print(f':experiment {experiment},submodel {index} didnt finish, skipping')
                submodels_to_run.append(f'{experiment}_index_{index}')

        base_paths[experiment] = experiment_base_paths

    print(f'base_paths: {base_paths}')
    print(f'submodels_to_run: {submodels_to_run}')
    return base_paths, submodels_to_run

