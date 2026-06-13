# BaggingCPP

## Enviroment Setup
To create the conda env run: <br>
<code>bash scripts/env/create_conda_env.sh </code>

## Inferece
In order to download the models ensemble run:
Jerome the huggingface model and data is still private,so use:<br>
<code>/home/iscb/wolfson/omriyakir/BaggingCPP/huggingface/download_test.py</code> for testing, I'll make everything public when we will publish.
<br>
Run inference using the model from huggingface:<br>
<code>python -m inference.inference \
    --sequences_fasta {path to fasta with sequences}  \
    --output_csv {path to output csv} \
    --folds_training_dicts_path huggingface_repo/data_folder/folds_training_dicts.pkl \
    --huggingface_model_folder_path huggingface_repo/model_folder/ensemble \
    --use_huggingface_repo \
</code>


## Data preparation and Training from scratch
### data preparation
<ol start="1">
<li>
Divide the dataset into 5 folds: <br>
<code>python -m python -m data_preparation.cross_validation</code></li>
<li>
Sample and save unlabled sequences for the different submodels: <br>
For the inductive setting: <br>
<code>python -m python -m data_preparation.cross_validation --algorithm_name --algorithm_name inductive_pu_learning</code><br>
For the transductive setting:<br>
<code>python -m python -m data_preparation.cross_validation --algorithm_name --algorithm_name transductive_pu_learning</code><br></li>
</ol>

### Run training
<ol start="1">
<li>
For each of the configurations under <code>configurations/data/{ensemble_inductive_pu_learning or ensemble_transductive_pu_learning}</code>, run: <br>
<code> python -m models.train --config path_to_configuration
</code><br>
e.g. <code> python -m models.train --config configurations/data/ensemble_inductive_pu_learning/groups_inductive_index_0.json
</code><br></li>
</ol>

### Create predictions table in the transductive setting:
After training the ensemble, run:<br>
<code> evaluation/aggregate_transductive_payload.py </code>.
The prediction table is saved in the results folder of the ensemble_transductive_pu_learning hypothesis, in:<br>
<code>results/hypothesis/ensemble_transductive_pu_learning/groups_transductive/aggregated_payload/predictions.csv</code>


### Evaluation
For AUC-ROC calculation(only valid for the inductive setting),run the notebook:<br>
 <code> evaluation/evaluation_ensemble.ipynb</code>:

### Run inference over the inductive pu learning trained ensemble:
In order to run inference using the trained model do: <br>
<code>python -m inference.inference \
    --sequences_fasta {path to fasta with sequences} \
    --output_csv {path to output csv} \
    --hypothesis ensemble_inductive_pu_learning \
    --experiment groups_inductive \
    --folds_training_dicts_path datasets/data_for_training/folds_training_dicts.pkl

</code>
