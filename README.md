<h1 align="center">BaggingCPP</h1>
<h3 align="center">A Deep Positive-Unlabeled Learning Approach to Cell Penetrating Peptide Discovery</h3>

![](figure.png)

Cell-penetrating peptides (CPPs) are a promising approach for the intracellular delivery of diverse molecular cargos. However, although hundreds of CPPs have been characterized, most are cationic peptides with poor pharmacoproperties or limited uptake efficiency; new high-throughput discovery approaches are thus imperative. Here, we introduce BaggingCPP, a deep learning-based CPP virtual screening framework that integrates protein language models, Positive-Unlabeled (PU) learning and parameter-efficient fine-tuning algorithms. Unlike prior works, we do not use an artificial negative set on which to train the model. Instead, we use PU learning to directly train and infer on the candidate library  - a large collection of naturally expressed peptides such as hormones, neuropeptides, and small proteins. We show that BaggingCPP is competitive with the state-of-the-art model GraphCPP when training and evaluating on the standard public CPP1708 benchmark. More importantly, we demonstrate that the PU learning formulation effectively addresses the distribution shift and prediction stability problems commonly encountered in the standard train-then-screen protocol, thereby significantly reducing the false positive rate. Using BaggingCPP, we identified and experimentally validated several CPPs with low similarity to known CPPs, including two with higher uptake efficiency than the gold-standard TAT peptide. The latter are cyclic peptides that may penetrate via GPCR-mediated endocytosis - a relatively rare mechanism of entry. BaggingCPP thus represents a data-driven approach to expand the chemical diversity of CPPs.

# Installation

## Python Environment Setup
A conda distribution is required. To create the conda and python environment, run: <br>
<code>bash scripts/env/create_conda_env.sh </code>

To activate the environment, run: <br> <code>activate BaggingCPP </code>

## Downloading model weights from HuggingFace

In order to download the models ensemble run:
Jerome the huggingface model and data is still private,so use:<br>
<code>/home/iscb/wolfson/omriyakir/BaggingCPP/huggingface/download_test.py</code> for testing, I'll make everything public when we will publish. <br>


# Model Inference

Run inference using the model from huggingface:<br>
<code>python -m inference.inference \
    --sequences_fasta example/example.fasta  \
    --output_csv example/example_output.csv \
    --folds_training_dicts_path huggingface_repo/data_folder/folds_training_dicts.pkl \
    --huggingface_model_folder_path huggingface_repo/model_folder/ensemble \
    --use_huggingface_repo \
</code>

# Data preparation and Training from scratch

This is the code for retraining from scratch the original model. It can be easily adapted to any other PU learning classification task over peptides.

The raw data should be provided as a .csv file (see: datasets/full_datasets/bagging_cpp_dataset.csv), with the following fields: 

id,sequence,source,description,label,cluster_id,fold_id


## Data preparation



<ol start="1">
<li>
Divide the dataset into 5 folds: <br>
<code>python -m python -m data_preparation.cross_validation</code></li>
<li>
Sample and save unlabeled sequences for the different submodels: <br>
For the inductive setting: <br>
<code>python -m python -m data_preparation.cross_validation --algorithm_name --algorithm_name inductive_pu_learning</code><br>
For the transductive setting:<br>
<code>python -m python -m data_preparation.cross_validation --algorithm_name --algorithm_name transductive_pu_learning</code><br></li>
</ol>

## Run training
<ol start="1">
<li>
For each of the configurations under <code>configurations/data/{ensemble_inductive_pu_learning or ensemble_transductive_pu_learning}</code>, run: <br>
<code> python -m models.train --config path_to_configuration
</code><br>
e.g. <code> python -m models.train --config configurations/data/ensemble_inductive_pu_learning/groups_inductive_index_0.json
</code><br></li>
</ol>

## Generate prediction tables in the transductive setting:
After training the ensemble, run:<br>
<code> evaluation/aggregate_transductive_payload.py </code>.
The prediction table is saved in the results folder of the ensemble_transductive_pu_learning hypothesis, in:<br>
<code>results/hypothesis/ensemble_transductive_pu_learning/groups_transductive/aggregated_payload/predictions.csv</code>


## Evaluation
For AUC-ROC calculation(only valid for the inductive setting),run the notebook:<br>
 <code> evaluation/evaluation_ensemble.ipynb</code>:

## Run inference over the inductive pu learning trained ensemble:
In order to run inference using the trained model do: <br>
<code>python -m inference.inference \
    --sequences_fasta {path to fasta with sequences} \
    --output_csv {path to output csv} \
    --hypothesis ensemble_inductive_pu_learning \
    --experiment groups_inductive \
    --folds_training_dicts_path datasets/data_for_training/folds_training_dicts.pkl
</code>
