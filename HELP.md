## What BaggingCPP does

BaggingCPP scores peptides for their likelihood of being **cell-penetrating peptides (CPPs)**. It is an ensemble
of 250 protein-language-model classifiers trained with Positive-Unlabeled (PU) learning: known CPPs are the
positives, and a large library of naturally expressed peptides is the unlabeled set.

## Input

Upload a FASTA file or paste FASTA text.

- Up to **128 sequences** per job.
- Each sequence: uppercase amino-acid letters only (`A`–`Z`), at most **48 residues**.
- Each header (the text after `>`) must be unique, contain no spaces, be at most 25 characters long, and use only
  letters, digits and `- _ . : * #`.

Example:

```
>TAT
GRKKRRQRRRPPQ
>Penetratin
RQIKIWFQNRRMKWKK
```

## Results

- **Results table**: the 50 highest-scoring sequences, best first.
- **Download Results**: a CSV with every sequence you submitted:

| Column | Meaning |
|---|---|
| `sequence` | The peptide sequence |
| `label` | Its FASTA header |
| `prediction` | Predicted probability of being a CPP (0–1) |
| `model_uncertainty` | Standard deviation of the prediction across the ensemble members |

A sequence that appears in BaggingCPP's own training set is scored only by the ensemble that did not see it
during training (it was used as an *unlabeled* example by the others, which would bias its score down). Every
other sequence is scored by all 5 ensembles (250 models), and the scores are averaged.

## Good to know

- A job usually takes a few minutes; it can take longer when other jobs are queued. Keep the page open until
  the results appear.
- The download link is valid for 24 hours. Results are kept on the server for a limited time only, so save the
  CSV.
- If the server is busy, you will be asked to try again a few minutes later.

## Citation

If you use BaggingCPP, please cite: *BaggingCPP: An Inductive PU-Learning Framework for Discovering
Cell-Penetrating Peptides* ([OpenReview](https://openreview.net/forum?id=teu3GAjqLi)).
