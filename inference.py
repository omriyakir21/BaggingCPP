import json
import os
from typing import Dict, Union

import pandas as pd
from tubiana_lab_utils.data import inputs, outputs
from inference_api.inference import load_ensemble, predict as run_ensemble_prediction

# Loading the 250 ensemble members takes ~70 s, so build the ensemble once per process and reuse it
# across calls. It is built lazily on first use, inside the worker process that serves the request.
_ENSEMBLE = None


def _get_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = load_ensemble()
    return _ENSEMBLE


def load() -> None:
    """Optional hook: the web server calls it when a worker process starts, so the ensemble is loaded before
    the first request instead of during it."""
    _get_ensemble()


def _one_line_per_sequence(fasta_content: str) -> str:
    """Join sequences wrapped over several lines (valid FASTA, accepted by the input validation): `read_fasta`
    takes every non-header line as a sequence of its own. Blank lines are dropped."""
    records = []
    for line in fasta_content.splitlines():
        line = line.strip()
        if line.startswith(">"):
            records.append([line, ""])
        elif line and records:
            records[-1][1] += line
    return "".join(f"{header}\n{sequence}\n" for header, sequence in records)


def predict(input_data: Union[inputs.FastaFile, inputs.FastaData], output_dir: str) -> Dict[str, outputs.scheme.Output]:
    if isinstance(input_data, inputs.FastaFile):
        fasta_content = input_data.fasta_file
    elif isinstance(input_data, inputs.FastaData):
        fasta_content = input_data.fasta_data
    else:
        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    os.makedirs(output_dir, exist_ok=True)
    fasta_path = os.path.join(output_dir, "input_sequences.fasta")
    output_csv = os.path.join(output_dir, "predictions.csv")
    with open(fasta_path, "w") as f:
        f.write(_one_line_per_sequence(fasta_content))

    run_ensemble_prediction(sequences_fasta=fasta_path, output_csv=output_csv, ensemble=_get_ensemble())

    # The table shows the CSV itself: every row and every column, under the same names (the table view sorts it
    # by `prediction`). json round trip: plain Python values, NaN as null.
    table = json.loads(pd.read_csv(output_csv).to_json(orient="split", index=False))

    return {
        "table": outputs.Table(columns=table["columns"], rows=table["data"]),
        "file_reference": outputs.FileReference(path=output_csv),
    }
