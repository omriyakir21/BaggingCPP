import os
import tempfile
from typing import Dict, Union
from tubiana_lab_utils.data import inputs, outputs
from inference_api.inference import predict as run_ensemble_prediction


def predict(input_data: Union[inputs.FastaFile, inputs.FastaData]) -> Dict[str, outputs.scheme.Output]:
    if isinstance(input_data, inputs.FastaFile):
        fasta_content = input_data.fasta_file
    elif isinstance(input_data, inputs.FastaData):
        fasta_content = input_data.fasta_data
    else:
        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    work_dir = tempfile.mkdtemp(prefix="bagging_cpp_")
    fasta_path = os.path.join(work_dir, "input_sequences.fasta")
    output_csv = os.path.join(work_dir, "predictions.csv")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    predictions_df = run_ensemble_prediction(sequences_fasta=fasta_path, output_csv=output_csv)

    # Keep the top (up to) 50 predictions, highest scoring first.
    top_df = predictions_df.sort_values("prediction", ascending=False).head(50)
    top_scores = {str(label): float(score) for label, score in zip(top_df["label"], top_df["prediction"])}

    return {
        "global_scores": outputs.GlobalScores(scores=top_scores),
        "file_reference": outputs.FileReference(path=output_csv),
    }
