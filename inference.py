import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
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

    work_dir = "/home/iscb/wolfson/lab_tools/data/bagging_cpp"
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    day_folder = now.strftime("%Y-%m-%d")
    time_random_folder = now.strftime("%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
    request_dir = os.path.join(work_dir, day_folder, time_random_folder)
    os.makedirs(request_dir, exist_ok=True)
    fasta_path = os.path.join(request_dir, "input_sequences.fasta")
    output_csv = os.path.join(request_dir, "predictions.csv")
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
