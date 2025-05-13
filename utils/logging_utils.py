import os
from datetime import datetime
import json

def create_experiment_folder(base_dir="logs", metadata=None):
    """
    Creates a timestamped experiment folder inside base_dir.
    Also creates subfolders for plots, models, renders.
    Optionally saves metadata.json if metadata dict is provided.
    Returns a dictionary with paths.
    """
    # Create timestamp like 20240428_1523
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # Create subfolders
    subfolders = ["plots", "infos"]
    paths = {"root": run_folder}
    for sub in subfolders:
        path = os.path.join(run_folder, sub)
        os.makedirs(path, exist_ok=True)
        paths[sub] = path

    # Optionally save metadata
    if metadata is not None:
        with open(os.path.join(run_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    return paths