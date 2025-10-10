import os
from datasets import load_dataset, DatasetDict
from typing import Dict

def load_local_dataset(data_dir="data/dataset"):

    dataset_paths: Dict[str, str] = {}

    for split_name in ["train", "validation", "test"]:
        path = os.path.join(data_dir, f"{split_name}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(data_dir, f"val.jsonl")
            if split_name == "validation" and os.path.exists(path):
                dataset_paths[split_name] = path
                continue
            path = os.path.join(data_dir, f"{split_name}.jsonl")

        if os.path.exists(path):
            dataset_paths[split_name] = path

    loaded_datasets: DatasetDict = load_dataset('json', data_files=dataset_paths)

    print("Datasets carregados:")
    for split, ds in loaded_datasets.items():
        print(f"  - {split}: {len(ds)} amostras")

    return loaded_datasets
