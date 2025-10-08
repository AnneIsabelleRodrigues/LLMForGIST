import os
from datasets import load_dataset

def load_local_dataset(data_dir="data/dataset"):

    files = {
        "seed_train": None,
        "seed_val": None,
        "seed_test": None
    }

    for split in files.keys():
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            files[split] = path
            break

    if files["train"] is None:
        raise FileNotFoundError("⚠️ Arquivo de treino não encontrado")

    dataset = load_dataset('csv', data_files={k: v for k, v in files.items() if v is not None})

    for split, ds in dataset.items():
        print(f"  - {split}: {len(ds)} amostras")

    return dataset
