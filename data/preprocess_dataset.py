import json
import os
import random
import re

from tqdm import tqdm

SEED = 42


def preprocess(x: str):
    x = re.sub(r"\n+", " ", x)
    x = re.sub(r" +", " ", x)
    x = x.strip()
    return x


def recursive_search(path: str) -> list[str]:
    """
    Recursively search for all json files
    :param path: Path to search
    :return: List of json files
    """
    files = []
    for file in os.listdir(path):
        if "unused" in file:
            print(f"Skipping unused file/directory: {file}")
            continue
        elif file == "train.json" or file == "test.json" or file == "valid.json":
            continue
        elif os.path.isdir(os.path.join(path, file)):
            files += recursive_search(os.path.join(path, file))
        elif file.endswith(".json"):
            files.append(os.path.join(path, file))

    return files


if __name__ == '__main__':
    json_files = recursive_search(os.curdir)  # List of json files
    print(f"Found {len(json_files)} json files")

    random.seed(SEED)

    all_dataset = []

    for json_file in tqdm(json_files, "Loading files", unit="file"):
        dataset = json.load(open(json_file, "r", encoding="utf-8"))
        all_dataset += [x["text"] for x in dataset]

    print(f"Total split_dataset size: {len(all_dataset)}")

    all_dataset = [preprocess(x) for x in tqdm(all_dataset, desc="Preprocessing split_dataset")
                   if isinstance(x, str) and len(x) > 32]

    train_size = int(len(all_dataset) * 0.90)
    valid_size = int(len(all_dataset) * 0.05)
    test_size = len(all_dataset) - train_size - valid_size

    random.shuffle(all_dataset)

    train_dataset = all_dataset[:train_size]
    valid_dataset = all_dataset[train_size:train_size + valid_size]
    test_dataset = all_dataset[train_size + valid_size:]

    json.dump(train_dataset, open(os.path.join(os.curdir, "train.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    json.dump(valid_dataset, open(os.path.join(os.curdir, "valid.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    json.dump(test_dataset, open(os.path.join(os.curdir, "test.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
