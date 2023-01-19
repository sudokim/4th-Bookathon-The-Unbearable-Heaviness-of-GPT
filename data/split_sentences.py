import json
import os
from sys import argv

from kss import split_sentences
from tqdm.contrib.concurrent import process_map


def ss(document: str):
    result = split_sentences(document)[:-1]

    result = map(lambda x: x.strip(), result)

    def filter_sentence(s: str):
        return not (s.startswith("* ") or s.startswith("- ") or s.startswith("# ") or "photo by" in s.lower() or (
                "사진" in s and "출처" in s))

    result = list(filter(filter_sentence, result))

    if len(result) <= 5:
        return None

    return result


def split(file: str):
    filename = file[:file.rfind('.')]
    extension = file[file.rfind('.') + 1:]
    print("Filename:", filename)
    print("Extension:", extension)

    # Rename file to _before.json
    os.rename(file, filename + "_before." + extension)

    dataset = json.load(open(filename + "_before." + extension, "r", encoding="utf-8"))

    print("Before:", len(dataset))

    dataset_split = process_map(ss, dataset, max_workers=16, chunksize=500)

    # Filter out None
    dataset_split = list(filter(lambda x: x is not None, dataset_split))

    print("After:", len(dataset_split))

    json.dump(dataset_split, open(filename + extension, 'w'), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(split_sentences("Start splitting!"))
    split(argv[1])
