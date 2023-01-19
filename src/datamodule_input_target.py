import json
import os
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.dataloader_collate import CollateTokenizerInputTarget


class BookathonDatasetInputTarget(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, train: bool,
                 max_length: int = 512, num_sentences: int = 12, step: int = 2, num_input_sentences: int = 3,
                 dataset_shuffle_seed: int = 42):
        """
        Bookathon split_dataset with sliding window and input, target to train the entire corpus
        :param file_path: Path to the split_dataset
        :param tokenizer: Tokenizer to use
        :param train: Whether this is the train split_dataset
        :param max_length: Max length of the input
        :param num_sentences: Number of sentences to use
        :param step: Step size for the sliding window
        :param num_input_sentences: Number of sentences to use as input
        :param dataset_shuffle_seed: Seed to shuffle the split_dataset
        """

        self.dataset_path = file_path
        self.tokenizer = tokenizer
        self.train = train
        self.raw_dataset: list[list[str]] = json.load(open(file_path, "r", encoding="utf-8"))
        self.max_length = max_length
        self.num_sentences = num_sentences
        self.step = step
        self.num_input_sentences = num_input_sentences
        self.dataset_shuffle_seed = dataset_shuffle_seed

        assert self.num_input_sentences <= self.num_sentences, \
            f"num_input_sentences ({self.num_input_sentences}) must be <= num_sentences ({self.num_sentences})"

        # Filter out prompts with less than num_sentences // 2 sentences
        before_length = len(self.raw_dataset)
        self.raw_dataset = [prompt for prompt in self.raw_dataset if len(prompt) >= num_sentences // 2]
        print(f"Dropped {before_length - len(self.raw_dataset)} prompts "
              f"with less than {num_sentences // 2} sentences")

        # Shuffle the split_dataset
        if self.train:
            current_seed = random.getstate()
            random.seed(dataset_shuffle_seed)
            random.shuffle(self.raw_dataset)
            random.setstate(current_seed)

        # Initialize the sliding window dataset
        self.sliding_window_dataset: list[tuple[str, str]] = []
        for prompt_id, prompt in enumerate(tqdm(self.raw_dataset, desc="Sliding window")):
            for i in range(0, len(prompt) - self.num_sentences + 1, self.step):
                prompt_slice = prompt[i:i + self.num_sentences]
                # Input/target split
                self.sliding_window_dataset.append((
                    " ".join(prompt_slice[:self.num_input_sentences]),
                    " ".join(prompt_slice[self.num_input_sentences:]))
                )

    def __len__(self):
        return len(self.sliding_window_dataset)

    def __getitem__(self, idx) -> tuple[str, str]:
        """
        Get an item from the split_dataset
        :param idx: Index of the item
        :return: keyword, prompt
        """
        return self.sliding_window_dataset[idx]


class BookathonDataModuleInputTarget(pl.LightningDataModule):
    def __init__(self, dataset_path: str, tokenizer: PreTrainedTokenizerBase, dataloader_num_workers: int = None,
                 batch_size: int = 8, max_length: int = 1024, num_sentences: int = 4, step: int = 2,
                 num_input_sentences: int = 3, dataset_shuffle_seed: int = 42):
        """
        Bookathon dataset with keywords to summarize previous sentences

        The dataset must be split into sentences and saved as a json file.
        :param dataset_path: Path to the split_dataset
        :param tokenizer: Tokenizer to use
        :param dataloader_num_workers: Number of workers for the dataloader
        :param batch_size: Batch size
        :param max_length: Max length of the input
        :param num_sentences: Number of sentences to use
        :param step: Step size for the sliding window
        :param num_input_sentences: Number of sentences to use as input
               (num_target_sentences = num_sentences - num_input_sentences)
        :param dataset_shuffle_seed: Seed to shuffle the split_dataset
        """

        super().__init__()

        self.dataset_path = dataset_path
        self.train_file_path = os.path.join(dataset_path, "train.json")
        self.val_file_path = os.path.join(dataset_path, "valid.json")
        self.test_file_path = os.path.join(dataset_path, "test.json")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataloader_num_workers = dataloader_num_workers
        self.num_sentences = num_sentences
        self.step = step
        self.num_input_sentences = num_input_sentences
        self.dataset_shuffle_seed = dataset_shuffle_seed

        self.train = None
        self.val = None
        self.test = None

        self.collate_fn = CollateTokenizerInputTarget(tokenizer, padding=True, truncation=True,
                                                      max_length=self.max_length, pad_to_multiple_of=8)

    def setup(self, stage=None):
        match stage:
            case "fit" | None:
                self.train = BookathonDatasetInputTarget(
                        file_path=self.train_file_path,
                        tokenizer=self.tokenizer,
                        train=True,
                        max_length=self.max_length,
                        num_sentences=self.num_sentences,
                        step=self.step,
                        num_input_sentences=self.num_input_sentences,
                        dataset_shuffle_seed=self.dataset_shuffle_seed,
                )
                self.val = BookathonDatasetInputTarget(
                        file_path=self.val_file_path,
                        tokenizer=self.tokenizer,
                        train=False,
                        max_length=self.max_length,
                        num_sentences=self.num_sentences,
                        step=self.step,
                        num_input_sentences=self.num_input_sentences,
                        dataset_shuffle_seed=self.dataset_shuffle_seed,
                )

            case "test":
                self.test = BookathonDatasetInputTarget(
                        file_path=self.test_file_path,
                        tokenizer=self.tokenizer,
                        train=False,
                        max_length=self.max_length,
                        num_sentences=self.num_sentences,
                        step=self.step,
                        num_input_sentences=self.num_input_sentences,
                        dataset_shuffle_seed=self.dataset_shuffle_seed,
                )

            case _:
                raise ValueError(f"Stage {stage} is not valid.")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.dataloader_num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers,
                          collate_fn=self.collate_fn)
