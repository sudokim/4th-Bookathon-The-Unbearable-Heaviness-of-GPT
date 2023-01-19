import json
import os
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.dataloader_collate import CollateTokenizer


class BookathonDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, train: bool, max_length: int = 1024,
                 dataset_shuffle_seed: int = 42):
        """
        Bookathon dataset
        :param file_path: Path to the dataset
        :param tokenizer: Tokenizer to use
        :param train: Whether this is the train dataset
        :param max_length: Max length of the input
        :param dataset_shuffle_seed: Seed to shuffle the dataset
        """
        self.dataset_path = file_path
        self.tokenizer = tokenizer
        self.dataset: list[list[str]] = json.load(open(file_path, "r", encoding="utf-8"))
        self.train = train
        self.max_length = max_length

        # Shuffle the dataset
        if self.train:
            current_seed = random.getstate()
            random.seed(dataset_shuffle_seed)
            random.shuffle(self.dataset)
            random.setstate(current_seed)

        self.dataset = [" ".join(prompt) for prompt in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        prompt = self.dataset[idx]

        return prompt


class BookathonDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, tokenizer: PreTrainedTokenizerBase,
                 input_ids_pad: int = None, attention_mask_pad: int = None, labels_pad: int = None,
                 dataloader_num_workers: int = None, batch_size: int = 8, max_length: int = 1024,
                 dataset_shuffle_seed: int = 42):
        """
        Bookathon dataset
        :param dataset_path: Path to the dataset
        :param tokenizer: Tokenizer to use
        :param input_ids_pad: Padding value for input_ids
        :param attention_mask_pad: Padding value for attention_mask
        :param labels_pad: Padding value for labels
        :param dataloader_num_workers: Number of workers for the dataloader
        :param batch_size: Batch size
        :param max_length: Maximum length of the input
        :param dataset_shuffle_seed: Seed to shuffle the dataset
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
        self.dataset_shuffle_seed = dataset_shuffle_seed

        self.train = None
        self.val = None
        self.test = None

        # Collate function
        if input_ids_pad is None:
            self.input_ids_pad = tokenizer.pad_token_id
        else:
            self.input_ids_pad = input_ids_pad

        if attention_mask_pad is None:
            self.attention_mask_pad = 0
        else:
            self.attention_mask_pad = attention_mask_pad

        if labels_pad is None:
            self.labels_pad = -100
        else:
            self.labels_pad = labels_pad

        self.collate_function = CollateTokenizer(self.tokenizer, padding=True, truncation=True,
                                                 max_length=self.max_length,
                                                 pad_to_multiple_of=8)

    def setup(self, stage=None):
        match stage:
            case "fit" | None:
                self.train = BookathonDataset(
                        self.train_file_path, self.tokenizer, True, self.max_length, self.dataset_shuffle_seed
                )
                self.val = BookathonDataset(
                        self.val_file_path, self.tokenizer, False, self.max_length, self.dataset_shuffle_seed
                )

            case "test":
                self.test = BookathonDataset(
                        self.test_file_path, self.tokenizer, False, self.max_length, self.dataset_shuffle_seed
                )

            case _:
                raise ValueError("Invalid stage")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_function, num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers,
                          collate_fn=self.collate_function)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers,
                          collate_fn=self.collate_function)
