import json
import os
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.dataloader_collate import CollateTokenizer


class BookathonDatasetWithSlidingWindow(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, train: bool,
                 max_length: int = 512, num_sentences: int = 8, step: int = 2,
                 dataset_shuffle_seed: int = 42):
        """
        Bookathon split_dataset with sliding window to train the entire corpus
        :param file_path: Path to the split_dataset
        :param tokenizer: Tokenizer to use
        :param train: Whether this is the train split_dataset
        :param max_length: Max length of the input
        :param num_sentences: Number of sentences to use
        :param step: Step size for the sliding window
        :param dataset_shuffle_seed: Seed to shuffle the split_dataset
        """
        self.dataset_path = file_path
        self.tokenizer = tokenizer
        self.train = train
        self.raw_dataset: list[list[str]] = json.load(open(file_path, "r", encoding="utf-8"))
        self.max_length = max_length
        self.num_sentences = num_sentences
        self.step = step

        # Shuffle the split_dataset
        if self.train:
            current_seed = random.getstate()
            random.seed(dataset_shuffle_seed)
            random.shuffle(self.raw_dataset)
            random.setstate(current_seed)

        # Use the sliding window to augment the split_dataset
        self.sliding_window_dataset: list[str] = []
        for prompt in tqdm(self.raw_dataset, desc="Sliding window"):
            for i in range(0, len(prompt) - num_sentences + 1, step):
                prompt_slice = prompt[i:i + self.num_sentences]

                self.sliding_window_dataset.append(" ".join(prompt_slice))

    def __len__(self):
        return len(self.sliding_window_dataset)

    def __getitem__(self, idx):
        """
        Get an idx from the split_dataset
        :param idx: Index of the idx
        :return: input_result[dict], target_result[dict]
        """
        return self.sliding_window_dataset[idx]


class BookathonDataModuleWithSlidingWindow(pl.LightningDataModule):
    def __init__(self, dataset_path: str, tokenizer: PreTrainedTokenizerBase,
                 input_ids_pad: int = None, attention_mask_pad: int = None, labels_pad: int = None,
                 dataloader_num_workers: int = None, batch_size: int = 8, max_length: int = 1024,
                 num_sentences: int = 4, step: int = 2,
                 input_sentence_loss: bool = True, dataset_shuffle_seed: int = 42):
        """
        Bookathon dataset with sliding window

        The dataset must be split into sentences and saved as a json file.
        :param dataset_path: Path to the split_dataset
        :param tokenizer: Tokenizer to use
        :param input_ids_pad: Padding value for input_ids
        :param attention_mask_pad: Padding value for attention_mask
        :param labels_pad: Padding value for labels
        :param dataloader_num_workers: Number of workers for the dataloader
        :param batch_size: Batch size
        :param max_length: Max length of the input
        :param num_sentences: Number of sentences to use
        :param step: Step size for the sliding window
        :param input_sentence_loss: Whether to calculate loss for the input sentences
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
        self.input_sentence_loss = input_sentence_loss
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

        self.collate_fn = CollateTokenizer(self.tokenizer, padding=True, truncation=True, max_length=self.max_length,
                                           pad_to_multiple_of=8)

    def setup(self, stage=None):
        match stage:
            case "fit" | None:
                self.train = BookathonDatasetWithSlidingWindow(
                        self.train_file_path, self.tokenizer, True, self.max_length,
                        self.num_sentences, self.step, self.dataset_shuffle_seed,
                )
                self.val = BookathonDatasetWithSlidingWindow(
                        self.val_file_path, self.tokenizer, False, self.max_length,
                        self.num_sentences, self.step, self.dataset_shuffle_seed,
                )

            case "test":
                self.test = BookathonDatasetWithSlidingWindow(
                        self.test_file_path, self.tokenizer, False, self.max_length,
                        self.num_sentences, self.step, self.dataset_shuffle_seed,
                )

            case _:
                raise ValueError("Invalid stage")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.dataloader_num_workers)
