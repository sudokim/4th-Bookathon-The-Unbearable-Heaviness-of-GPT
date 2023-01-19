import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding, PreTrainedTokenizerBase


class Collate:
    def __init__(self, input_ids_pad: int, attention_mask_pad: int, labels_pad: int):
        """
        Collate function for the dataloader
        :param input_ids_pad: Padding value for input_ids
        :param attention_mask_pad: Padding value for attention_mask
        :param labels_pad: Padding value for labels
        """
        self.input_ids_pad = input_ids_pad
        self.attention_mask_pad = attention_mask_pad
        self.labels_pad = labels_pad

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = [], [], []
        for d in batch:
            input_ids.append(d["input_ids"].squeeze())
            attention_mask.append(d["attention_mask"].squeeze())
            labels.append(d["labels"].squeeze())

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.input_ids_pad)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.attention_mask_pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.labels_pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CollateInputTarget:
    def __init__(self, input_ids_pad: int, attention_mask_pad: int, labels_pad: int, is_encoder_decoder: bool = False):
        """
        Collate function for the dataloader with two inputs
        :param input_ids_pad: Padding value for input_ids
        :param attention_mask_pad: Padding value for attention_mask
        :param labels_pad: Padding value for labels
        :param is_encoder_decoder: Whether the model is an encoder-decoder model
        """
        self.input_ids_pad = input_ids_pad
        self.attention_mask_pad = attention_mask_pad
        self.labels_pad = labels_pad
        self.is_encoder_decoder = is_encoder_decoder

    def __call__(self, batch: list[tuple[dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = [], [], []
        for input_result, target_result in batch:
            if self.is_encoder_decoder:
                input_ids.append(input_result["input_ids"].squeeze())
                attention_mask.append(input_result["attention_mask"].squeeze())
                labels.append(target_result["input_ids"].squeeze())
            else:
                input_ids.append(torch.cat([input_result["input_ids"], target_result["input_ids"]], dim=1).squeeze())
                attention_mask.append(
                        torch.cat([input_result["attention_mask"], target_result["attention_mask"]], dim=1).squeeze()
                )
                labels.append(torch.cat([input_result["labels"], target_result["labels"]], dim=1).squeeze())

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.input_ids_pad)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.attention_mask_pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.labels_pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CollateTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        """
        Collate function for the dataloader
        :param tokenizer: Tokenizer to use
        :param kwargs: Keyword arguments for the tokenizer
        """
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, batch: list[str]) -> BatchEncoding:
        tokenized = self.tokenizer(batch, return_tensors="pt", **self.kwargs)

        # Add labels
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["labels"][tokenized["labels"] == self.tokenizer.pad_token_id] = -100

        return tokenized


class CollateTokenizerKeywords:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ignore_input_loss: bool = False,
                 shuffle_keywords: bool = False, sep_token: str = None, join_token: str = " ",
                 train_sep_token: bool = True, **kwargs):
        """
        Collate function for the dataloader with keywords
        :param tokenizer: Tokenizer to use
        :param ignore_input_loss: Whether to ignore the loss for the input
        :param shuffle_keywords: Whether to shuffle the keywords
        :param sep_token: Token to use as separator
        :param join_token: Token to use to join the keywords
        :param train_sep_token: Whether to train the separator token
        :param kwargs: Keyword arguments for the tokenizer
        """
        self.tokenizer = tokenizer
        self.ignore_input_loss = ignore_input_loss
        self.shuffle_keywords = shuffle_keywords
        if sep_token is None:
            self.sep_token = self.tokenizer.eos_token
        else:
            self.sep_token = sep_token
        self.join_token = join_token
        self.train_sep_token = train_sep_token
        self.kwargs = kwargs

    def __call__(self, batch: list[tuple[list[str], str]]) -> BatchEncoding:
        """
        Tokenize the input and target
        :param batch: [(keywords, sentence), ...]
        :return: BatchEncoding
        """
        batch_joined = []
        keywords, target_sentences = zip(*batch)
        keywords: tuple[list[str]]
        target_sentences: list[str]

        for keyword, target_sentence in zip(keywords, target_sentences):
            if self.shuffle_keywords:
                keyword = keyword.copy()
                random.shuffle(keyword)

            keyword_joined = " ".join(keyword)
            batch_joined.append([keyword_joined, self.sep_token + target_sentence] if self.train_sep_token
                                else [keyword_joined + self.sep_token, target_sentence])

        tokenized = self.tokenizer(batch_joined, return_tensors="pt", return_token_type_ids=True, **self.kwargs)

        if self.ignore_input_loss:
            tokenized["labels"] = tokenized["input_ids"].clone()
            tokenized["labels"][tokenized["token_type_ids"] == 0] = -100
        else:
            tokenized["labels"] = tokenized["input_ids"].clone()
            tokenized["labels"][tokenized["labels"] == self.tokenizer.pad_token_id] = -100

        del tokenized["token_type_ids"]
        return tokenized


class CollateTokenizerKeywordsInputTarget:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, shuffle_keywords: bool = False, sep_token: str = None,
                 join_token: str = " ", **kwargs):
        """
        Collate function for the dataloader with keywords and two inputs
        :param tokenizer: Tokenizer to use
        :param shuffle_keywords: Whether to shuffle the keywords
        :param sep_token: Token to use as separator
        :param join_token: Token to use to join the keywords
        :param kwargs: Keyword arguments for the tokenizer
        """
        self.tokenizer = tokenizer
        self.shuffle_keywords = shuffle_keywords
        if sep_token is None:
            self.sep_token = self.tokenizer.eos_token
        else:
            self.sep_token = sep_token
        self.join_token = join_token
        self.kwargs = kwargs

    def __call__(self, batch: list[tuple[list[str], str, str]]) -> BatchEncoding:
        """
        Tokenize the input and target
        :param batch: [(keywords, input, target), ...]
        :return: BatchEncoding
        """
        batch_joined = []
        keywords, input_sentences, target_sentences = zip(*batch)
        keywords: tuple[list[str]]
        input_sentences: list[str]
        target_sentences: list[str]

        for keyword, input_sentence, target_sentence in zip(keywords, input_sentences, target_sentences):
            if self.shuffle_keywords:
                keyword = keyword.copy()
                random.shuffle(keyword)

            keyword_joined = " ".join(keyword)
            batch_joined.append([keyword_joined + self.sep_token + input_sentence, target_sentence])

        tokenized = self.tokenizer(batch_joined, return_tensors="pt", return_token_type_ids=True, **self.kwargs)

        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["labels"][tokenized["token_type_ids"] == 0] = -100

        del tokenized["token_type_ids"]
        return tokenized


class CollateTokenizerInputTarget:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, join_token: str = " ", **kwargs):
        """
        Collate function for the dataloader with keywords and two inputs
        :param tokenizer: Tokenizer to use
        :param join_token: Token to use to join the keywords
        :param kwargs: Keyword arguments for the tokenizer
        """
        self.tokenizer = tokenizer
        self.join_token = join_token
        self.kwargs = kwargs

    def __call__(self, batch: list[tuple[str, str]]) -> BatchEncoding:
        """
        Tokenize the input and target
        :param batch: [(keywords, input, target), ...]
        :return: BatchEncoding
        """
        batch_joined = []
        input_sentences, target_sentences = zip(*batch)
        input_sentences: list[str]
        target_sentences: list[str]

        for input_sentence, target_sentence in zip(input_sentences, target_sentences):
            batch_joined.append([input_sentence, target_sentence])

        tokenized = self.tokenizer(batch_joined, return_tensors="pt", return_token_type_ids=True, **self.kwargs)

        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["labels"][tokenized["token_type_ids"] == 0] = -100

        del tokenized["token_type_ids"]
        return tokenized
