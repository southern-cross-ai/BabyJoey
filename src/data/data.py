import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, GPT2Tokenizer

from src.utils import BabyJoeyUtil
from src.config import BabyJoeyConfig

class BabyJoeyDataset:
    def __init__(self, cfg: BabyJoeyConfig) -> None:
        r"""Initialise a dataset class for BabyJoey using configuration.

        Args:
            cfg (BabyJoeyConfig): Configuration object containing dataset parameters.
        """
        self.data_path = cfg.data.data_path
        self.sequence_length = cfg.data.sequence_length
        self.train_file = cfg.data.train_file
        self.valid_file = cfg.data.valid_file
        self.split_ratio = cfg.data.split_ratio
        self.sample_ratio = cfg.data.sample_ratio
        self.column_name = cfg.data.column_name

        # Tokenizer setup
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, dataset: DatasetDict) -> BatchEncoding:
        r"""Tokenise a dataset. Truncate input sequence if it's longer than `sequence_length`.

        Args:
            dataset (DatasetDict): Input dataset to tokenise.

        Returns:
            BatchEncoding: Tokenised dataset.
        """
        return self.tokenizer(
            dataset[self.column_name],
            truncation=True,
            padding='max_length',
            max_length=self.sequence_length,
            return_attention_mask=True
        )

    def load_or_create_datasets(self) -> Tuple[Dataset, Dataset]:
        r"""Load tokenised datasets from Hugging Face if they are not existed. Otherwise, load from local files.

        Returns:
            Tuple[Dataset, Dataset]: Return the tokenised training set and validation set.
        """
        # Load tokenised datasets from local files if they exist
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            training_dataset = torch.load(self.train_file, weights_only=False)
            validation_dataset = torch.load(self.valid_file, weights_only=False)
        else:
            dataset = load_dataset(self.data_path)['train']
            dataset = dataset.train_test_split(test_size=self.split_ratio)

            # Tokenise datasets
            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)

            # Set format for attention
            training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            # Save tokenised datasets to local paths
            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)

        return training_dataset, validation_dataset


class BabyJoeyDataLoader:
    def __init__(self, cfg: BabyJoeyConfig, training_dataset: Dataset, validation_dataset: Dataset):
        r"""Initialise dataloaders for training and validation sets using configuration.

        Args:
            cfg (BabyJoeyConfig): Configuration object containing dataloader parameters.
            training_dataset (Dataset): Training dataset to use.
            validation_dataset (Dataset): Validation dataset to use.
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = cfg.dataloader.batch_size

    def get_dataloaders(self, distributed: bool = False):
        """Create dataloaders for training and validation datasets, with optional distributed mode."""
        if distributed:
            train_sampler = DistributedSampler(self.training_dataset)
            val_sampler = DistributedSampler(self.validation_dataset)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
        val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, sampler=val_sampler)

        return train_loader, val_loader
