import os
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding, GPT2Tokenizer

from src.config.config import DataConfig, DataLoaderConfig
from src.utils import BabyJoeyUtil


class BabyJoeyDataset:
    def __init__(self, config: DataConfig) -> None:
        r"""Initialise a dataset class for BabyJoey based on configurations defined in 
            `src.config.config.DataConfig`.

        Args:
            config (DataConfig): Data configurations defined in `src.config.config.DataConfig`
        """

        self.data_path = config.data_path
        self.column_name = config.column_name
        self.sequence_length = config.sequence_length
        self.train_file = config.train_file
        self.valid_file = config.valid_file
        self.split_ratio = config.split_ratio
        # TODO: Hard-coded tokenizer. Allow users to customise their own tokeniser.
        # TODO: Consider to load tokenizer from config.py
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _tokenize_function(self, dataset: DatasetDict) -> BatchEncoding:
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

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        r"""Load tokenised datasets. Download from Hugging Face if they don't exist at local.

        Returns:
            Tuple[Dataset, Dataset]: A tuple of tokenised training dataset and tokenised validation dataset.
        """
        # Load tokenised datasets from local files if they exist
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            # print(f"Loading tokenised training set from `{self.train_file}`, tokenised validation set from `{self.valid_file}`...")
            train_dataset = torch.load(self.train_file, weights_only=False)
            val_dataset = torch.load(self.valid_file, weights_only=False)
            # print(f"Training set has {len(training_dataset)} data, validation set has {len(validation_dataset)} data")
        else:
            # print(f"Downloading dataset from `{self.data_path}`...")
            dataset = load_dataset(self.data_path)['train']
            # print("Finished downloading dataset from Hugging Face")

            # Split dataset into training and validation sets
            # print(f"Splitting dataset with split ratio of {self.split_ratio}...")
            dataset = dataset.train_test_split(test_size=self.split_ratio)
            # print(f"Training set has {len(dataset['train'])} data, validation set has {len(dataset['test'])} data")

            # Tokenise datasets
            # print("Tokenising training and validation sets...")
            train_dataset = dataset['train'].map(self._tokenize_function, batched=True)
            val_dataset = dataset['test'].map(self._tokenize_function, batched=True)

            # Set format for attention
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            # Save tokenised datasets to local paths
            torch.save(train_dataset, self.train_file)
            torch.save(val_dataset, self.valid_file)
            # print(f"Saved tokenised training set at `{self.train_file}`, tokenised validation set at `{self.valid_file}`")

        return train_dataset, val_dataset
        


class BabyJoeyDataLoader:
    def __init__(self, config: DataLoaderConfig) -> None:
        r"""Initialise a dataloader class for BabyJoey based on configurations defined in 
            `src.config.config.DataLoaderConfig`.

        Args:
            config (DataLoaderConfig): Data configurations defined in `src.config.config.DataLoaderConfig`
        """
        self.batch_size = config.batch_size

    def get_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        r"""Create dataloaders for training dataset and validation dataset.

        Args:
            train_dataset (Dataset): Tokenised training dataset
            val_dataset (Dataset): Tokenised validation dataset

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple of created training dataloader and validation dataloader
        """
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
