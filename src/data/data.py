import os
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, GPT2Tokenizer

from src.utils import BabyJoeyUtil

class BabyJoeyDataset:
    def __init__(self, 
                 data_path: str, 
                 column_name: str, 
                 train_file: str, 
                 valid_file: str, 
                 split_ratio: float = 0.2, 
                 sample_ratio: float = 0.3, 
                 sequence_length: int = 128
                 ) -> None:
        r"""Initialise a dataset class for BabyJoey.

        Args:
            data_path (str): Dataset in Hugging Face (e.g., SouthernCrossAI/Project_Gutenberg_Australia)
            column_name (str): Column name that contains the text in the dataset
            sequence_length (int): Maximum sequence length for input sequences
            train_file (str): File path for training set
            valid_file (str): File path for validation set
            split_ratio (float, optional): Split ratio for validation set. Defaults to 0.2.
            sample_ratio (float, optional): Sample ratio of whole dataset. Set to 1 for using whole dataset. 
                                            Defaults to 0.3.
        """
        self.data_path = data_path
        # TODO: move this attr to load_or_create_datasets or add new arg to tokenize_function
        self.sequence_length = sequence_length
        self.train_file = train_file
        self.valid_file = valid_file
        # TODO: Current tokeniser is hard-encoded. Should allow users to load their own tokenisers
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.split_ratio = split_ratio
        self.column_name = column_name
        self.sample_ratio = sample_ratio

    # TODO: name is redundant
    def tokenize_function(self, dataset: DatasetDict) -> BatchEncoding:
        r"""Tokenise a dataset. Truncate input sequence if it's longer than `sequence_length`.

        Args:
            dataset (DatasetDict): input dataset to tokenise

        Returns:
            BatchEncoding: Tokenised dataset
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
            Tuple[Dataset, Dataset]: Return the tokenised training set and validation set
        """
        # Load tokenised datasets from local files if they are existed
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            print(f"Loading tokenised training set from `{self.train_file}`, tokenised validation set from `{self.valid_file}`...")
            training_dataset = torch.load(self.train_file)
            validation_dataset = torch.load(self.valid_file)
            print(f"Training set has {len(training_dataset)} data, validation set has {len(validation_dataset)} data")
        else:
            print(f"Downloading dataset from `{self.data_path}`...")
            # Pull datasets from Hugging Face
            dataset = load_dataset(self.data_path)['train']
            print("Finished downloading dataset from Hugging Face")
            # Sample from dataset if needed
            if 0 < self.sample_ratio < 1:
                _n_old, _n_new = len(dataset), int(len(dataset) * self.sample_ratio)
                print(f"Sampling dataset with ratio of {self.sample_ratio}...")
                dataset = BabyJoeyUtil.sample_dataset(dataset, self.sample_ratio)
                print(f'Original dataset has {_n_old} data, sampled dataset has {_n_new} data')
            # Split dataset into training set and validation set
            print(f"Splitting dataset with split ratio of {self.split_ratio}...")
            dataset = dataset.train_test_split(test_size=self.split_ratio)
            print(f"Training set has {len(dataset['train'])} data, validation set has {len(dataset['test'])} data")
            # Tokenise the loaded datasets by a tokeniser
            print("Tokenising training set and validation set...")
            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)
            # Set format for attention
            training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            # Save the tokenised dataset into local paths
            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)
            print(f"Saved tokenised training set at `{self.train_file}`, tokenised validation set at `{self.valid_file}`")
            
        return training_dataset, validation_dataset

class BabyJoeyDataLoader:
    def __init__(self, training_dataset, validation_dataset, batch_size):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Generate dataloaders for training and validation.

        Returns:
            Tuple[DataLoader, DataLoader]: Returned dataloaders
        """
        training_dataloader = DataLoader(self.training_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        validation_dataloader = DataLoader(self.validation_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
        return training_dataloader, validation_dataloader
