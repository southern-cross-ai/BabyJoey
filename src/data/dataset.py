import os
from typing import Tuple

import torch
from datasets import (  # load datasets from Hugging Face
    Dataset,
    DatasetDict,
    load_dataset,
)

# load pre-trained GPT2 tokenizer
from transformers import BatchEncoding, GPT2Tokenizer


class BabyJoeyDataset:
    def __init__(self, data_path: str, column_name: str, train_file: str, valid_file: str, 
                 split_ratio: float = 0.2, sequence_length: int = 128) -> None:
        """Initialise a dataset class for BabyJoey

        Args:
            data_path (str): Dataset in Hugging Face (e.g., "SouthernCrossAI/Project_Gutenberg_Australia")
            column_name (str): Column name that contains the text in the dataset.
            sequence_length (int): Maximum sequence length for input sequences
            train_file (str): File path for training set
            valid_file (str): File path for validation set
            split_ratio (float, optional): Split ratio for validation set. Defaults to 0.2.
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

    # TODO: name is redundant
    def tokenize_function(self, dataset: DatasetDict) -> BatchEncoding:
        """Tokenise a dataset. Truncate input sequence if it's longer than `sequence_length`.

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
        """Load tokenised datasets from Hugging Face if they are not existed. Otherwise, load from local files.

        Returns:
            Tuple[Dataset, Dataset]: Return the tokenised training set and validation set
        """
        # Load tokenised datasets from local files if they are existed
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            training_dataset = torch.load(self.train_file)
            validation_dataset = torch.load(self.valid_file)
            print("Loaded existing transformed datasets.")
        else:
            # Pull datasets from Hugging Face
            dataset = load_dataset(self.data_path)
            dataset = dataset['train'].train_test_split(test_size=self.split_ratio)
            # Tokenise the loaded datasets by a tokeniser
            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)
            # Set format for attention
            training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            # Save the tokenised dataset into local paths
            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)
            print(f"Tokenised training set has been saved at:\n{self.train_file}")
            print(f"Tokenised validation set has been saved at:\n{self.valid_file}")
        return training_dataset, validation_dataset
