import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import DatasetConfig

# @dataclass
# class DatasetConfig:
#     batch_size: int = 2
#     data_path: str = "SouthernCrossAI/Tweets_Australian_Cities"
#     max_seq_len: int = 1024
#     split_ratio: float = 0.2
#     sample_ratio: float = 1.0
#     column_name: str = "tweet"
#     tokenizer_name: str = "gpt2"
#     output_dir: str = field(init=False)

#     def __post_init__(self):
#         # Extract dataset name from the data_path
#         dataset_name = self.data_path.split("/")[-1]
#         # Incorporate the dataset name into the output directory
#         self.output_dir = f"./processed_data_{dataset_name}"

class GetJoeyData:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name, 
            clean_up_tokenization_spaces=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_loader = None
        self.val_loader = None
        self.load_data()

    def datasets(self):
        # Check if dataset is already saved to disk
        if os.path.exists(self.config.output_dir) and os.path.isdir(self.config.output_dir):
            print(f"Loading raw datasets from disk at {self.config.output_dir}...")
            datasets = DatasetDict.load_from_disk(self.config.output_dir)
        else:
            print("Loading dataset from Hugging Face Hub...")
            # Load raw dataset
            raw_dataset = load_dataset(self.config.data_path)
            dataset = raw_dataset['train']

            # (Optional) sampling logic could be placed here

            # Split dataset
            datasets = dataset.train_test_split(test_size=self.config.split_ratio)
            # Save dataset to disk
            datasets.save_to_disk(self.config.output_dir)
            print(f"Datasets saved to {self.config.output_dir}")

        return datasets

    def tokenize(self, dataset):
        # Ensure that the column exists
        if self.config.column_name not in dataset.column_names:
            raise ValueError(f"Column '{self.config.column_name}' does not exist in the dataset.")

        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples[self.config.column_name],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_seq_len,
                return_attention_mask=False
            ),
            batched=True
        )
        # Remove the original text column to save space
        tokenized_dataset = tokenized_dataset.remove_columns([self.config.column_name])
        tokenized_dataset.set_format(type="torch", columns=["input_ids"])
        return tokenized_dataset

    def create_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(
            self.config.training_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            self.config.validation_dataset, 
            batch_size=self.config.batch_size
        )
        return train_loader, val_loader

    def load_data(self):
        # First, load (or create) raw datasets
        raw_datasets = self.datasets()
        
        # Directory to store tokenized data
        tokenized_dir = os.path.join(self.config.output_dir, "tokenized")

        if os.path.exists(tokenized_dir) and os.path.isdir(tokenized_dir):
            print("Loading tokenized datasets from disk...")
            tokenized_datasets = DatasetDict.load_from_disk(tokenized_dir)
        else:
            print("Tokenizing datasets...")
            tokenized_train = self.tokenize(raw_datasets['train'])
            tokenized_val = self.tokenize(raw_datasets['test'])
            
            # Combine into a DatasetDict
            tokenized_datasets = DatasetDict({
                "train": tokenized_train,
                "test": tokenized_val
            })
            tokenized_datasets.save_to_disk(tokenized_dir)
            print(f"Tokenized datasets saved to {tokenized_dir}")

        self.config.training_dataset = tokenized_datasets['train']
        self.config.validation_dataset = tokenized_datasets['test']

        # Create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()


if __name__ == "__main__":
    # Testing
    config = DatasetConfig(tokenizer_name="gpt2")
    data_handler = GetJoeyData(config)

    for batch in data_handler.train_loader:
        print(batch["input_ids"].shape)
        break
