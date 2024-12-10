import os
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, GPT2Tokenizer

class BabyJoeyDataset:
    def __init__(self) -> None:
        self.data_path = "SouthernCrossAI/Project_Gutenberg_Australia"
        self.max_seq_len = 1024
        self.train_file = "train_data.pt"
        self.valid_file = "valid_data.pt"
        self.split_ratio = 0.2
        self.sample_ratio = 0.1
        self.column_name = "Paragraph"

        # Tokenizer setup
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, dataset: DatasetDict) -> BatchEncoding:
        return self.tokenizer(
            dataset[self.column_name],
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_attention_mask=False  # Removed attention_mask generation
        )

    def load_or_create_datasets(self):
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            training_dataset = torch.load(self.train_file)
            validation_dataset = torch.load(self.valid_file)
            
        else:
            dataset = load_dataset(self.data_path)['train']

            # Split dataset into training and validation sets
            dataset = dataset.train_test_split(test_size=self.split_ratio)

            # Tokenize datasets
            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)

            # Format the datasets to return only input_ids
            training_dataset.set_format(type='torch', columns=['input_ids'])
            validation_dataset.set_format(type='torch', columns=['input_ids'])

            # Save tokenized datasets to local paths
            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)

        return training_dataset, validation_dataset


class BabyJoeyDataLoader:
    def __init__(self, training_dataset: Dataset, validation_dataset: Dataset):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = 2

    def get_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size)

        # Inspect the shape of one batch (now only input_ids)
        for batch in train_loader:
            input_ids = batch['input_ids']
            print(f"Input IDs shape: {input_ids.size()}")  # Shape: (batch_size, seq_length)
            break  # Print shapes for the first batch only
        
        return train_loader, val_loader
