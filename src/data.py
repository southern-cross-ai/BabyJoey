import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.config import BabyJoeyConfig

class GutenbergData:
    def __init__(self, config: BabyJoeyConfig, model_name="gpt2"):
        self.config = config
        print("Initializing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenized_dataset = None  # Add this to cache the tokenized dataset
        
        # Set a padding token if not already set
        if self.tokenizer.pad_token is None:
            print("Setting the pad token...")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def raw_dataset(self):
        """
        Load the raw dataset based on the configuration.
        """
        print("Loading the raw dataset...")
        raw_dataset = load_dataset(self.config.dataset_name)
        return raw_dataset

    def tokenized_dataset(self):
        """
        Tokenize the entire dataset and split it into training and validation sets.
        """
        if self._tokenized_dataset is not None:
            print("Using cached tokenized dataset...")
            return self._tokenized_dataset
        
        # Load the raw dataset
        print("Loading and tokenizing the dataset...")
        raw_dataset = self.raw_dataset()

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["Paragraph"],  # Ensure "Paragraph" matches your dataset's column name
                padding="max_length",
                truncation=True,
                max_length=self.config.max_position_embeddings
            )

        # Apply tokenization to the entire dataset
        print("Mapping the tokenization function over the dataset...")
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["Paragraph"])

        # Check if the dataset has a validation split, if not create one
        if 'validation' not in tokenized_dataset:
            print("Creating validation split...")
            train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
            tokenized_dataset = DatasetDict({
                'train': train_test_split['train'],
                'validation': train_test_split['test']
            })

        self._tokenized_dataset = tokenized_dataset  # Cache the tokenized dataset
        return tokenized_dataset

    def dataloader(self, split='train'):
        """
        Prepare the DataLoader for training or validation.
        """
        # Load and tokenize the dataset, then split if necessary
        print(f"Preparing DataLoader for {split} split...")
        tokenized_dataset = self.tokenized_dataset()

        # Prepare the DataLoader
        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        print(f"Creating DataLoader for the {split} dataset...")
        dataloader = DataLoader(
            tokenized_dataset[split], 
            batch_size=self.config.batch_size, 
            shuffle=True if split == 'train' else False, 
            collate_fn=collate_fn
        )
        
        return dataloader
