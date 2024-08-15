import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.config import BabyJoeyConfig

class GutenbergData:
    def __init__(self, config: BabyJoeyConfig, model_name="gpt2"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set a padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def dataset(self):
        # Load the dataset based on the configuration
        raw_dataset = load_dataset(self.config.dataset_name)
        
        # Check if the dataset has a validation split, if not create one
        if 'validation' not in raw_dataset:
            train_test_split = raw_dataset['train'].train_test_split(test_size=0.1)
            raw_dataset = DatasetDict({
                'train': train_test_split['train'],
                'validation': train_test_split['test']
            })

        return raw_dataset

    def tokenized(self, raw_dataset):
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["Paragraph"],  # Ensure "Paragraph" matches your dataset's column name
                padding="max_length",
                truncation=True,
                max_length=self.config.max_position_embeddings
            )

        # Apply tokenization to the dataset
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["Paragraph"])
        return tokenized_dataset

    def dataloader(self, split='train'):
        # Step 1: Load the raw dataset
        raw_dataset = self.dataset()
        
        # Step 2: Tokenize the dataset
        tokenized_dataset = self.tokenized(raw_dataset)

        # Step 3: Prepare the DataLoader
        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        dataloader = DataLoader(
            tokenized_dataset[split], 
            batch_size=self.config.batch_size, 
            shuffle=True if split == 'train' else False, 
            collate_fn=collate_fn
        )
        
        return dataloader
