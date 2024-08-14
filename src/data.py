import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class GutenbergData:
    def __init__(self, dataset_name="SouthernCrossAI/Project_Gutenberg_Australia", 
                 model_name="gpt2", 
                 batch_size=8, 
                 max_length=512):
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length

        # Set a padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_tokenize_dataset(self):
        # Load the dataset
        dataset = load_dataset(self.dataset_name)

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(examples["Paragraph"], 
                                  padding="max_length", 
                                  truncation=True, 
                                  max_length=self.max_length)

        # Apply tokenization to the dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Paragraph"])
        
        return tokenized_datasets

    def get_dataloader(self, split='train'):
        # Load and tokenize the dataset
        tokenized_datasets = self.load_and_tokenize_dataset()

        # Convert lists to tensors in the DataLoader
        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        # Create and return the DataLoader
        dataloader = DataLoader(tokenized_datasets[split], batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        return dataloader
