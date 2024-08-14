import torch
from datasets import load_dataset  # Add this import
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.config import BabyJoeyConfig

class GutenbergData:
    def __init__(self, config: BabyJoeyConfig, dataset_name="SouthernCrossAI/Project_Gutenberg_Australia", model_name="gpt2"):
        self.dataset_name = dataset_name
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set a padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_tokenize_dataset(self):
        dataset = load_dataset(self.dataset_name)

        def tokenize_function(examples):
            return self.tokenizer(examples["Paragraph"], 
                                  padding="max_length", 
                                  truncation=True, 
                                  max_length=self.config.max_position_embeddings)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Paragraph"])
        
        return tokenized_datasets

    def get_dataloader(self, split='train'):
        tokenized_datasets = self.load_and_tokenize_dataset()

        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        dataloader = DataLoader(tokenized_datasets[split], batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        return dataloader
