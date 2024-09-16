import os
from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch

class BabyJoeyDataset:
    def __init__(self, data_path, sequence_length, train_file, valid_file):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.train_file = train_file
        self.valid_file = valid_file
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, dataset):
        return self.tokenizer(
            dataset['Paragraph'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.sequence_length,
            return_attention_mask=True
        )

    def load_or_create_datasets(self):
        if os.path.exists(self.train_file) and os.path.exists(self.valid_file):
            training_dataset = torch.load(self.train_file)
            validation_dataset = torch.load(self.valid_file)
            print("Loaded existing transformed datasets.")
        else:
            dataset = load_dataset(self.data_path)
            dataset = dataset['train'].train_test_split(test_size=0.2)

            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)

            training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)
            print("Transformed datasets created and saved.")

        return training_dataset, validation_dataset
