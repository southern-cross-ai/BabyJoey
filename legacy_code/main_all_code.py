import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit

# Constants
TRAIN_FILE = 'training_dataset.pt'
VALID_FILE = 'validation_dataset.pt'
SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
DATA = "SouthernCrossAI/Project_Gutenberg_Australia"
VOCAB_SIZE = 50257
N_EMBD = 512
N_HEAD = 8
N_LAYER_DECODER = 1

# Dataset Class
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
            # Load the transformed datasets
            training_dataset = torch.load(self.train_file)
            validation_dataset = torch.load(self.valid_file)
            print("Loaded existing transformed datasets.")
        else:
            # Load the raw dataset and split
            dataset = load_dataset(self.data_path)
            dataset = dataset['train'].train_test_split(test_size=0.2)

            # Tokenize and transform the datasets
            training_dataset = dataset['train'].map(self.tokenize_function, batched=True)
            validation_dataset = dataset['test'].map(self.tokenize_function, batched=True)

            # Set format for PyTorch
            training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            # Save for future use
            torch.save(training_dataset, self.train_file)
            torch.save(validation_dataset, self.valid_file)
            print("Transformed datasets created and saved.")

        return training_dataset, validation_dataset

# DataLoader Class
class BabyJoeyDataLoader:
    def __init__(self, training_dataset, validation_dataset, batch_size):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size

    def get_dataloaders(self):
        training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        return training_dataloader, validation_dataloader

# Model Definition
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, N_EMBD)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
        return tokens + positions

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD)
        )
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x, key_padding_mask=None):
        x = x.transpose(0, 1)
        seq_len = x.size(0)  
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool() 
        attn_mask = attn_mask.unsqueeze(0).expand(x.size(1) * N_HEAD, -1, -1)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_output
        x = self.ln1(x)
        x = x.transpose(0, 1)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        return x

class BabyJoey(nn.Module):
    def __init__(self):
        super(BabyJoey, self).__init__()
        self.embeddings = Embeddings()
        self.decoder_blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER_DECODER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.embeddings(x)
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class BabyJoeyUnit(AutoUnit):
    def __init__(self, module, device=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, state, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()
        logits = self.module(input_ids, key_padding_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module):
        optimizer = optim.AdamW(module.parameters(), lr=1e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler

class TestingCallback(Callback):
    def on_train_start(self, state, unit):
        print("Training started!")
    def on_train_end(self, state, unit):
        print("Training ended!")
    def on_train_epoch_start(self, state, unit):
        print(f"Training epoch {unit.train_progress.num_epochs_completed} started!")
    def on_train_epoch_end(self, state, unit):
        print(f"Training epoch {unit.train_progress.num_epochs_completed} ended!")
    def on_train_step_start(self, state, unit):
        print(f"Training batch {unit.train_progress.num_steps_completed} started!")
    def on_train_step_end(self, state, unit):
        print(f"Training batch {unit.train_progress.num_steps_completed} ended!")
    def on_eval_start(self, state, unit):
        print("Evaluation started!")
    def on_eval_end(self, state, unit):
        print("Evaluation ended!")
    def on_eval_epoch_start(self, state, unit):
        print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} started!")
    def on_eval_epoch_end(self, state, unit):
        print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} ended!")
    def on_eval_step_start(self, state, unit):
        print(f"Evaluation batch {unit.eval_progress.num_steps_completed} started!")
    def on_eval_step_end(self, state, unit):
        print(f"Evaluation batch {unit.eval_progress.num_steps_completed} ended!")
    def on_exception(self, state, unit, exc: BaseException):
        print(f"Exception occurred: {exc}")

def main():
    # Load the datasets
    dataset = BabyJoeyDataset(DATA, SEQUENCE_LENGTH, TRAIN_FILE, VALID_FILE)
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Create DataLoaders
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset, BATCH_SIZE)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()

    # Define device and initialize the BabyJoey model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BabyJoey().to(device)
    baby_joey_unit = BabyJoeyUnit(module=model, device=device)

    # Train the model using the defined AutoUnit and callback
    fit(
        baby_joey_unit,
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=2,
        callbacks=[TestingCallback()]
    )

if __name__ == "__main__":
    main()
