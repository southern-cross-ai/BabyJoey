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

# File paths to save/load the datasets
train_file = 'training_dataset.pt'
valid_file = 'validation_dataset.pt'

# Data and model configuration
data = "SouthernCrossAI/Project_Gutenberg_Australia"
sequence_length = 512
batch_size = 32
vocab_size = 50257
n_embd = 512
n_head = 8
n_layer_decoder = 1

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(dataset):
    return tokenizer(
        dataset['Paragraph'], 
        truncation=True, 
        padding='max_length', 
        max_length=sequence_length,
        return_attention_mask=True
    )

# Load and tokenize datasets
if os.path.exists(train_file) and os.path.exists(valid_file):
    # Load the transformed datasets
    training_dataset = torch.load(train_file)
    validation_dataset = torch.load(valid_file)
    print("Loaded existing transformed datasets.")
else:
    # Load the raw dataset
    ds = load_dataset(data)
    dataset = ds['train'].train_test_split(test_size=0.2)
    
    # Tokenize and transform the datasets
    training_dataset = dataset['train'].map(tokenize_function, batched=True)
    validation_dataset = dataset['test'].map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Save for future use
    torch.save(training_dataset, train_file)
    torch.save(validation_dataset, valid_file)
    print("Transformed datasets created and saved.")

# DataLoader for training and validation datasets
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Model Definition
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
        return tokens + positions

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, key_padding_mask=None):
        # Transpose for MultiheadAttention (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # Generate causal mask (triangular lower matrix to prevent attending to future tokens)
        seq_len = x.size(0)  # Sequence length after transpose
        batch_size = x.size(1)  # Batch size after transpose
        
        # Adjust causal mask for batch size and number of heads
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()  # Convert to boolean
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size * n_head, -1, -1)

        # Pass the mask to the attention layer
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        x = x + attn_output
        x = self.ln1(x)

        # Transpose back to (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)

        return x

class BabyJoey(nn.Module):
    def __init__(self):
        super(BabyJoey, self).__init__()
        
        # Embeddings
        self.embeddings = Embeddings()
        
        # Decoder Blocks (based on n_layer_decoder)
        self.decoder_blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layer_decoder)])

        # Output layer
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Get embeddings
        x = self.embeddings(x)

        # Apply decoder blocks with attention mask
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)  # Pass key_padding_mask to the block

        # Layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

# BabyJoey AutoUnit definition
class BabyJoeyUnit(AutoUnit):
    def __init__(self, module, device=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, state, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        
        # Ensure the attention mask is of type bool (for key_padding_mask)
        key_padding_mask = (attention_mask == 0).bool()

        # The model will now handle both causal masking (inside the TransformerBlock) and key_padding_mask
        logits = self.module(input_ids, key_padding_mask=key_padding_mask)

        # Shift the input ids by one to get the target sequence
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module):
        # Define the optimizer
        optimizer = optim.AdamW(module.parameters(), lr=1e-4, weight_decay=1e-2)
        
        # For simplicity, we'll return `None` for the learning rate scheduler.
        return optimizer, None

# Callback for training
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
