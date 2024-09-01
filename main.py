import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch.optim as optim

from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.train import train


# Set variables
data = "SouthernCrossAI/Project_Gutenberg_Australia"
sequence_length = 512
batch_size = 16

# Model configuration
vocab_size = 50257
n_embd = 512
n_head = 8
n_layer = 8
n_layer_decoder = 1


# Load the dataset
ds = load_dataset(data)

# Split the dataset into training and testing sets
dataset = ds['train'].train_test_split(test_size=0.2)
print(f"test = {len(dataset['test'])} and train = {len(dataset['train'])}")

# Initialize the tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize the dataset
def tokenize_function(dataset):
    return tokenizer(dataset['Paragraph'], 
                     truncation=True, 
                     padding='max_length', 
                     max_length=sequence_length,
                     return_attention_mask=True)

# Convert datasets to torch format and map tokenization function
training_dataset = dataset['train'].map(tokenize_function, batched=True)
validation_dataset = dataset['test'].map(tokenize_function, batched=True)
training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create data loaders
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Print shapes of a batch for debugging
for batch in validation_dataloader:
    print("Batch input_ids shape:", batch['input_ids'].shape)
    print("Batch attention_mask shape:", batch['attention_mask'].shape)
    break  # Exit after printing the size of the first batch


# Define Embeddings module
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
        x = tokens + positions
        return x


# Define Transformer block
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
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_output
        x = self.ln1(x)

        x = x.transpose(0, 1)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)

        return x


# Define BabyJoey model
class BabyJoey(nn.Module):
    def __init__(self):
        super(BabyJoey, self).__init__()
        self.embeddings = Embeddings()
        self.decoder_blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layer_decoder)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attn_mask=None):
        x = self.embeddings(x)
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# Define custom training unit using AutoUnit
class BabyJoeyUnit(AutoUnit):
    def __init__(self, module, device=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, state, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()
        logits = self.module(input_ids, attn_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module):
        optimizer = optim.AdamW(module.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer, None


# Main function to run training and evaluation
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BabyJoey().to(device)
    baby_joey_unit = BabyJoeyUnit(module=model, device=device)

    # Train the model
    train(baby_joey_unit, train_dataloader=training_dataloader, max_epochs=2)

    # Evaluate the model
    evaluate(baby_joey_unit, eval_dataloader=validation_dataloader)
