import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.fit import fit

# File paths to save/load the datasets
train_file = 'training_dataset.pt'
valid_file = 'validation_dataset.pt'

# Variables
data = "SouthernCrossAI/Project_Gutenberg_Australia"
sequence_length = 512
batch_size = 16

# Model Configuration
vocab_size = 50257
n_embd = 512
n_head = 8
n_layer_decoder = 1

# Load the dataset
ds = load_dataset(data)

# Split the dataset into training and validation sets
dataset = ds['train'].train_test_split(test_size=0.2)
print(f"test = {len(dataset['test'])} and train = {len(dataset['train'])}")

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(dataset):
    """
    Tokenizes the input dataset.

    Args:
        dataset: The dataset containing text data to be tokenized.

    Returns:
        A dictionary containing tokenized input IDs and attention masks.
    """
    return tokenizer(dataset['Paragraph'], 
                     truncation=True, 
                     padding='max_length', 
                     max_length=sequence_length,
                     return_attention_mask=True)

# Check if transformed datasets already exist
if os.path.exists(train_file) and os.path.exists(valid_file):
    # Load the transformed datasets
    training_dataset = torch.load(train_file)
    validation_dataset = torch.load(valid_file)
    print("Loaded existing transformed datasets.")
else:
    # Create and save the transformed datasets
    training_dataset = dataset['train'].map(tokenize_function, batched=True)
    validation_dataset = dataset['test'].map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    torch.save(training_dataset, train_file)
    torch.save(validation_dataset, valid_file)
    print("Transformed datasets created and saved.")

# Create DataLoaders for training and validation
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

class Embeddings(nn.Module):
    """
    Embeddings module to handle token and positional embeddings.
    """
    def __init__(self):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x):
        """
        Forward pass through the embeddings layer.

        Args:
            x: Input tensor containing token IDs.

        Returns:
            The sum of token and position embeddings.
        """
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
        return tokens + positions

class TransformerBlock(nn.Module):
    """
    Transformer block that includes attention and feed-forward network.
    """
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
        """
        Forward pass for the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            key_padding_mask: Boolean mask for attention.

        Returns:
            Output tensor after processing through attention and MLP layers.
        """
        x = x.transpose(0, 1)  # Transpose for MultiheadAttention
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_output
        x = self.ln1(x)

        x = x.transpose(0, 1)  # Transpose back
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)

        return x

class BabyJoey(nn.Module):
    """
    A simplified version of the GPT model named BabyJoey.
    """
    def __init__(self):
        super(BabyJoey, self).__init__()
        self.embeddings = Embeddings()
        self.decoder_blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layer_decoder)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attn_mask=None):
        """
        Forward pass through the BabyJoey model.

        Args:
            x: Input tensor containing token IDs.
            attn_mask: Attention mask for padding tokens.

        Returns:
            Logits for the next token prediction.
        """
        x = self.embeddings(x)
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=attn_mask)
        x = self.ln_f(x)
        return self.head(x)

# Initialize the BabyJoey model
model = BabyJoey()

class BabyJoeyUnit(AutoUnit):
    """
    AutoUnit class for training BabyJoey using TorchTNT.
    """
    def __init__(self, module, device=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, state, data):
        """
        Computes the loss for a batch of data.

        Args:
            state: The current state of training.
            data: A batch of input data.

        Returns:
            The computed loss and logits.
        """
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()  # Boolean mask for padding
        logits = self.module(input_ids, attn_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module):
        """
        Configures the optimizer and learning rate scheduler.

        Args:
            module: The model to optimize.

        Returns:
            The optimizer and learning rate scheduler (if any).
        """
        optimizer = optim.AdamW(module.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer, None

# Initialize device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BabyJoey().to(device)

# Define the custom AutoUnit with the correct device object
baby_joey_unit = BabyJoeyUnit(module=model, device=device)

# Train the model using TorchTNT's fit function
fit(baby_joey_unit,
    train_dataloader=training_dataloader,
    eval_dataloader=validation_dataloader,
    max_epochs=2,
)
