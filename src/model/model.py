import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import BabyJoeyConfig

class Embeddings(nn.Module):
    def __init__(self, config: BabyJoeyConfig):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.n_embd)

    def forward(self, x):
        # Get token embeddings
        tokens = self.token_embedding(x)

        # Get positional embeddings
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)

        # Combine token and positional embeddings
        x = tokens + positions
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: BabyJoeyConfig):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.ln1(x)

        # Feed-forward
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)

        return x

class BabyJoey(nn.Module):
    def __init__(self, config: BabyJoeyConfig):
        super(BabyJoey, self).__init__()
        
        # Embeddings
        self.embeddings = Embeddings(config)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Output layer
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        # Get embeddings
        x = self.embeddings(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
