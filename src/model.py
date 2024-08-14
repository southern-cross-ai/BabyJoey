import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import BabyJoeyConfig

class Embeddings(nn.Module):
    def __init__(self, config: BabyJoeyConfig):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.n_embd)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
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
        # Transpose for MultiheadAttention (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.ln1(x)

        # Transpose back to (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)

        return x

class BabyJoey(nn.Module):
    def __init__(self, config: BabyJoeyConfig):
        super(BabyJoey, self).__init__()
        
        # Embeddings
        self.embeddings = Embeddings(config)
        
        # Encoder Blocks (based on n_layer)
        self.encoder_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Decoder Blocks (based on n_layer_decoder)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer_decoder)])

        # Output layer
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, encoder_output=None):
        # Get embeddings
        x = self.embeddings(x)

        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        if encoder_output is not None:  # If using a decoder
            for block in self.decoder_blocks:
                x = block(x + encoder_output)  # Combine with encoder output

        # Layer norm and output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
