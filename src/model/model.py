import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

# Dataclass for model configuration
@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_embd: int
    n_head: int
    n_layers: int
    max_seq_len: int
    dropout_rate: float = 0.1  # Default dropout rate


# Embedding Layer
class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)  # [vocab_size, n_embd]
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)  # [max_seq_len, n_embd]
        self.dropout = nn.Dropout(config.dropout_rate)  # Dropout after embeddings

    def forward(self, x: Tensor) -> Tensor:
        tok_embed_mat = self.token_embedding(x)  # [batch, seq_size, n_embd]
        pos = torch.arange(0, x.size(1), device=x.device)  # [seq_size]
        pos_embed_mat = self.position_embedding(pos)  # [seq_size, n_embd]
        return self.dropout(tok_embed_mat + pos_embed_mat)  # [batch, seq_size, n_embd]


# Multi-Head Attention Layer
class MultiheadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        causal_mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.dropout(attn_output)



# FeedForward Layer
class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # [n_embd -> 4 * n_embd]
            nn.GELU(),                                    # GELU for smoother gradients
            nn.Dropout(config.dropout_rate),             # Dropout after activation
            nn.Linear(4 * config.n_embd, config.n_embd),  # [4 * n_embd -> n_embd]
            nn.Dropout(config.dropout_rate)              # Dropout after final linear
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [batch, seq_size, n_embd]


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)  # Residual dropout

    def forward(self, x: Tensor) -> Tensor:
        attn_output = self.attn(self.ln1(x))  # [batch, seq_size, n_embd]
        x = x + attn_output
        x = self.dropout(x)  # Residual connection with dropout
        ff_output = self.ff(self.ln2(x))  # [batch, seq_size, n_embd]
        x = x + self.dropout(ff_output)  # Residual connection with dropout
        return x


# BabyJoey Model
class BabyJoeyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)  # Enable bias for final layer

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)  # [batch, seq_size, n_embd]
        for block in self.blocks:
            x = block(x)  # [batch, seq_size, n_embd]
        x = self.ln_f(x)  # [batch, seq_size, n_embd]
        logits = self.head(x)  # [batch, seq_size, vocab_size]
        return logits
