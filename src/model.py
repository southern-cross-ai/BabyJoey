import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

if __name__ == "__main__":
    from config import ModelConfig
else:
  from src.config import ModelConfig

# Embedding Layer with padding handling
class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
            padding_idx=config.padding_idx
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        assert seq_len <= self.position_embedding.num_embeddings, (
            f"Sequence length {seq_len} exceeds maximum sequence length {self.position_embedding.num_embeddings}"
        )
        tok_embed_mat = self.token_embedding(x)  # [batch_size, seq_len, n_embd]
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed_mat = self.position_embedding(pos)  # [1, seq_len, n_embd]
        x = tok_embed_mat + pos_embed_mat  # Broadcasting over batch_size
        return self.dropout(x)  # [batch_size, seq_len, n_embd]

# Multi-Head Attention Layer with padding handling
class MultiheadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Create a boolean causal mask
        causal_mask = torch.triu(
            torch.ones((config.max_seq_len, config.max_seq_len), dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)  # [max_seq_len, max_seq_len]

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]  # [seq_len, seq_len]

        # Ensure masks are boolean tensors
        assert causal_mask.dtype == torch.bool, "causal_mask must be a boolean tensor"
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool, "key_padding_mask must be a boolean tensor"

        attn_output, _ = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )
        return self.dropout(attn_output)  # [batch_size, seq_len, n_embd]

# FeedForward Layer
class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [batch_size, seq_len, n_embd]

# Transformer Block with padding handling
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        attn_output = self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + attn_output  # Residual connection
        x = self.dropout(x)
        ff_output = self.ff(self.ln2(x))
        x = x + self.dropout(ff_output)  # Residual connection
        return x

# BabyJoey Model with padding handling
class BabyJoeyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config  # Store config for use in forward method
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len]
        key_padding_mask = (x == self.config.padding_idx)  # [batch_size, seq_len]
        x = self.embeddings(x)  # [batch_size, seq_len, n_embd]
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)  # [batch_size, seq_len, n_embd]
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        return logits

if __name__ == '__main__':
    print('------------- Testing -----------------')
    
    def test_baby_joey():
        # Create a default model configuration
        config = ModelConfig()

        # Create a model instance
        model = BabyJoeyModel(config)

        # Generate random test data
        batch_size = 4
        seq_len = 16  # Use a smaller sequence length for testing
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
        # Add padding tokens for testing
        input_ids[:, -2:] = config.padding_idx  # Pad the last two tokens

        # Pass the data through the model
        logits = model(input_ids)

        # Verify output shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size), (
            f"Expected output shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
        )

        return f"Test passed! Output shape: {logits.shape}"
    
    result = test_baby_joey()
    print(result)
    print("---------- Testing Complete ---------------")
