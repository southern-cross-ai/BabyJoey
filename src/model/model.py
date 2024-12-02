import torch
import torch.nn as nn
from torch import Tensor

# Embedding Layer
class Embeddings(nn.Module):
    """Embedding layer for token and positional embeddings"""

    def __init__(self, vocab_size: int, sequence_length: int, n_embd: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        tok_embed_mat = self.token_embedding(x)
        pos = torch.arange(0, x.size(1), device=x.device)
        pos_embed_mat = self.position_embedding(pos)
        embed_mat = tok_embed_mat + pos_embed_mat
        return embed_mat

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None) -> Tensor:
        x_copy = x       
        x = self.ln1(x)  
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        attn_output = attn_output.transpose(0, 1)
        x = x_copy + attn_output
        x_copy = x
        x = self.ln2(x)
        mlp_output = self.mlp(x)
        x = x_copy + mlp_output
        return x

# BabyJoey Model
class BabyJoeyModel(nn.Module):
    def __init__(self, vocab_size: int, sequence_length: int, n_embd: int, n_head: int, n_layer_decoder: int) -> None:
        super().__init__()
        self.embeddings = Embeddings(vocab_size, sequence_length, n_embd)
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head) for _ in range(n_layer_decoder)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None) -> Tensor:
        x = self.embeddings(x)
        for block in self.decoder_blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Instantiate the Model
vocab_size = 50257
sequence_length = 512
n_embd = 512
n_head = 8
n_layer_decoder = 1

model = BabyJoeyModel(
    vocab_size=vocab_size,
    sequence_length=sequence_length,
    n_embd=n_embd,
    n_head=n_head,
    n_layer_decoder=n_layer_decoder
)

# Example Input Tensor
input_tensor = torch.randint(0, vocab_size, (2, sequence_length))  # Batch size = 2
output = model(input_tensor)

print("Output shape:", output.shape)  # Expected: (2, sequence_length, vocab_size)
