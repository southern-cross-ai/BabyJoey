import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, n_embd, sequence_length):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        positions = self.position_embedding(positions)
        return tokens + positions

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        self.n_head = n_head  # Store n_head as an instance variable
        self.attn = nn.MultiheadAttention(n_embd, self.n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, key_padding_mask=None):
        x = x.transpose(0, 1)
        seq_len = x.size(0)
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        # Use self.n_head here
        attn_mask = attn_mask.unsqueeze(0).expand(x.size(1) * self.n_head, -1, -1)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_output
        x = self.ln1(x)
        x = x.transpose(0, 1)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        return x

class BabyJoeyModel(nn.Module):  # Renamed from BabyJoey
    def __init__(self, vocab_size, n_embd, n_head, n_layer_decoder):
        super(BabyJoeyModel, self).__init__()
        self.embeddings = Embeddings(vocab_size, n_embd, sequence_length=512)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer_decoder)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.embeddings(x)
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
