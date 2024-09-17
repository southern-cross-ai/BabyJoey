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
        ############# 1. Attention #############
        # Keep the input for the later Attention Residual
        x_copy = x.clone()
        # 1.1 Layer Normalisation
        x = self.ln1(x)
        # 1.2 Self-Attention
        # Reshape for Multi-Head Self-Attention: [batch_size, seq_len, n_embd] -> [seq_len, batch_size, n_embd]
        x = x.transpose(0, 1)
        # Calculate the Attention Mask
        seq_len = x.size(0)
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        # Use self.n_head here
        attn_mask = attn_mask.unsqueeze(0).expand(x.size(1) * self.n_head, -1, -1)
        # Get the Attention Output
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Reshape to the shape of the input: [seq_len, batch_size, n_embd] -> [batch_size, seq_len, n_embd]
        attn_output = attn_output.transpose(0, 1)
        # 1.3 Attention Residual
        x = x_copy + attn_output
        ############# 2. MLP #############
        # Keep the Attention Residual for the later MLP Residual
        x_copy = x.clone()
        # 2.1 Layer Normalisation
        x = self.ln2(x)
        # 2.2 Two-Layer Fully Connected MLP
        mlp_output = self.mlp(x)
        # 2.3 MLP Residual
        x = x_copy + mlp_output
        return x

class BabyJoeyModel(nn.Module):  
    def __init__(self, vocab_size, n_embd, n_head, n_layer_decoder, sequence_length):
        super(BabyJoeyModel, self).__init__()
        self.embeddings = Embeddings(vocab_size, n_embd, sequence_length)
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
