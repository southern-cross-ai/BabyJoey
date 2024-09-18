import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """
    Embeddings layer that combines token and positional embeddings.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Dimensionality of embeddings.
        sequence_length (int): Maximum sequence length for positional embeddings.
    """
    def __init__(self, vocab_size, n_embd, sequence_length):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # (vocab_size, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)  # (sequence_length, n_embd)

    def forward(self, x):
        """
        Forward pass through the embeddings layer.

        Args:
            x (Tensor): Input tensor of token indices, shape (batch_size, seq_len).
        
        Returns:
            Tensor: Combined token and positional embeddings, shape (batch_size, seq_len, n_embd).
        """
        # Token embeddings: shape (batch_size, seq_len, n_embd)
        tokens = self.token_embedding(x)  
        # Positional embeddings: shape (1, seq_len), then expand to (batch_size, seq_len)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        # Positional embeddings: shape (batch_size, seq_len, n_embd)
        positions = self.position_embedding(positions)
        # Sum token and positional embeddings: shape (batch_size, seq_len, n_embd)
        return tokens + positions


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of multi-head self-attention, 
    layer normalization, and a position-wise feed-forward network (MLP).

    Args:
        n_embd (int): Dimensionality of embeddings.
        n_head (int): Number of attention heads.
    """
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        self.n_head = n_head
        # Multihead attention with n_head heads, projecting to n_embd dimensions
        self.attn = nn.MultiheadAttention(n_embd, self.n_head)  # (n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # LayerNorm applied to (batch_size, seq_len, n_embd)
        # MLP with intermediate layer size 4 * n_embd, followed by ReLU and output layer
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # (batch_size, seq_len, 4 * n_embd)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)  # (batch_size, seq_len, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)  # LayerNorm applied to (batch_size, seq_len, n_embd)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Forward pass through the transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd).
            attn_mask (Tensor, optional): Attention mask to prevent attention to future tokens (causal masking).
            key_padding_mask (Tensor, optional): Mask to avoid attending to padding tokens.

        Returns:
            Tensor: Output tensor after attention and MLP, shape (batch_size, seq_len, n_embd).
        """
        ############# 1. Attention #############
        # Input shape: (batch_size, seq_len, n_embd)
        x_copy = x  # For residual connection later
        x = self.ln1(x)  # Layer normalization, shape remains (batch_size, seq_len, n_embd)

        # Transpose for MultiheadAttention: shape becomes (seq_len, batch_size, n_embd)
        x = x.transpose(0, 1)
        # Attention output shape: (seq_len, batch_size, n_embd)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Transpose back to original shape: (batch_size, seq_len, n_embd)
        attn_output = attn_output.transpose(0, 1)
        # Add residual connection: shape remains (batch_size, seq_len, n_embd)
        x = x_copy + attn_output

        ############# 2. MLP #############
        x_copy = x.clone()  # Clone for residual connection
        x = self.ln2(x)  # Layer normalization, shape remains (batch_size, seq_len, n_embd)
        # MLP output shape: (batch_size, seq_len, n_embd)
        mlp_output = self.mlp(x)
        # Add residual connection: shape remains (batch_size, seq_len, n_embd)
        x = x_copy + mlp_output

        return x


class BabyJoeyModel(nn.Module):
    """
    A transformer-based language model that stacks multiple transformer blocks.

    Args:
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Dimensionality of embeddings.
        n_head (int): Number of attention heads.
        n_layer_decoder (int): Number of transformer blocks in the decoder.
        sequence_length (int): Maximum sequence length for positional embeddings.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer_decoder, sequence_length):
        super(BabyJoeyModel, self).__init__()
        # Embeddings layer: token and positional embeddings, output shape (batch_size, seq_len, n_embd)
        self.embeddings = Embeddings(vocab_size, n_embd, sequence_length)
        # Decoder blocks: List of TransformerBlock, each with output shape (batch_size, seq_len, n_embd)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer_decoder)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # Linear layer to project the output to vocabulary size: output shape (batch_size, seq_len, vocab_size)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Forward pass through the BabyJoey model.

        Args:
            x (Tensor): Input tensor of token indices, shape (batch_size, seq_len).
            attn_mask (Tensor, optional): Attention mask for causal masking.
            key_padding_mask (Tensor, optional): Mask to avoid attention to padding tokens.

        Returns:
            Tensor: Logits over the vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        # Embedding output shape: (batch_size, seq_len, n_embd)
        x = self.embeddings(x)
        # Pass through each transformer block
        for block in self.decoder_blocks:
            # Output shape remains (batch_size, seq_len, n_embd) after each block
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Apply final layer normalization, shape remains (batch_size, seq_len, n_embd)
        x = self.ln_f(x)
        # Project to vocabulary size: shape becomes (batch_size, seq_len, vocab_size)
        logits = self.head(x)
        return logits
