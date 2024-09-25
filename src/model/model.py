r"""BabyJoey Model Architecture

This module contains all the structures for building a `BabyJoeyModel`:
    - `Embeddings`: Token Embedding Matrix
                    Positional Embedding Matrix
                    Embedding Matrix
    - `TransformerBlock`: Layer Normalisation for Attention, 
                          Multi-Head Attention, 
                          Layer Normalisation for MLP,
                          Two-Layer FC MLP with ReLU
                          
`BabyJoeyModel` will run through the following steps:
1. Pass token indices through `Embeddings` to get embedding matrix
2. Pass embedding matrices through a sequence of `TransformerBlock`s to get MLP residual matrix
3. Pass last MLP residual matrix to linear norm layer
4. Pass the linear norm output to linear layer to caluculate logits
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embeddings(nn.Module):
    """Embedding layer for token and positional embeddings"""

    def __init__(self, vocab_size: int, n_embd: int, sequence_length: int) -> None:
        r"""Initialise token and positional embedding matrices.

        Args:
            vocab_size (int): total number of unique tokens
            n_embd (int): length of embedding vector, a.k.a. channel size (C) or dimensionality 
                          of the embedding space
            sequence_length (int): maximum number of tokens in input or output sequence that the 
                                   model can processes or generates, a.k.a. time steps (T)
        """
        
        super(Embeddings, self).__init__()
        # token embed matrix with shape (vocab_size, n_embd)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # pos embed matrix with shape (sequence_length, n_embd)
        self.position_embedding = nn.Embedding(sequence_length, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass through embedding layer and return an embedding matrix for inputs.

        Args:
            x (Tensor): batch of token indices, shape (batch_size, sequence_length).
                        Each entry is the corresponding index of a token in the vocabulary.

        Returns:
            Tensor: embedding matrix for input batch of token indices with shape 
                    (batch_size, sequence_length, n_embd).
        """
        
        # tok embed matrix with shape (batch_size, sequence_length, n_embd)
        tok_embed_mat = self.token_embedding(x)
        
        # ------ old version ------
        # (bach_size, sequence_length)
        # pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        # (batch_size, sequence_length, n_embd)
        # pos_embed_mat = self.position_embedding(pos)
        # (batch_size, sequence_length, n_embd) + (batch_size, sequence_length, n_embd)
        # embed_mat = tok_embed_mat + pos_embed_mat
        
        # ------ new version with dimension alignment handled by torch to save memory ------
        # index tensor with shape (sequence_length,)
        pos = torch.arange(0, x.size(1), device=x.device)
        # pos embed matrix with shape (sequence_length, n_embd)
        pos_embed_mat = self.position_embedding(pos)
        # add two embeddings element-wisely, shape (batch_size, sequence_length, n_embd) + (sequence_length, n_embd) -> (batch_size, sequence_length, n_embd)
        embed_mat = tok_embed_mat + pos_embed_mat
        
        return embed_mat


class TransformerBlock(nn.Module):
    r"""Transformer block with attention and MLP"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        r"""Initialise basic components for a transformer block.

        Args:
            n_embd (int): length of embedding vector, a.k.a. channel size (C) or 
                          dimensionality of the embedding space
            n_head (int): number of parallel attention heads
        """
        
        super(TransformerBlock, self).__init__()
        # 1. layer norm applied to last block's output (embd mat for the first block)
        # input and output both have shape (batch_size, sequence_length, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        # 2. multi-head attention with n_head heads, projecting to n_embd dimensions
        self.attn = nn.MultiheadAttention(n_embd, n_head)  # output shape (seq_len, batch_size, n_embd)
        # 3. layer norm applied to attention residual 
        # input and output have shape (batch_size, sequence_length, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # 4. two-layer FC MLP with hidden layer of size 4 * n_embd and ReLU
        self.mlp = nn.Sequential(
            # transform to a larger embedding representation to capture more information
            # (batch_size, sequence_length, n_embd) -> (batch_size, sequence_length, 4 * n_embd)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # transform back to original space for future blocks
            # (batch_size, sequence_length, 4 * n_embd) -> (batch_size, sequence_length, n_embd)
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: Tensor, attn_mask: Tensor, key_padding_mask: Tensor) -> Tensor:
        r"""Forward pass through the transformer block (attention layer and MLP layer).

        Args:
            x (Tensor): output from former transformer block (embedding matrix for first block)
            attn_mask (Tensor, optional): # TODO
            key_padding_mask (Tensor, optional): # TODO

        Returns:
            Tensor: output (MLP residual) of transformer block
        """

        ############# Attention #############
        # 1. keep a copy for calculating attention residual
        # shape (batch_size, sequence_length, n_embd)
        x_copy = x       
        # 2. first layer norm
        # input and output shape (batch_size, sequence_length, n_embd)
        x = self.ln1(x)  
        # 3. transpose shape for MultiheadAttention
        # input shape (batch_size, sequence_length, n_embd) -> 
        # output shape (sequence_length, batch_size, n_embd)
        x = x.transpose(0, 1)
        # 4. attention output
        # output shape (sequence_length, batch_size, n_embd)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # 5. transpose shape back for calculating attention residual
        # input shape (sequence_length, batch_size, n_embd) ->
        # output shape (batch_size, sequence_length, n_embd)
        attn_output = attn_output.transpose(0, 1)
        # 6. calculate attention residual
        # input and output shape (batch_size, sequence_length, n_embd)
        x = x_copy + attn_output
        ############# MLP #############
        # 1. keep a copy for calculating MLP residual
        # shape (batch_size, sequence_length, n_embd)
        x_copy = x
        # 2. second layer norm
        # input and output shape (batch_size, sequence_length, n_embd)
        x = self.ln2(x)
        # 3. MLP output 
        # input and output shape (batch_size, sequence_length, n_embd)
        mlp_output = self.mlp(x)
        # 4. calculate MLP residual
        # input and output shape (batch_size, sequence_length, n_embd)
        x = x_copy + mlp_output

        return x


class BabyJoeyModel(nn.Module):
    r"""Transformer-based language model that stacks multiple transformer blocks"""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer_decoder: int, sequence_length: int) -> None:
        r"""Initialise the basic components for BabyJoey.

        Args:
            vocab_size (int): total number of unique tokens
            n_embd (int): length of embedding vector, a.k.a. channel size (C) or dimensionality 
                          of the embedding space
            n_head (int): number of parallel attention heads
            n_layer_decoder (int): number of decoder blocks (TransformerBlock)
            sequence_length (int): maximum number of tokens in input or output sequence that the 
                                   model can processes or generates, a.k.a. time steps (T)
        """
        
        super(BabyJoeyModel, self).__init__()
        # 1. embeddings layer: summation of token and positional embedding matrices
        self.embeddings = Embeddings(vocab_size, n_embd, sequence_length)
        # 2. decoder blocks: sequence of n_layer_decoder decoder blocks
        self.decoder_blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer_decoder)])
        # 3. final layer norm for calculating logits
        self.ln_f = nn.LayerNorm(n_embd)
        # 4. linear layer to project final layer norm to logits
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None) -> Tensor:
        r"""Forward pass batch of token indices through whole BabyJoey.

        Args:
            x (Tensor): batch of token indices with shape (batch_size, sequence_length)
            attn_mask (Tensor, optional): # TODO
            key_padding_mask (Tensor, optional): # TODO

        Returns:
            Tensor: Corresponding batch of logits for each token in vocabulary with shape
                    (batch_size, sequence_length, vocab_size)
        """
        
        # 1. transform batch of token indices into batch of embedding matrices
        x = self.embeddings(x)
        # 2. pass through transformer blocks sequentially
        for block in self.decoder_blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # 3. apply final layer normalization for calculating logits
        x = self.ln_f(x)
        # 4. project batch of final layer norm to batch of logits
        logits = self.head(x)
        
        return logits
