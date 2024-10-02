import torch
import torch.nn as nn
from torch import Tensor

from src import config
from src.config.config import ModelConfig, _EmbeddingConfig, _TransformerConfig


class EmbeddingLayer(nn.Module):
    """Embedding layer for token and positional embeddings"""

    def __init__(self, config: _EmbeddingConfig) -> None:
        r"""Initialise token and positional embedding matrices based on configs in ModelConfig.

        Args:
            config (ModelConfig): model configurations
                - vocab_size (int): total number of unique tokens
                - n_embd (int): length of embedding vector, a.k.a. channel size (C) or dimensionality 
                          of the embedding space
                - sequence_length (int): maximum number of tokens in input or output sequence that the 
                                   model can processes or generates, a.k.a. time steps (T)
        """

        super(EmbeddingLayer, self).__init__()
        # token embed matrix with shape (vocab_size, n_embd)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # pos embed matrix with shape (sequence_length, n_embd)
        self.position_embedding = nn.Embedding(config.sequence_length, config.n_embd)

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
        # index tensor with shape (sequence_length,)
        pos = torch.arange(0, x.size(1), device=x.device)
        # pos embed matrix with shape (sequence_length, n_embd)
        pos_embed_mat = self.position_embedding(pos)
        # add two embeddings element-wisely, shape (batch_size, sequence_length, n_embd) + (sequence_length, n_embd) -> (batch_size, sequence_length, n_embd)
        embed_mat = tok_embed_mat + pos_embed_mat

        return embed_mat


class _TransformerBlock(nn.Module):
    def __init__(self, config: _TransformerConfig):
        super(_TransformerBlock, self).__init__()
        # 1. layer norm applied to last block's output (embd mat for the first block)
        # input and output both have shape (batch_size, sequence_length, n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        # 2. multi-head attention with n_head heads, projecting to n_embd dimensions
        # output shape (seq_len, batch_size, n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head)
        # 3. layer norm applied to attention residual
        # input and output have shape (batch_size, sequence_length, n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # 4. two-layer FC MLP with hidden layer of size 4 * n_embd and ReLU
        self.mlp = nn.Sequential(
            # transform to a larger embedding representation to capture more information
            # (batch_size, sequence_length, n_embd) -> (batch_size, sequence_length, 4 * n_embd)
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            # transform back to original space for future blocks
            # (batch_size, sequence_length, 4 * n_embd) -> (batch_size, sequence_length, n_embd)
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x, attn_mask, key_padding_mask):
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
        

class TransformerLayer(nn.Module):
    def __init__(self, config: _TransformerConfig):
        super(TransformerLayer, self).__init__()
        self.transformer_blocks = [
            _TransformerBlock(config) for _ in range(config.n_layer_decoder)
        ]
    
    def forward(self, x, attn_mask, key_padding_mask):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask, key_padding_mask)
            
        return x


class BabyJoeyModel(nn.Module):
    r"""Transformer-based language model that stacks multiple transformer blocks"""

    def __init__(self, config: ModelConfig) -> None:
        r"""Initialise the basic components for BabyJoey.

        Args:
            config (Config): Configuration for BabyJoey.
                - embedding (EmbeddingConfig): Configuration for the embedding layer.
                - transformer (TransformerConfig): Configuration for the transformer blocks.
                - model (ModelConfig): Model-specific parameters such as learning rate.
        """

        super(BabyJoeyModel, self).__init__()
        
        print(config)
        
        # 1. embeddings layer: summation of token and positional embedding matrices
        self.embedding_layer = EmbeddingLayer(config.embedding)
        # 2. decoder blocks: sequence of n_layer_decoder transformer blocks
        self.transformer_layer = TransformerLayer(config.transformer)
        # 3. final layer norm for calculating logits
        self.norm_layer = nn.LayerNorm(config.embedding.n_embd)
        # 4. linear layer to project final layer norm to logits
        self.logit_layer = nn.Linear(config.embedding.n_embd, config.embedding.vocab_size, bias=False)

    def forward(self, x, attn_mask, key_padding_mask):
        r"""Forward pass batch of token indices through whole BabyJoey.

        Args:
            x (Tensor): batch of token indices with shape (batch_size, sequence_length)

        Returns:
            Tensor: Corresponding batch of logits for each token in vocabulary with shape
                    (batch_size, sequence_length, vocab_size)
        """

        # 1. transform batch of token indices into batch of embedding matrices
        x = self.embedding_layer(x)
        # 2. pass through transformer blocks sequentially
        x = self.transformer_layer(x, attn_mask, key_padding_mask)
        # 3. apply final layer normalization for calculating logits
        x = self.norm_layer(x)
        # 4. project batch of final layer norm to batch of logits
        logits = self.logit_layer(x)

        return logits
