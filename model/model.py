import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(12)])
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return self.output_layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config['hidden_size'])
        self.attention = nn.MultiheadAttention(config['hidden_size'], config['num_heads'])
        self.layer_norm2 = nn.LayerNorm(config['hidden_size'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['hidden_size'], config['intermediate_size']),
            nn.ReLU(),
            nn.Linear(config['intermediate_size'], config['hidden_size'])
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.layer_norm2(x + ff_output)
