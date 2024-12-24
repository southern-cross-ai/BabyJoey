from dataclasses import dataclass, field
import torch
from torch.optim import AdamW

@dataclass
class JoeyConfig:
    batch_size: int = 16
    epochs: int = 2  
    context_window: int = 512  
    token_vocab_size: int = 50254

# Create a single instance of JoeyConfig to share across components
joey_config = JoeyConfig()

@dataclass
class DataSetConfig:
    dataset_name: str = "SouthernCrossAI/Project_Gutenberg_Australia"
    data_dir: str = "data"
    context_window: int = joey_config.context_window  
    stride: int = 256
    batch_size: int = joey_config.batch_size  

@dataclass
class ModelConfig:
    token_vocab_size: int = joey_config.token_vocab_size 
    n_embd: int = 1280
    n_head: int = 16
    n_layers: int = 24
    context_window: int = joey_config.context_window  
    padding_idx: int = 512
    dropout_rate: float = 0.1
