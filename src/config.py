from dataclasses import dataclass, field
import torch
from torch.optim import AdamW

class BabyConfig:
    # Core configuration shared across components
    batch_size: int = 16
    epoch = 2
    context_window: int = 512
    features = 512
    padding_idx: int = 50256
    token_vocab_size: int = 50254

@dataclass
class TrainingConfig:
    device: torch.device
    token_vocab_size: int = 50226
    padding_idx: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 1
    label_smoothing: float = 0.1
    optimizer_name: str = "AdamW"  # Store as string instead of class
    optimizer_params: dict = field(default_factory=lambda: {"weight_decay": 1e-2})
    optimizer: tuple = (AdamW, {'weight_decay': 1e-2})

@dataclass
class DataSetConfig:
    dataset_name: str
    data_dir: str = "data"
    chunk_size: int = 512
    stride: int = 256
    batch_size: int = 32

# Model configuration using dataclass
@dataclass
class ModelConfig:
    token_vocab_size: int = 50257
    n_embd: int = 512
    n_head: int = 1
    n_layers: int = 1
    context_window: int = 512
    padding_idx: int = 50256
    dropout_rate: float = 0.1 