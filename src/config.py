from dataclasses import dataclass, field
import torch
from torch.optim import AdamW

@dataclass
class TrainingConfig:
    device: torch.device
    vocab_size: int = 512
    padding_idx: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 1
    label_smoothing: float = 0.1
    optimizer_name: str = "AdamW"  # Store as string instead of class
    optimizer_params: dict = field(default_factory=lambda: {"weight_decay": 1e-2})
    optimizer: tuple = (AdamW, {'weight_decay': 1e-2})


@dataclass
class DatasetConfig:
    batch_size: int = 2
    data_path: str = "SouthernCrossAI/Tweets_Australian_Cities"
    max_seq_len: int = 512
    split_ratio: float = 0.2
    sample_ratio: float = 1.0
    column_name: str = "tweet"
    tokenizer_name: str = "gpt2"
    output_dir: str = field(init=False)

    def __post_init__(self):
        # Extract dataset name from the data_path
        dataset_name = self.data_path.split("/")[-1]
        # Incorporate the dataset name into the output directory
        self.output_dir = f"./processed_data_{dataset_name}"

# Model configuration using dataclass
@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 512
    n_head: int = 1
    n_layers: int = 1
    max_seq_len: int = 512
    padding_idx: int = 50256
    dropout_rate: float = 0.1 