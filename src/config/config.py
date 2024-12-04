from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import torch

# EmbeddingConfig for token embedding
@dataclass
class EmbeddingConfig:
    vocab_size: int = 50257
    max_seq_len: int = 512
    n_embd: int = 512

# TransformerConfig for transformer blocks
@dataclass
class TransformerConfig:
    n_head: int = 8
    n_layer_decoder: int = 1

# ModelConfig for model hyperparameters
@dataclass
class ModelConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    step_size: int = 1
    gamma: float = 0.9

# DataLoaderConfig for dataloader settings
@dataclass
class DataLoaderConfig:
    batch_size: int = 2

# OptimizationConfig for optimization hyperparameters
@dataclass
class OptimizationConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    step_size: int = 1
    gamma: float = 0.9

# DatasetConfig for dataset settings
@dataclass
class BabyJoeyDataConfig:
    data_path: str = "SouthernCrossAI/Tweets_cricket"
    max_seq_len: int = 512
    train_file: str = "train_data.pt"
    valid_file: str = "valid_data.pt"
    split_ratio: float = 0.2
    column_name: str = "tweet"
    sample_ratio: float = 0.1
    seed: int = 42

# Config for additional training parameters
@dataclass
class TrainingConfig:
    max_epochs: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# BabyJoeyConfig for overall model configuration
@dataclass
class BabyJoeyConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: BabyJoeyDataConfig = field(default_factory=BabyJoeyDataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# Register the configuration in Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="baby_joey_config", node=BabyJoeyConfig)
