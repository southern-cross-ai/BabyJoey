from typing import Literal
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from networkx import from_nested_tuple
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# Embedding layer configs for token embedding and positional embedding
@dataclass(frozen=True)
class _EmbeddingConfig:
    vocab_size: int = 50257
    sequence_length: int = 512
    n_embd: int = 512


# Transformer layer configs for transformer blocks and their stacked structure
@dataclass(frozen=True)
class _TransformerConfig:
    n_head: int = 8
    n_layer_decoder: int = 1
    n_embd: int = _EmbeddingConfig.n_embd
    attn_mask = None
    key_padding_mask = None


# Model configs for embedding layer and transformer layer 
@dataclass(frozen=True)
class ModelConfig:
    embedding: _EmbeddingConfig = _EmbeddingConfig()
    transformer: _TransformerConfig = _TransformerConfig()
    

# DataLoaderConfig for dataloader settings
@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 2


# OptimizationConfig for optimization hyperparameters
@dataclass(frozen=True)
class _OptimizerConfig:
    # TODO: Is there a better way for users to organise their optimiser configs?
    # optimizer: Optimizer = torch.optim.adamw.AdamW
    # gradient_accumulation_steps: int = 1  # number of batches to accumulate gradients for back propagation
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3


@dataclass(frozen=True)
class _SchedulerConfig:
    # TODO: Is there a better way for users to organise their lr_scheduler configs?
    # scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR
    # step_lr_interval: Literal["step", "epoch"] = "epoch"  # time to step scheduler
    step_size: int = 1
    gamma: float = 0.9


@dataclass(frozen=True)
class OptimisationConfig:
    optimizer: _OptimizerConfig = _OptimizerConfig()
    scheduler: _SchedulerConfig = _SchedulerConfig()


# DatasetConfig for dataset settings
@dataclass(frozen=True)
class DataConfig:
    data_path: str = "SouthernCrossAI/Tweets_cricket"
    column_name: str = "tweet"
    sequence_length: int = 512
    train_file: str = "train_data.pt"
    valid_file: str = "valid_data.pt"
    split_ratio: float = 0.2


# Config for additional training parameters
@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# BabyJoeyConfig for overall model configuration
@dataclass(frozen=True)
class BabyJoeyConfig:    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimization: OptimisationConfig = field(default_factory=OptimisationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Register the configuration in Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="baby_joey_config", node=BabyJoeyConfig)
