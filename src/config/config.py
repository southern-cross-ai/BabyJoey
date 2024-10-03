from dataclasses import dataclass, field

import torch
from hydra.core.config_store import ConfigStore


# TODO: Is there a better way instead of using and integrating "_PrivateConfig"?
# TODO: To what extent, users can customise configs for BabyJoey? E.g., customisable dataset/optimiser/scheduler?


#####################     MODEL     #####################
@dataclass(frozen=True)
class _EmbeddingConfig:
    r"""Embedding layer configs for token embedding and positional embedding
    
    Args:
        vocab_size (int): Total number of unique tokens
        sequence_length (int): Maximum number of tokens in input/output sequences
                               that can be processed/generated, also known as time 
                               steps (T)
        n_embd (int): Length of embedding vectors, also known as channel size (C) 
                      or dimensionality of the embedding space
    """
    
    vocab_size: int = 50257
    sequence_length: int = 512
    n_embd: int = 512


@dataclass(frozen=True)
class _TransformerConfig:
    r"""Transformer configs for transformer blocks and their stacked structure
    
    Args:
        n_head (int): Number of attention heads in each transformer block
        n_layer_decoder (int): Total number of transformer blocks stacked together
        n_embd (int): Length of embedding vectors, also known as channel size (C) 
                      or dimensionality of the embedding space
    """
    
    n_head: int = 8
    n_layer_decoder: int = 1
    n_embd: int = _EmbeddingConfig.n_embd  # TODO: Same config in both classes. How to better manage it?


@dataclass(frozen=True)
class ModelConfig:
    r"""Model configs for embedding layer and transformer layer
    
    Args:
        embedding (_EmbeddingConfig): Embedding layer configs defined in class 
                                      _EmbeddingConfig
        transformer (_TransformerConfig): Transformer layer configs defined in 
                                          class _TransformerConfig
    """
    
    # TODO: Is there a better way to manage/integrate separate configs?
    # TODO: Is it reasonable to have "_PrivateConfig" in config.py?
    embedding: _EmbeddingConfig = _EmbeddingConfig()
    transformer: _TransformerConfig = _TransformerConfig()
#####################     MODEL     #####################


#####################     DATA     #####################
@dataclass(frozen=True)
class DataConfig:
    r"""Dataset configs for downloading/storing/loading data from Hugging Face
    
    Args:
        data_path (str): Dataset from Hugging Face Hub in format "namespace/dataset"
        column_name (str): Column that contains text data in "namespace/dataset"
        sequence_length (int): Maximum number of tokens in input/output sequences
                               that can be processed/generated, also known as time 
                               steps (T)
        train_file (str): Path for storing/loading training dataset
        valid_file (str): Path for storing/loading validation dataset
        split_ratio (float): split_ratio of whole dataset will be used to create 
                             validation dataset
    """
    
    data_path: str = "SouthernCrossAI/Tweets_cricket"
    column_name: str = "tweet"
    sequence_length: int = _EmbeddingConfig.sequence_length
    train_file: str = f"{data_path.split('/')[-1]}_train_data.pt"
    valid_file: str = f"{data_path.split('/')[-1]}_val_data.pt"
    split_ratio: float = 0.2
#####################     DATA     #####################


#####################     DATALOADER     #####################
@dataclass(frozen=True)
class DataLoaderConfig:
    r"""Dataloader configs for feeding data in training/validation
    
    Args:
        batch_size (int): Number of batches of data for training/validation
    """
    batch_size: int = 2
#####################     DATALOADER     #####################


#####################     OPTIMIZATION     #####################
@dataclass(frozen=True)
class _OptimizerConfig:
    r"""Optimizer configs for all optimiser hyperparameters
    
    Args:
        TODO
    """
    # TODO: Allow users to customise their own optimiser instead of hard-coded AdamW
    # TODO: Different optimisers have different arguments. How to better design dataclass to solve this issue?
    # optimizer: Optimizer = torch.optim.adamw.AdamW
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3


@dataclass(frozen=True)
class _SchedulerConfig:
    r""" Scheduler configs for all learning rate scheduler hyperparameters
    
    Args:
        TODO
    """
    # TODO: Allow users to customise their own scheduler instead of hard-coded StepLR
    # TODO: Different optimisers have different arguments. How to better design dataclass to solve this issue?
    step_size: int = 1
    gamma: float = 0.9


@dataclass(frozen=True)
class OptimisationConfig:
    # TODO: Is there a better way to manage/integrate separate configs?
    # TODO: Is it reasonable to have "_PrivateConfig" in config.py?
    optimizer: _OptimizerConfig = _OptimizerConfig()
    scheduler: _SchedulerConfig = _SchedulerConfig()
# ------------------------------- OPTIMISATION -------------------------------


# -------------------------------TRAINING-------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    r"""Training config for additional training parameters
    
    Args:
        TODO
    """
    
    # TODO: What's the definition of "additional" params? When to put configs here?
    max_epochs: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------------TRAINING-------------------------------


# -------------------------------WANDB-------------------------------
@dataclass(frozen=True)
class WandBConfig:
    r"""WandB config for recording training/validation results
    
    Args:
        TODO
    """
    api_key: str = '6f03df3e00fa251e8f7c9345a5fa7198f3bfbc56'  # TODO: How to mange users' private keys?
    project_name: str = 'wandb-callback-test'
    dataset: str = DataConfig.data_path
    # TODO: Same question. How to handle duplicated configs when they need to be separated into several dataclass?
    num_embedding: int = _EmbeddingConfig.n_embd
    num_attention_head: int = _TransformerConfig.n_head
    num_decoder_layer: int = _TransformerConfig.n_layer_decoder
    max_epochs: int = TrainingConfig.max_epochs
    batch_size: int = DataLoaderConfig.batch_size
    learning_rate: float = _OptimizerConfig.learning_rate
    weight_decay: float = _OptimizerConfig.weight_decay
    step_size: int = _SchedulerConfig.step_size
    gamma: float = _SchedulerConfig.gamma
# -------------------------------WANDB-------------------------------


# -------------------------------BABYJOEY-------------------------------
@dataclass(frozen=True)
class BabyJoeyConfig:
    r"""BabyJoey config for overall model configuration
    
    Args:
        TODO
    """
    # TODO: It doesn't necessarily need to be exposed to users. 
    #       Once they filled in the above configs, it should be good to run main.py.
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimization: OptimisationConfig = field(default_factory=OptimisationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
# -------------------------------BABTJOEY-------------------------------


# Register the configuration in Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="baby_joey_config", node=BabyJoeyConfig)  # TODO: Can users customise to different name?
