from dataclasses import dataclass, field
# from hydra.core.config_store import ConfigStore
# import torch
# from torch.optim import AdamW

@dataclass
class TrainingConfig:
    vocab_size: int = 512
    padding_idx: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 32
    label_smoothing: float = 0.1
    optimizer_name: str = "AdamW"  # Store as string instead of class
    optimizer_params: dict = field(default_factory=lambda: {"weight_decay": 1e-2})

@dataclass
class DatasetConfig:
    batch_size: int = 2
    data_path: str = "SouthernCrossAI/Tweets_Australian_Cities"
    max_seq_len: int = 1024
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






# # Register the configuration in Hydra's ConfigStore
# cs = ConfigStore.instance()
# cs.store(name="training_config", node=TrainingConfig)