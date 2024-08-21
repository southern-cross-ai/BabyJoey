from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    n_layer_decoder: int = 1
    max_position_embeddings: int = 512

@dataclass
class DataConfig:
    dataset_name: str
    batch_size: int
    shuffle: bool = True
    num_workers: int = 4

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer_type: str = "Adam"

@dataclass
class TrainingConfig:
    epochs: int
    log_interval: int = 10
    save_model: bool = True
    save_path: str = "models/babyjoey.pt"
    use_scheduler: bool = False

@dataclass
class BabyJoeyConfig:
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

def load_config(config_path: str = 'src/config.yaml') -> BabyJoeyConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    model = ModelConfig(**config_dict['model'])
    data = DataConfig(**config_dict['data'])
    optimizer = OptimizerConfig(**config_dict['optimizer'])
    training = TrainingConfig(**config_dict['training'])

    return BabyJoeyConfig(
        model=model,
        data=data,
        optimizer=optimizer,
        training=training
    )

# Optionally, automatically load the config when this module is imported
config = load_config()
