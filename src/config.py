import yaml
from dataclasses import dataclass

@dataclass
class BabyJoeyConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    max_position_embeddings: int = 512
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10

def load_config(config_path: str) -> BabyJoeyConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return BabyJoeyConfig(**config_dict)

# Automatically load the config when this module is imported
config = load_config('config.yaml')
