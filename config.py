@dataclass

class BabyJoeyConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    max_position_embeddings: int = 512
