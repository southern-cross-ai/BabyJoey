# Dataset Settings
TRAIN_FILE = 'training_dataset.pt'    # file path for training set
VALID_FILE = 'validation_dataset.pt'  # file path for validation set
DATA = "SouthernCrossAI/Tweets_cricket"  # format: hf_namespace/dataset_name
BATCH_SIZE = 32  # batch size for BabyJoeyDataLoader in dataloader.py

# Model Configurations
VOCAB_SIZE = 50257      # Unique Tokens
SEQUENCE_LENGTH = 512   # Sequence Length (T)
N_EMBD = 512            # Hidden Size (C)
N_HEAD = 8              # Attention Heads
N_LAYER_DECODER = 1     # Decoder Layers
# TODO: current Feed-Forward Network Size is hard-encoded as 4 * N_EMBD in model.py

# SGD Optimiser Hyperparameters
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
STEP_SIZE = 1
GAMMA = 0.9

"""
For the reference:
    
    GPT-2 Small
	- Hidden Size (C): 768 dimensions
	- Sequence Length (T): 1024 tokens
    - Attention Heads: 12 heads
    - Decoder Layers: 12 layers
	- Feed-Forward Network Size (4C): 3072 neurons
	- Total Parameters: 117 M
    
    GPT-2 Medium
	- Hidden Size (C): 1024 dimensions
	- Sequence Length (T): 1024 tokens
    - Attention Heads: 16 heads
    - Decoder Layers: 24 layers
	- Feed-Forward Network Size (4C): 4096 neurons
	- Total Parameters: 345 M
    
    GPT-2 Large
	- Hidden Size (C): 1280 dimensions
	- Sequence Length (T): 1024 tokens
    - Attention Heads: 20 heads
    - Decoder Layers: 36 layers
	- Feed-Forward Network Size (4C): 5120 neurons
	- Total Parameters: 762 M

    GPT-2 Extra-Large
    - Hidden Size (C): 1600 dimensions
	- Sequence Length (T): 1024 tokens
    - Attention Heads: 25 heads
    - Decoder Layers: 48 layers
	- Feed-forward Network Size (4C): 6400 neurons
	- Total Parameters: 1.5 B

"""
