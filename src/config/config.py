##########################################################
#                   Hugging Face Setup                   #
##########################################################

# dataset from Hugging Face for BabyJoeyDataset (data/dataset.py)
DATA = "SouthernCrossAI/Tweets_cricket"
# column name from the dataset used as input for BabyJoeyDataset (data/dataset.py)
COLUMN_NAME = 'tweet'
# path to load/store tokenised training set for BabyJoeyDataset (data/dataset.py)
TRAIN_FILE = 'training_dataset.pt'
# path to load/store tokenised validation set for BabyJoeyDataset (data/dataset.py)
VALID_FILE = 'validation_dataset.pt'


##########################################################
#                    Dataset Settings                    #
##########################################################

# batch size for BabyJoeyDataLoader (data/dataloader.py)
BATCH_SIZE = 2
# split ratio for validation set for BabyJoeyDataset (data/dataset.py)
SPLIT_RATIO = 0.2


##########################################################
#                   BabyJoey Structure                   #
##########################################################

# number of unique tokens for BabyJoeyModel (model/model.py)
VOCAB_SIZE = 50257
# maximum number of tokens in one sequence for BabyJoeyModel (model/model.py), a.k.a. time steps (T)
SEQUENCE_LENGTH = 512
# dimensionality of embedding vectors for tokens for BabyJoeyModel (model/model.py), a.k.a. channel size (C) or hidden size
N_EMBD = 512
# number of attention heads for BabyJoeyModel (model/model.py)
N_HEAD = 8
# number of decoder blocks for BabyJoeyModel (model/model.py)
N_LAYER_DECODER = 1
# hidden layer size for two-layer FC MLP # TODO: current hidden layer size is hard-coded as 4 * N_EMBD in model.py
# N_HIDDEN_MLP = 4 * N_EMBD


##########################################################
#               Optimisation Hyperparameters             #
##########################################################

# learning rate of AdamW optimizer for BabyJoeyUnit (training/train.py) # TODO: can users change to different optimisers?
LEARNING_RATE = 1e-5
# weight decay of AdamW optimizer for BabyJoeyUnit (training/train.py)
WEIGHT_DECAY = 1e-3
# period of learning rate decay of StepLR scheduler for BabyJoeyUnit (training/train.py) # TODO: can users change to different schedulers?
STEP_SIZE = 1
# multiplicative factor of learning rate decay of StepLR scheduler for BabyJoeyUnit (training/train.py)
GAMMA = 0.9


##########################################################
#                   GPT-2 References                     #
##########################################################

"""
GPT-2 Small
- Hidden Size (C): 768 dimensions
- Sequence Length (T): 1024 tokens
- Attention Heads: 12 heads
- Decoder Layers: 12 layers
- Feed-Forward Network Size (4C): 3072 neurons
- Total Parameters: 117 M
"""
# N_EMBD = 768
# SEQUENCE_LENGTH = 1024
# N_HEAD = 12
# N_LAYER_DECODER = 12

""" 
GPT-2 Medium
- Hidden Size (C): 1024 dimensions
- Sequence Length (T): 1024 tokens
- Attention Heads: 16 heads
- Decoder Layers: 24 layers
- Feed-Forward Network Size (4C): 4096 neurons
- Total Parameters: 345 M
"""
# N_EMBD = 1024
# SEQUENCE_LENGTH = 1024
# N_HEAD = 16
# N_LAYER_DECODER = 24

"""
GPT-2 Large
- Hidden Size (C): 1280 dimensions
- Sequence Length (T): 1024 tokens
- Attention Heads: 20 heads
- Decoder Layers: 36 layers
- Feed-Forward Network Size (4C): 5120 neurons
- Total Parameters: 762 M
"""
# N_EMBD = 1280
# SEQUENCE_LENGTH = 1024
# N_HEAD = 20
# N_LAYER_DECODER = 36

"""
GPT-2 Extra-Large
- Hidden Size (C): 1600 dimensions
- Sequence Length (T): 1024 tokens
- Attention Heads: 25 heads
- Decoder Layers: 48 layers
- Feed-forward Network Size (4C): 6400 neurons
- Total Parameters: 1.5 B
"""
# N_EMBD = 1600
# SEQUENCE_LENGTH = 1024
# N_HEAD = 25
# N_LAYER_DECODER = 48
