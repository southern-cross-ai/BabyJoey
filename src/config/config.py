#--------------------------------------------------------#
#                   Hugging Face Setup                   #
#            BabyJoeyDataset - data/dataset.py           #
#--------------------------------------------------------#

# dataset from Hugging Face Hub in format 'username/dataset_name'
DATA = 'SouthernCrossAI/Tweets_cricket'
# dataset column that contains text to use, check on Hugging Face to find column name
COLUMN_NAME = 'tweet'


#--------------------------------------------------------#
#                   Local Dataset Setup                  #
#            BabyJoeyDataset - data/dataset.py           #
#--------------------------------------------------------#

# local path to load/store tokenised training set, default to 'dataset_name_train.py'
TRAIN_FILE = f'{DATA.split('/')[-1]}_train.pt'
# local path to load/store tokenised validation set, default to 'dataset_name_valid.py'
VALID_FILE = f'{DATA.split('/')[-1]}_valid.pt'
# split ratio for validation set
SPLIT_RATIO = 0.2


#--------------------------------------------------------#
#                    Dataloader Setup                    #
#         BabyJoeyDataLoader - data/dataloader.py        #
#--------------------------------------------------------#

# batch size for both training and validation
BATCH_SIZE = 2


#--------------------------------------------------------#
#                    Model Structure                     #
#            BabyJoeyModel - model/model.py              #
#--------------------------------------------------------#

# total number of unique tokens
VOCAB_SIZE = 50257
# maximum number of tokens in one sequence, or time steps (T)
SEQUENCE_LENGTH = 512
# dimension of embedding vectors, or channel size (C)
N_EMBD = 512
# number of multi-head attentions in each decoder block
N_HEAD = 8
# total number of decoder blocks
N_LAYER_DECODER = 1
# hidden layer size for two-layer FC MLP # TODO: current hidden layer size is hard-coded as 4 * N_EMBD in model.py
# N_HIDDEN_MLP = 4 * N_EMBD


#--------------------------------------------------------#
#               Optimisation Hyperparameters             #
#            BabyJoeyUnit - training/train.py            #
#--------------------------------------------------------#

# learning rate for AdamW optimizer # TODO: can users change to different optimisers?
LEARNING_RATE = 1e-5
# weight decay for AdamW optimizer
WEIGHT_DECAY = 1e-3
# period of learning rate decay for StepLR scheduler # TODO: can users change to different schedulers?
STEP_SIZE = 1
# multiplicative factor of learning rate decay for StepLR scheduler
GAMMA = 0.9


#--------------------------------------------------------#
#                   GPT-2 References                     #
#--------------------------------------------------------#

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
