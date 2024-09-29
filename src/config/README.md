# BabyJoey Configs

## TODO
- [ ] Parameterize hidden layer size in MLP instead of hard-coding it (`N_HIDDEN_MLP`).
- [ ] Allow users to configure different optimizers and schedulers.
- [ ] Validate that the dataset column `tweet` is used correctly during tokenization.
- [ ] Extend the configuration to support multiple datasets.
- [ ] Update the `ConfigStore` to allow runtime changes to model configuration.

## Overview

BabyJoey is a model based on GPT-like transformers with a focus on tweet data from the Hugging Face dataset. This repository contains the model structure, configurations, and training setup required to train the BabyJoey model.

## Model Structure

### Embedding Layer
- **vocab_size**: 50257
- **sequence_length**: 512 tokens
- **embedding_dimension (n_embd)**: 512 dimensions

### Transformer Configuration
- **n_head**: 8 multi-head attention mechanisms in each decoder block.
- **n_layer_decoder**: 1 decoder block.

### Model Hyperparameters
- **learning_rate**: `1e-5`
- **weight_decay**: `1e-3`
- **step_size**: `1` (for learning rate decay)
- **gamma**: `0.9` (for learning rate decay)

## Hugging Face Setup

The dataset is hosted on Hugging Face under `SouthernCrossAI/Tweets_cricket`. It uses the column `tweet` to access the text.

## Dataset Setup

- **Train file**: `train_data.pt`
- **Validation file**: `valid_data.pt`
- **Sample ratio**: `1` (use the entire dataset)
- **Validation split**: `0.2`

## Dataloader Setup

- **batch_size**: `2`

## Optimization Hyperparameters

- **Optimizer**: AdamW
  - **learning_rate**: `1e-5`
  - **weight_decay**: `1e-3`
  
- **Scheduler**: StepLR
  - **step_size**: `1`
  - **gamma**: `0.9`

## GPT-2 References

Here are the key parameters of various GPT-2 models for reference:

### GPT-2 Small
- Hidden Size: 768 dimensions
- Sequence Length: 1024 tokens
- Attention Heads: 12 heads
- Decoder Layers: 12 layers
- Total Parameters: 117M

### GPT-2 Medium
- Hidden Size: 1024 dimensions
- Sequence Length: 1024 tokens
- Attention Heads: 16 heads
- Decoder Layers: 24 layers
- Total Parameters: 345M

### GPT-2 Large
- Hidden Size: 1280 dimensions
- Sequence Length: 1024 tokens
- Attention Heads: 20 heads
- Decoder Layers: 36 layers
- Total Parameters: 762M

### GPT-2 Extra-Large
- Hidden Size: 1600 dimensions
- Sequence Length: 1024 tokens
- Attention Heads: 25 heads
- Decoder Layers: 48 layers
- Total Parameters: 1.5B
