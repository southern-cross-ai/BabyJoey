import os
import torch
from torch.utils.data import DataLoader
from src.model import BabyJoeyModel
from src.dataset import DataSetCreate
from src.config import DataSetConfig, ModelConfig

# Set up configurations
dataset_config = DataSetConfig()
model_config = ModelConfig()

# Load datasets
dataset_factory = DataSetCreate(dataset_config)
train_dataset, val_dataset = dataset_factory()

# DataLoader with increased `num_workers` for faster data loading
train_loader = DataLoader(train_dataset, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize model, optimizer, and scaler

model = BabyJoeyModel(model_config).to(device)
