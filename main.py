from helper_scrips.dataloader import GetJoeyData
from src.model import BabyJoeyModel
from src.training import ModelTrainer
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from src.config import ModelConfig, TrainingConfig, DatasetConfig

dataset_instance = GetJoeyData(DatasetConfig())
training_dataset, validation_dataset = dataset_instance.create_dataloaders()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BabyJoeyModel(ModelConfig()).to(device)

train = TrainingConfig(device=device)

trainer = ModelTrainer(model, training_dataset, validation_dataset, train)
trainer.fit(num_epochs=2)

print("working to here")
