import os
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model import BabyJoeyModel
from src.dataset import DataSetFactory
from src.config import DataSetConfig, ModelConfig

# Initialize distributed processing
torch.distributed.init_process_group(backend="nccl")  # For multi-GPU
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
local_rank = rank % torch.cuda.device_count()
device = torch.device(f"cuda:{local_rank}")

# Set up configurations
dataset_config = DataSetConfig()
model_config = ModelConfig()

# Load datasets
dataset_factory = DataSetFactory(dataset_config)
train_dataset, val_dataset = dataset_factory()

# Use DistributedSampler for data loading
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler)

# Initialize model, optimizer
model = BabyJoeyModel(model_config).to(device)
model = DDP(model, device_ids=[local_rank])  # Wrap model in DDP
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Loss function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Tracking the best validation loss
best_val_loss = float("inf")

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_sampler.set_epoch(epoch)  # Ensure shuffling in DistributedSampler
    total_loss = 0.0

    for i, (inputs,) in enumerate(train_loader):
        inputs = inputs.to(device)

        # Forward pass
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            inputs.view(-1),
            label_smoothing=0.1
        )

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if (i + 1) % 10 == 0 and rank == 0:
            print(f"Epoch {epoch + 1}, Step {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

    if rank == 0:
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs.view(-1),
                label_smoothing=0.1
            )
            val_loss += loss.item()

    # Average validation loss across all processes
    val_loss = torch.tensor(val_loss / len(val_loader)).to(device)
    torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
    val_loss = val_loss.item() / world_size

    if rank == 0:
        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"model_epoch_{epoch + 1}.pt"
            torch.save(model.module.state_dict(), save_path)  # Save the DDP-wrapped model
            print(f"Model saved at {save_path} with validation loss {val_loss:.4f}")
