from src.data.data import BabyJoeyDataset, BabyJoeyDataLoader
from src.model.model import BabyJoeyModel
from dataclasses import dataclass
import torch
from torch.optim import AdamW
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layers: int
    max_seq_len: int
    padding_idx: int  # Index of the padding token
    dropout_rate: float = 0.1  # Default dropout rate

# Sample configuration
config = ModelConfig(
    vocab_size=50257,  # Example vocabulary size
    n_embd=768,
    n_head=12,
    n_layers=12,
    max_seq_len=1024,
    padding_idx=50256,  # Padding token index
    dropout_rate=0.1
)

dataset_instance = BabyJoeyDataset()
training_dataset, validation_dataset = dataset_instance.load_or_create_datasets()

data_loader_instance = BabyJoeyDataLoader(training_dataset, validation_dataset)
train_loader, val_loader = data_loader_instance.get_dataloaders()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BabyJoeyModel(config).to(device)
model.train()

# Use AdamW optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)  # Example learning rate

# Simple training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]

        # Forward pass
        logits = model(input_ids)  # [batch_size, seq_len, vocab_size]

        # Shift inputs and logits for causal language modeling
        shifted_logits = logits[:, :-1, :].contiguous()   # [batch_size, seq_len-1, vocab_size]
        shifted_input_ids = input_ids[:, 1:].contiguous()  # [batch_size, seq_len-1]

        # Compute loss with label smoothing and ignoring padding tokens
        loss = F.cross_entropy(
            shifted_logits.view(-1, config.vocab_size),
            shifted_input_ids.view(-1),
            ignore_index=config.padding_idx,
            label_smoothing=0.1
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            logits = model(input_ids)

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_input_ids = input_ids[:, 1:].contiguous()

            val_batch_loss = F.cross_entropy(
                shifted_logits.view(-1, config.vocab_size),
                shifted_input_ids.view(-1),
                ignore_index=config.padding_idx,
                label_smoothing=0.1
            )

            val_loss += val_batch_loss.item()
            val_steps += 1

    avg_val_loss = val_loss / max(val_steps, 1)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    model.train()

    # Save a checkpoint after each epoch
    checkpoint_path = f"baby_joey_checkpoint_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'config': config,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
