import os
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast  # For mixed precision
from src.model import BabyJoeyModel
from src.dataset import DataSetFactory
from src.config import DataSetConfig, ModelConfig
import tiktoken

# Initialize distributed processing
torch.distributed.init_process_group(backend="nccl")  # For multi-GPU
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
local_rank = rank % torch.cuda.device_count()
device = torch.device(f"cuda:{local_rank}")

# Set up configurations
dataset_config = DataSetConfig()
model_config = ModelConfig()

# Check for existing model
save_path = "model_latest.pt"
if os.path.exists(save_path):
    print("Model already exists. Restarting training with the saved model.")

# Load datasets
dataset_factory = DataSetFactory(dataset_config)
train_dataset, val_dataset = dataset_factory()

# Use DistributedSampler for data loading
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

# DataLoader with increased `num_workers` for faster data loading
train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, num_workers=16, pin_memory=True)

# Initialize model, optimizer, and scaler
model = BabyJoeyModel(model_config).to(device)
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Loaded saved model.")
model = DDP(model, device_ids=[local_rank])  # Wrap model in DDP
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scaler = GradScaler()  # Mixed precision scaler

# Loss function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Tracking the best validation loss
best_train_loss = float("inf")
best_val_loss = float("inf")

class TextGenerator:
    def __init__(self, model_path, config):
        """
        Initialize the text generator.
        
        :param model_path: Path to the saved model file.
        :param config: ModelConfig instance for configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Load the model
        self.model = BabyJoeyModel(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(self, prompt, max_length):
        """
        Generate text based on a prompt.
        
        :param prompt: Starting text for the generator.
        :param max_length: Number of tokens to generate.
        :return: Generated text as a string.
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]

        # Ensure input does not exceed context window
        if input_tensor.size(1) > self.config.context_window:
            raise ValueError("Input sequence exceeds the model's context window.")

        # Generate tokens
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(input_tensor)  # [1, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

                # Sample from the distribution
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()

                # Append the generated token
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)

                # Stop generation if the end-of-sequence token is generated
                if next_token_id == self.config.padding_idx:
                    break

        # Decode tokens back into text
        generated_ids = input_tensor[0].tolist()
        return self.tokenizer.decode(generated_ids)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_sampler.set_epoch(epoch)  # Ensure shuffling in DistributedSampler
    total_loss = 0.0

    for i, (inputs,) in enumerate(train_loader):
        inputs = inputs.to(device)

        # Align inputs and targets for next-token prediction
        targets = inputs[:, 1:]  # Shift targets by one token
        inputs = inputs[:, :-1]  # Exclude the last token from inputs

        # Mixed precision forward pass
        with autocast():
            logits = model(inputs)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.contiguous().view(-1)
            )

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Print loss every 1000 steps
        if rank == 0 and (i + 1) % 1000 == 0:
            print(f"Step {i + 1}, Loss: {loss.item():.4f}")

    if rank == 0:
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1}: Training Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, in val_loader:
            inputs = inputs.to(device)

            # Align inputs and targets for next-token prediction in validation
            targets = inputs[:, 1:]  # Shift targets by one token
            inputs = inputs[:, :-1]  # Exclude the last token from inputs

            with autocast():  # Mixed precision validation
                logits = model(inputs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.contiguous().view(-1)
                )
                val_loss += loss.item()

    # Average validation loss across all processes
    val_loss = torch.tensor(val_loss / len(val_loader)).to(device)
    torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
    val_loss = val_loss.item() / world_size

    if rank == 0:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

        # Save model after every epoch
        torch.save(model.module.state_dict(), save_path)  # Overwrite the model
        print(f"Model saved at {save_path} after Epoch {epoch + 1}.")

        # Run text generation test after each epoch
        generator = TextGenerator(save_path, model_config)
        prompt = "How are you doing"
        generated_text = generator.generate(prompt, max_length=10)
        print(f"Epoch {epoch + 1}: Generated Text ======================: {generated_text}")
