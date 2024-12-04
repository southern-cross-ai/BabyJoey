import torch
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.framework.callbacks import TQDMProgressBar
from dataclasses import dataclass


import hydra

# Model configuration using dataclass
@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layers: int
    max_seq_len: int
    padding_idx: int  # Index of the padding token
    dropout_rate: float = 0.1  # Default dropout rate

config = ModelConfig(
    vocab_size=50,  # Example vocabulary size
    n_embd=512,
    n_head=8,
    n_layers=1,
    max_seq_len=768,
    padding_idx=0,  # Padding token index
    dropout_rate=0.1
)

from src.data import BabyJoeyDataLoader, BabyJoeyDataset
from src.model import BabyJoeyModel
from src.config.config import BabyJoeyConfig

@hydra.main(version_base=None, config_name="baby_joey_config")
def main(cfg: BabyJoeyConfig):
    # Dataset setup
    dataset = BabyJoeyDataset()
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Dataloader setup
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()

    # Model setup
    device = torch.device(cfg.training.device)
    model = BabyJoeyModel(config).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Define training step
    def train_step(state: State, data):
        model.train()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    # Define evaluation step
    def eval_step(state: State, data):
        model.eval()
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        return {"val_loss": loss.item()}

    # Setup callbacks
    callbacks = [TQDMProgressBar()]

    # Run the training loop
    fit(
        model=model,
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=cfg.training.max_epochs,
        train_step=train_step,
        eval_step=eval_step,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()
