import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


class ModelTrainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, config: DictConfig):
        self.model = model.to(config.device)
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        self.config = config

        # Initialize optimizer dynamically
        # Dynamically resolve the optimizer class
        opt_class = eval(config.optimizer.cls)
        opt_params = config.optimizer.params
        self.optimizer = opt_class(
            self.model.parameters(),
            lr=config.learning_rate,
            **opt_params
        )

    def train_epoch(self) -> None:
        self.model.train()
        for step, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.config.device)

            logits = self.model(input_ids)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_input_ids = input_ids[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shifted_logits.view(-1, self.config.vocab_size),
                shifted_input_ids.view(-1),
                ignore_index=self.config.padding_idx,
                label_smoothing=self.config.label_smoothing
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}, Loss: {loss.item():.4f}")

    def validate_epoch(self) -> float:
        self.model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                logits = self.model(input_ids)
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_input_ids = input_ids[:, 1:].contiguous()

                val_batch_loss = nn.functional.cross_entropy(
                    shifted_logits.view(-1, self.config.vocab_size),
                    shifted_input_ids.view(-1),
                    ignore_index=self.config.padding_idx,
                    label_smoothing=self.config.label_smoothing
                )

                val_loss += val_batch_loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        self.model.train()
        return avg_val_loss

    def save_checkpoint(self, epoch: int, avg_val_loss: float) -> None:
        checkpoint_path = f"baby_joey_checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'config': OmegaConf.to_container(self.config),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def fit(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.train_epoch()
            avg_val_loss = self.validate_epoch()
            self.save_checkpoint(epoch, avg_val_loss)


if __name__ == "__main__":
    @hydra.main(config_path="../conf", config_name="config")
    def main(cfg: DictConfig):
        # Define a simple dataset
        class SimpleDataset(Dataset):
            def __init__(self, size: int, seq_length: int, vocab_size: int):
                self.data = torch.randint(0, vocab_size, (size, seq_length))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx: int):
                return {'input_ids': self.data[idx]}

        # Define a simple model
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int):
                super(SimpleModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.linear = nn.Linear(embed_dim, vocab_size)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embedding(x)
                logits = self.linear(x)
                return logits

        # Load configurations and initialize model, dataset, and trainer
        model = SimpleModel(vocab_size=cfg.vocab_size, embed_dim=64)
        train_dataset = SimpleDataset(100, 10, cfg.vocab_size)
        val_dataset = SimpleDataset(50, 10, cfg.vocab_size)

        trainer = ModelTrainer(model, train_dataset, val_dataset, cfg)
        trainer.fit(num_epochs=cfg.training.num_epochs)

    # Call the main function
    main()
