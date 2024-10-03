import logging
import torch
import deepspeed
import hydra
from src.data import BabyJoeyDataLoader, BabyJoeyDataset
from src.model import BabyJoeyModel
from src.train import BabyJoeyUnit
from src.utils import BabyJoeyUtil
from src.config.config import BabyJoeyConfig


@hydra.main(version_base=None, config_name="baby_joey_config")
def main(cfg: BabyJoeyConfig):
    # Dataset setup
    dataset = BabyJoeyDataset(cfg)
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Dataloader setup
    dataloader = BabyJoeyDataLoader(cfg, training_dataset, validation_dataset)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders(distributed=False)

    # Model setup
    device = torch.device(cfg.training.device)
    model = BabyJoeyModel(cfg)  # No need to move to device here; DeepSpeed will handle this
    n_params = BabyJoeyUtil.count_params(model)
    print(f"Model initialized with {n_params} parameters")

    # Initialize BabyJoeyUnit with DeepSpeed
    baby_joey_unit = BabyJoeyUnit(
        module=model,
        device=device,
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
        step_size=cfg.optimization.step_size,
        gamma=cfg.optimization.gamma,
        use_fp16=cfg.deepspeed.fp16,  # Enable FP16 if configured
        checkpoint_dir='src/checkpoint'  # Directory for saving checkpoints
    )

    # Training loop
    for epoch in range(cfg.training.max_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for step, batch in enumerate(training_dataloader):
            baby_joey_unit.train_step(batch)
            loss, _ = baby_joey_unit.compute_loss(None, batch)
            total_loss += loss.item()

        avg_loss = total_loss / len(training_dataloader)
        print(f"Epoch {epoch + 1}/{cfg.training.max_epochs}, Average Loss: {avg_loss}")

        # Save the best model based on the validation loss
        baby_joey_unit.maybe_save_best_checkpoint(epoch + 1, avg_loss)

        # Optionally evaluate at the end of each epoch
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for step, batch in enumerate(validation_dataloader):
                baby_joey_unit.compute_loss(None, batch)  # Only computing loss for evaluation

if __name__ == "__main__":
    main()
