import logging
import hydra
import torch
from torchtnt.framework.callbacks import TQDMProgressBar
from torchtnt.framework.fit import fit

from src.callbacks import WandB
from src.config import BabyJoeyConfig
from src.data import BabyJoeyDataLoader, BabyJoeyDataset
from src.model import BabyJoeyModel
from src.train import BabyJoeyUnit
from src.utils import BabyJoeyUtil

logging.getLogger("torchtnt").setLevel(logging.WARNING)  # suppress INFO logs from TorchTNT

@hydra.main(version_base=None, config_name="baby_joey_config")
def main(config: BabyJoeyConfig):
    # Datasets
    train_dataset, val_dataset = BabyJoeyDataset(config.data).get_datasets()
    
    # Dataloaders
    train_dataloader, val_dataloader = BabyJoeyDataLoader(config.dataloader).get_dataloaders(train_dataset, val_dataset)

    # Model setup
    device = torch.device(config.training.device)
    model = BabyJoeyModel(config.model)
    
    print(f"Model initialized on {device} with {BabyJoeyUtil.count_params(model)} parameters")

    # Training logic setup
    baby_joey_unit = BabyJoeyUnit(
        module=model,
        device=device,
        lr=config.optimization.optimizer.learning_rate,
        weight_decay=config.optimization.optimizer.weight_decay,
        step_size=config.optimization.scheduler.step_size,
        gamma=config.optimization.scheduler.gamma
    )

    # Training loop
    fit(
        baby_joey_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        max_epochs=config.training.max_epochs,
        callbacks=[
            TQDMProgressBar(),
            WandB(config.wandb)
        ]
    )

if __name__ == "__main__":
    main()
