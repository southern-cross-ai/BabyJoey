import logging

# Set the logging level to WARNING or higher to suppress INFO logs from TorchTNT
logging.getLogger("torchtnt").setLevel(logging.WARNING)


# Make sure you install the required packages >>> pip install -r requirements.txt
import torch
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks import TQDMProgressBar, LearningRateMonitor

import hydra

from src.data import BabyJoeyDataLoader, BabyJoeyDataset
from src.model import BabyJoeyModel
from src.train import BabyJoeyUnit
from src.utils import BabyJoeyUtil
from src.config.config import BabyJoeyConfig
from src.callbacks import WandbLogger, Log

@hydra.main(version_base=None, config_name="baby_joey_config")
def main(cfg: BabyJoeyConfig):
    # Dataset setup
    dataset = BabyJoeyDataset(cfg)
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Dataloader setup
    dataloader = BabyJoeyDataLoader(cfg, training_dataset, validation_dataset)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()

    # Model setup
    device = torch.device(cfg.training.device)
    model = BabyJoeyModel(cfg).to(device)
    n_params = BabyJoeyUtil.count_params(model)
    print(f"Model initialized on {device} with {n_params} parameters")

    # Training logic setup
    baby_joey_unit = BabyJoeyUnit(
        module=model,
        device=device,
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
        step_size=cfg.optimization.step_size,
        gamma=cfg.optimization.gamma
    )

    # Training loop
    fit(
        baby_joey_unit,
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=cfg.training.max_epochs,
        callbacks=[Log()]
    )

if __name__ == "__main__":
    main()
