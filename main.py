import logging
# Set the logging level to WARNING or higher to suppress INFO logs from TorchTNT
logging.getLogger("torchtnt").setLevel(logging.WARNING)

# Make sure you install the required packages >>> pip install -r requirements.txt
import torch
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks import TQDMProgressBar
import hydra

from src.data import BabyJoeyDataLoader, BabyJoeyDataset
from src.model import BabyJoeyModel
from src.train import BabyJoeyUnit
from src.utils import BabyJoeyUtil
from src.config.config import BabyJoeyConfig

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
    
    # baby_joey_unit = BabyJoeyUnit(config.unit)(model)

    # Training loop
    fit(
        baby_joey_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        max_epochs=config.training.max_epochs,
        callbacks=[TQDMProgressBar()]
    )
    
    # fit(config.fit)(baby_joey_unit)  # TODO

if __name__ == "__main__":
    main()
