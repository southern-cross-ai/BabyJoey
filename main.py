import logging 
logging.getLogger("torchtnt").setLevel(logging.WARNING)  # suppress INFO logs from TorchTNT

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


@hydra.main(version_base=None, config_name="baby_joey_config")  # TODO: Can users change the hard-coded config_name?
def main(config: BabyJoeyConfig):
    
    # TODO: The idea of changes in config is that:
    # When people create datasets, they should only need to worry about the function to use - BabyJoeyDataset
    #                                           and the configs corresponds to the function - DataConfig
    # Another example, when creating dataloaders, users only need BabyJoeyDataLoader + DataLoaderConfig
    # When creating BabyJoeyModel, it needs BabyJoeyModel + ModelConfig, not EmbeddingConfig or TransformerConfig
    # In main.py, users should only call BabyJoeyFuncName with its corresponding NameConfig
    
    # Datasets <- DataConfig
    # TODO: I want to change it to:
    # train_dataset, val_dataset = BabyJoeyDataset(config.data)
    # by designing a __new__() function inside BabyJoeyDataset
    train_dataset, val_dataset = BabyJoeyDataset(config.data).get_datasets()
    # Dataloaders <- DataLoaderConfig
    # TODO: Same as above. Can we do something about "state" to register datasets and dataloaders on it? Maybe not a good idea.
    # train_dataloader, val_dataloader = BabyJoeyDataloader(config.dataloader)
    train_dataloader, val_dataloader = BabyJoeyDataLoader(config.dataloader).get_dataloaders(train_dataset, val_dataset)

    # Model setup
    # TODO: It's better to be:
    # model = BabyJoeyModel(config.model)
    # Users once configed device in config.py, they should not worry about device anymore.
    device = torch.device(config.training.device)
    model = BabyJoeyModel(config.model)
    
    print(f"Model initialized on {device} with {BabyJoeyUtil.count_params(model)} parameters")

    # Training logic setup
    baby_joey_unit = BabyJoeyUnit(
        module=model,
        device=device,
        # TODO: A bit ugly to use configs in this tedious way.
        lr=config.optimization.optimizer.learning_rate,
        weight_decay=config.optimization.optimizer.weight_decay,
        step_size=config.optimization.scheduler.step_size,
        gamma=config.optimization.scheduler.gamma
    )
    # TODO: I manage to change the above code block into this single line.
    # baby_joey_unit = BabyJoeyUnit(config.unit)(model)

    # Training loop
    fit(
        baby_joey_unit,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        max_epochs=config.training.max_epochs,
        callbacks=[
            TQDMProgressBar(), # FIXME: TQDMProgressBar's temporally fixed
            WandB(config.wandb)
        ]
    )
    # TODO: I manage to change the above code block into this single line.
    # fit(config.fit)(baby_joey_unit)  # TODO

if __name__ == "__main__":
    main()
