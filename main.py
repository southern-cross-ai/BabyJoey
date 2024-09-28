# Make sure you install the required packages >>> pip install -r requirements.txt

from src.data.data import BabyJoeyDataLoader, BabyJoeyDataset

# load hyperparameters defined in config/config.py
print("Loading configurations from `config.py`...")
from src.config.config import (
    DATA, COLUMN_NAME,                                             # Hugging Face Setup
    TRAIN_FILE, VALID_FILE, SAMPLE_RATIO, SPLIT_RATIO,             # Local Dataset Setup
    BATCH_SIZE,                                                    # Dataloader Setup
    VOCAB_SIZE, SEQUENCE_LENGTH, N_EMBD, N_HEAD, N_LAYER_DECODER,  # Model Structure
    LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, GAMMA,                 # Optimisation Hyperparameters
)
# load functional classes from submodules under src
print("Loading core functional classes under `src`...")
from src import (
    BabyJoeyModel,       # model/model.py - definitions of model structure
    BabyJoeyUnit,        # training/train.py - logic of training and validation
    Log,                 # logs/log.py - logging functions
    WandB,               # TODO: finish wandb callback
)
# TODO: Is there a better design to manage util functions? Will src/__init.__py cause problems?
print("Loading other utility classes under `src`...")
from src import (
    BabyJoeyUtil         # util/utils.py
)

import torch
from torch.utils.data import Subset
from torchtnt.framework.fit import fit


def main():
    # download/save datasets if not existed, otherwise load tokenised datasets
    print("Preparing training and validation datasets...")
    dataset = BabyJoeyDataset(
        data_path=DATA,                   # hf dataset
        column_name=COLUMN_NAME,          # column of dataset to use as input
        sequence_length=SEQUENCE_LENGTH,  # max token length of input
        train_file=TRAIN_FILE,            # path to load/save tokenised training set
        valid_file=VALID_FILE,            # path to load/save tokenised validation set
        split_ratio=SPLIT_RATIO,          # split ratio of validation set
        sample_ratio=SAMPLE_RATIO         # sample ratio of whole dataset
    )
    # TODO: Integrate sample_dataset when initialising BabyJoeyDataset.
    #       Consider moving it into a function under BabyJoeyDataset class,
    #       or calling from utils insider BabyJoeyDataset.
    training_dataset, validation_dataset = dataset.load_or_create_datasets()
    print("Created training and validation datasets")

    # prepare dataloaders given predefined batch size
    print("Preparing training and validation dataloaders...")
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset, BATCH_SIZE)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()
    print(f"Training dataloader has {len(training_dataloader)} batches, "\
          f"validation dataloader has {len(validation_dataloader)} batches")

    # initialise a model based on predefined model structure
    print("Building a model...")
    # TODO: Load device config from config.py? When to explicitly specify device?
    # TODO: Support more devices later https://pytorch.org/docs/stable/distributed.html
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BabyJoeyModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER_DECODER, SEQUENCE_LENGTH).to(device)
    n_params = BabyJoeyUtil.count_params(model)
    print(f"Initialised a model on {device} with {n_params} trainable parameters")
    
    # TODO: Add comments for AutoUnit
    # prepare AutoUnit
    print("Preparing for training/evaluation/prediction logic...")
    baby_joey_unit = BabyJoeyUnit(
        module=model, 
        device=device, 
        lr=LEARNING_RATE,           # default 1e-5
        weight_decay=WEIGHT_DECAY,  # default 1e-3
        step_size=STEP_SIZE,        # default 1
        gamma=GAMMA                 # default 0.9
        # TODO: Add rank arguments for DDP
        )
    # TODO: Based on which functions are implemented, provide more info on what logic will be executed.
    print("Created training/evaluation/prediction logic")

    # Train and evaluate the model using the defined AutoUnit and Callbacks
    print("Executing training/evaluation/prediction process...")
    fit(
        baby_joey_unit,  # training AutoUnit in train.py
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=2,  # TODO: Load from config.py
        callbacks=[Log()]
    )
    print("Finished training/evaluation/prediction process")


if __name__ == "__main__":
    main()
