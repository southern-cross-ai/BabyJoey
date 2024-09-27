# Make sure you install the required packages >>> pip install -r requirements.txt

# load hyperparameters defined in config/config.py
print("Loading configurations from `config.py`...")
from src.config.config import (
    DATA, COLUMN_NAME,                                             # Hugging Face Setup
    TRAIN_FILE, VALID_FILE, SPLIT_RATIO,                           # Local Dataset Setup
    BATCH_SIZE,                                                    # Dataloader Setup
    VOCAB_SIZE, SEQUENCE_LENGTH, N_EMBD, N_HEAD, N_LAYER_DECODER,  # Model Structure
    LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, GAMMA,                 # Optimisation Hyperparameters
)
# load functional classes from submodules under src
print("Loading core functional classes under `src`...")
from src import (
    BabyJoeyDataLoader,  # data/dataloader.py - dataloaders for training and validation sets
    BabyJoeyDataset,     # data/dataset.py - datasets for training and validation
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
    
    print('Getting data...')
    # Load datasets
    dataset = BabyJoeyDataset(
        data_path=DATA, 
        column_name=COLUMN_NAME,
        sequence_length=SEQUENCE_LENGTH, 
        train_file=TRAIN_FILE, 
        valid_file=VALID_FILE,
        split_ratio=SPLIT_RATIO)
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Prepare DataLoaders
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset, BATCH_SIZE)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()
    print(f"Total number of training batches: {len(training_dataloader)}")
    print(f"Total number of validation batches: {len(validation_dataloader)}")

    # Prepare BabyJoey
    print("Getting Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Load device from config.py?
    model = BabyJoeyModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER_DECODER, SEQUENCE_LENGTH).to(device)
    print(f"Number of trainable parameters: {BabyJoeyUtil.count_params(model)}")
    
    # Prepare AutoUnit
    baby_joey_unit = BabyJoeyUnit(module=model, device=device, 
                                  lr=LEARNING_RATE,           # default 1e-5
                                  weight_decay=WEIGHT_DECAY,  # default 1e-3
                                  step_size=STEP_SIZE,        # default 1
                                  gamma=GAMMA,                # default 0.9
                                  # TODO: Add rank for DDP
                                  )                 
    

    # Train and evaluate the model using the defined AutoUnit and callback
    print("Starting training")
    fit(
        baby_joey_unit,  # training AutoUnit in train.py
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=2,  # TODO: Load from config.py
        callbacks=[Log()]
    )


if __name__ == "__main__":
    main()
