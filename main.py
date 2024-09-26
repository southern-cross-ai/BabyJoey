print('Getting imports')

# Make sure you install the required packages >>> pip install -r requirements.txt

from src.config.config import (  # load user-specified hyperparams from config.py
    TRAIN_FILE, VALID_FILE, DATA, BATCH_SIZE,                      # dataset settings
    VOCAB_SIZE, SEQUENCE_LENGTH, N_EMBD, N_HEAD, N_LAYER_DECODER,  # model configs
    LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, GAMMA,                 # SGD hyperparams
)
from src import (        # load functional classes from submodules under src
    BabyJoeyDataLoader,  # dataloader.py - return dataloaders for training and validation
    BabyJoeyDataset,     # dataset.py - load dataset configured by DATA from Hugging Face
    BabyJoeyModel,       # model.py - nn.Module subclass, contains definitions of Embeddings and TransformerBlock
    BabyJoeyUnit,        # train.py - AutoUnit subclass
    Log,                 # log.py - Callback subclass
    WandB,               # wandb.py - Callback subclass
)
from src.utils.count_param import count_parameters  # utils.py - count model parameters

import torch
from torch.utils.data import Subset
from torchtnt.framework.fit import fit


print('Starting Main()')

def main():
    print('Getting data...')
    # Load datasets
    dataset = BabyJoeyDataset(
        data_path=DATA, 
        column_name=COLUM_NAME,
        sequence_length=SEQUENCE_LENGTH, 
        train_file=TRAIN_FILE, 
        valid_file=VALID_FILE)
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
    print(f"Number of trainable parameters: {count_parameters(model)}")
    
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
