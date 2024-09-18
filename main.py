print('Getting imports')

import torch
from src import (
    BabyJoeyDataset,
    BabyJoeyDataLoader, 
    BabyJoeyModel, 
    BabyJoeyUnit,
    TestingCallback
)
from src.config.config import (
    DATA, SEQUENCE_LENGTH, TRAIN_FILE, VALID_FILE, BATCH_SIZE, 
    VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER_DECODER, LEARNING_RATE, 
    WEIGHT_DECAY, STEP_SIZE, GAMMA
)
from torchtnt.framework.fit import fit  # Ensure this is the correct import for the 'fit' function

print('Starting Main()')

def main():
    print('Getting data...')
    # Load datasets
    dataset = BabyJoeyDataset(DATA, SEQUENCE_LENGTH, TRAIN_FILE, VALID_FILE)
    training_dataset, validation_dataset = dataset.load_or_create_datasets()

    # Create DataLoaders
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset, BATCH_SIZE)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()

    print("Getting Model")

    # Define device and initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BabyJoeyModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER_DECODER, SEQUENCE_LENGTH).to(device)
    baby_joey_unit = BabyJoeyUnit(module=model, device=device)

    print("Starting training")

    # Train the model using the defined AutoUnit and callback
    fit(
        baby_joey_unit,
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=2,
        callbacks=[TestingCallback()]
    )

if __name__ == "__main__":
    main()
