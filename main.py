print('Getting imports')

import torch
import torch
from torch.utils.data import Subset
import random
import itertools
from src import (
    BabyJoeyDataset,
    BabyJoeyDataLoader, 
    BabyJoeyModel, 
    BabyJoeyUnit,
    WandB,
    Log
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

###############################################################################
    # Uncomment to Reduce the datasets to 1/16 of their original size
    def reduce_dataset(dataset, fraction):
        dataset_size = len(dataset)
        reduced_size = int(dataset_size * fraction)
        indices = random.sample(range(dataset_size), reduced_size)
        return Subset(dataset, indices)

    training_dataset = reduce_dataset(training_dataset, 1/256)
    validation_dataset = reduce_dataset(validation_dataset, 1/265)
###############################################################################

    # Create DataLoaders
    dataloader = BabyJoeyDataLoader(training_dataset, validation_dataset, BATCH_SIZE)
    training_dataloader, validation_dataloader = dataloader.get_dataloaders()

    total_training_batches = len(training_dataloader)
    total_validation_batches = len(validation_dataloader)

    print(f"Total number of training batches: {total_training_batches}")
    print(f"Total number of validation batches: {total_validation_batches}")

    print("Getting Model")

    # Define device and initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BabyJoeyModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER_DECODER, SEQUENCE_LENGTH).to(device)
    baby_joey_unit = BabyJoeyUnit(module=model, device=device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(model)}")

    print("Starting training")

    # Train the model using the defined AutoUnit and callback
    fit(
        baby_joey_unit,
        train_dataloader=training_dataloader,
        eval_dataloader=validation_dataloader,
        max_epochs=5,
        callbacks=[Log()]
    )

if __name__ == "__main__":
    main()
