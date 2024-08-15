import torch
from src.trainer import Loop
from src.model import BabyJoey
from src.config import config  # Import the config directly
from src.data import GutenbergData

def main():
    # Set up the device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the dataset and dataloaders
    data = GutenbergData(config=config)

    # Create DataLoader for the training split
    train_loader = data.dataloader(split='train')

    # Create DataLoader for the validation split
    val_loader = data.dataloader(split='validation')

    # Initialize the model with the configuration
    model = BabyJoey(config)

    # Initialize the training loop with the model, dataloaders, and device
    loop = Loop(model, train_loader, val_loader, config, device=device)

    # Create the training state
    state = loop.create()

    # Run the training loop
    loop.run(state)

if __name__ == "__main__":
    main()
