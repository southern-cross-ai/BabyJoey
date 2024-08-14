print("Starting main.py")
from src.data import GutenbergData
from src.config import config
print("Imports done - Running main function")

def main():
    print("Starting BabyJoey model")
    # Device management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize the model, data loaders, and training loop
    model = BabyJoey(config)
    gutenberg_data = GutenbergData(config=config)
    dataloader = gutenberg_data.get_dataloader(split='train')
    print("Model and data loaded successfully")

    # Test the DataLoader by iterating over it and printing out a batch
    for batch in dataloader:  # Use 'dataloader' here
        print("Batch input_ids shape:", batch['input_ids'].shape)
        print("Batch attention_mask shape:", batch['attention_mask'].shape)
        print("Batch input_ids:", batch['input_ids'][0])  # Print the first example in the batch
        print("Batch attention_mask:", batch['attention_mask'][0])  # Print the first example's attention mask

        # Break after one batch for testing purposes
        break 

    # training_loop = Loop(model, train_loader, val_loader, config, device=device)
    # state = training_loop.create()

    # # Run the training loop
    # training_loop.run(state)
    
    print("Main function done")

if __name__ == '__main__':
    main()
