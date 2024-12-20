from src.config import ModelConfig, TrainingConfig, DataSetConfig
from src.dataset import DataSetFactory
from torch.utils.data import DataLoader
import tiktoken


creat_data = DataSetFactory(DataSetConfig())
train_dataset, val_dataset = creat_data()

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Inspect the DataLoaders and decode the first two batches
print("Training DataLoader:")
for i, batch in enumerate(train_loader):
    print(f"Batch {i + 1} shape: {batch[0].shape}")  # Should print [batch_size, chunk_size]
    decoded_texts = [tokenizer.decode(seq.tolist()) for seq in batch[0]]
    print("Decoded Texts:")
    for j, text in enumerate(decoded_texts):
        print(f"Text {j + 1}: {text}")
    if i == 1:  # Stop after first two batches
        break

print("Validation DataLoader:")
for i, batch in enumerate(val_loader):
    print(f"Batch {i + 1} shape: {batch[0].shape}")  # Should print [batch_size, chunk_size]
    decoded_texts = [tokenizer.decode(seq.tolist()) for seq in batch[0]]
    print("Decoded Texts:")
    for j, text in enumerate(decoded_texts):
        print(f"Text {j + 1}: {text}")
    if i == 1:  # Stop after first two batches
        break





# dataset_instance = GetJoeyData(DatasetConfig())
# training_dataset, validation_dataset = dataset_instance.create_dataloaders()

# # Initialize model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BabyJoeyModel(ModelConfig()).to(device)

# train = TrainingConfig(device=device)

# trainer = ModelTrainer(model, training_dataset, validation_dataset, train)
# trainer.fit(num_epochs=2)

print("working to here")
