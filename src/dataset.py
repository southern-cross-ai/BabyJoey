import os
from dataclasses import dataclass
# from huggingface_hub import login
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
import tiktoken

if __name__ == "__main__":
    from config import DataSetConfig
else:
  from src.config import DataSetConfig


class DataSetFactory:
    def __init__(self, config: DataSetConfig):
        self.config = config
        self.dataset_name = config.dataset_name
        self.chunk_size = config.chunk_size
        self.stride = config.stride
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir

        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Check if training and validation tensors exist
        train_exists = os.path.exists(os.path.join(self.data_dir, "train_tensor.pt"))
        val_exists = os.path.exists(os.path.join(self.data_dir, "validation_tensor.pt"))

        if not (train_exists and val_exists):
            # Load the dataset only if the tensors do not exist
            print("Loading dataset...")
            self.dataset = load_dataset(self.dataset_name)
        else:
            self.dataset = None  # Dataset is not needed if tensors already exist

    def save_tensor(self, tensor, filename):
        """Save the processed tensor to a .pt file."""
        filepath = os.path.join(self.data_dir, filename)
        torch.save(tensor, filepath)

    def load_tensor_from_file(self, filename):
        """Load a processed tensor from a .pt file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return torch.load(filepath)
        return None

    def process_split(self, split):
        """Processes a dataset split into overlapping chunks."""
        # Check if the processed tensor already exists
        tensor = self.load_tensor_from_file(f"{split}_tensor.pt")
        if tensor is not None:
            print(f"Loaded {split} tensor from disk.")
            return tensor

        # Process the dataset if it does not exist
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Cannot process split.")

        cl100k_base_column = self.dataset[split]["cl100k_base"]
        print(f"Number of sequences in {split}: {len(cl100k_base_column)}")

        # Flatten the tokens
        all_tokens = [token for sequence in cl100k_base_column for token in sequence]
        print(f"Total tokens in {split}: {len(all_tokens)}")

        # Create overlapping chunks
        chunks = [
            all_tokens[i:i + self.chunk_size]
            for i in range(0, len(all_tokens) - self.chunk_size + 1, self.stride)
        ]
        print(f"Number of overlapping chunks in {split}: {len(chunks)}")

        tensor = torch.tensor(chunks, dtype=torch.long)

        # Save the processed tensor
        self.save_tensor(tensor, f"{split}_tensor.pt")
        print(f"Saved {split} tensor to disk.")

        return tensor

    def __call__(self):
        """Creates datasets for training and validation splits."""
        train_tensor = self.process_split("train")
        val_tensor = self.process_split("validation")

        # Create TensorDatasets
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)

        return train_dataset, val_dataset

# Example usage
if __name__ == "__main__":
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
