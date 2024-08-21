import torch
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.config import BabyJoeyConfig

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GutenbergData:
    def __init__(self, config: BabyJoeyConfig, model_name="gpt2"):
        self.config = config
        self.tokenizer = self._initialize_tokenizer(model_name)
        self._tokenized_dataset = None  # Cache for the tokenized dataset

    def _initialize_tokenizer(self, model_name: str):
        """
        Initialize the tokenizer.
        """
        logger.info("Initializing the tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            logger.info("Setting the pad token...")
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer

    def raw_dataset(self):
        """
        Load the raw dataset based on the configuration.
        """
        logger.info("Loading the raw dataset...")
        try:
            raw_dataset = load_dataset(self.config.data.dataset_name)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        return raw_dataset

    def _tokenize_function(self, examples):
        """
        Tokenization function to be applied to the dataset.
        """
        if "Paragraph" not in examples:
            logger.error("The 'Paragraph' column is missing from the dataset.")
            raise ValueError("The 'Paragraph' column is missing from the dataset.")

        return self.tokenizer(
            examples["Paragraph"],
            padding="max_length",
            truncation=True,
            max_length=self.config.model.max_position_embeddings
        )

    def tokenized_dataset(self):
        """
        Tokenize the entire dataset and cache it.
        """
        if self._tokenized_dataset:
            logger.info("Using cached tokenized dataset...")
            return self._tokenized_dataset

        raw_dataset = self.raw_dataset()
        logger.info("Mapping the tokenization function over the dataset...")
        tokenized_dataset = raw_dataset.map(self._tokenize_function, batched=True, remove_columns=["Paragraph"])

        if 'validation' not in tokenized_dataset:
            logger.info("Creating validation split...")
            tokenized_dataset = self._create_validation_split(tokenized_dataset)

        self._tokenized_dataset = tokenized_dataset
        return tokenized_dataset

    def _create_validation_split(self, tokenized_dataset):
        """
        Create a validation split if it doesn't exist.
        """
        train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
        return DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })

    def dataloader(self, split='train'):
        """
        Prepare the DataLoader for training or validation.
        """
        logger.info(f"Preparing DataLoader for {split} split...")
        tokenized_dataset = self.tokenized_dataset()

        logger.info(f"Creating DataLoader for the {split} dataset...")
        return DataLoader(
            tokenized_dataset[split], 
            batch_size=self.config.data.batch_size, 
            shuffle=True if split == 'train' else False, 
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        """
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
