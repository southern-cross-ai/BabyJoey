import random
from ast import List
from typing import Tuple

from rdflib import Dataset
from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class BabyJoeyUtil:
    def __init__(self) -> None:
        pass

    @staticmethod
    def count_params(model: nn.Module) -> int:
        """Count the total number of model parameters.

        Args:
            model (nn.Module): Input model

        Returns:
            int: Total number of parameters
        """
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def generate_text(model: nn.Module, tokenizer, input_text: str, 
                      max_length: int = 20, strategy: str = 'greedy', top_k: int = 10, 
                      device: str = 'cpu') -> str:
        """Generate text using a trained BabyJoeyModel.

        Args:
            model (nn.Module): The trained model used for text generation.
            tokenizer: Tokenizer used to encode input text and decode output tokens.
            input_text (str): The input text to begin generation from.
            max_length (int): The maximum length of the generated sequence.
            strategy (str): The generation strategy to use ('greedy', 'sampling', 'top_k').
            top_k (int): The number of top tokens to sample from when using top-k sampling.
            device (str): The device to run the generation ('cpu' or 'cuda').

        Returns:
            str: The generated text.
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode

        # Encode the input text to token indices
        input_tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
        generated_tokens = input_tokens

        for _ in range(max_length):
            # Forward pass through the model to get logits for the last token
            logits = model(generated_tokens)[0, -1, :]  # Shape: (vocab_size,)
            
            if strategy == 'greedy':
                # Greedy strategy: Pick the token with the highest probability
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            elif strategy == 'sampling':
                # Sampling strategy: Sample from the softmax probability distribution
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            elif strategy == 'top_k':
                # Top-k sampling strategy
                top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
                top_k_probabilities = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(top_k_probabilities, num_samples=1)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Append the next token to the generated sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)

            # Check if the model has generated the end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode the generated tokens to text
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return generated_text
