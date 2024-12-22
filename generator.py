import torch
from torch.nn.functional import softmax
from src.model import BabyJoeyModel
from src.config import ModelConfig
import tiktoken

class TextGenerator:
    def __init__(self, model_path, config):
        """
        Initialize the text generator.
        
        :param model_path: Path to the saved model file.
        :param config: ModelConfig instance for configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Load the model
        self.model = BabyJoeyModel(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(self, prompt, max_length):
        """
        Generate text based on a prompt.
        
        :param prompt: Starting text for the generator.
        :param max_length: Number of tokens to generate.
        :return: Generated text as a string.
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]

        # Ensure input does not exceed context window
        if input_tensor.size(1) > self.config.context_window:
            raise ValueError("Input sequence exceeds the model's context window.")

        # Generate tokens
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(input_tensor)  # [1, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                probabilities = softmax(next_token_logits, dim=-1)

                # Sample from the distribution
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()

                # Append the generated token
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)

                # Stop generation if the end-of-sequence token is generated
                if next_token_id == self.config.padding_idx:
                    break

        # Decode tokens back into text
        generated_ids = input_tensor[0].tolist()
        return self.tokenizer.decode(generated_ids)

if __name__ == "__main__":
    # Example usage
    model_path = "model_epoch_1.pt"  # Replace with your actual model file
    config = ModelConfig()

    generator = TextGenerator(model_path, config)
    prompt = "yellow who are"
    max_length = 50

    generated_text = generator.generate(prompt, max_length)
    print("Generated Text:")
    print(generated_text)
