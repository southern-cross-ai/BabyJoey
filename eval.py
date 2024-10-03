import torch
import deepspeed
import gradio as gr
from transformers import GPT2Tokenizer
from src.model import BabyJoeyModel
from src.train import BabyJoeyUnit
from src.config.config import BabyJoeyConfig
from hydra import initialize, compose


class BabyJoeyEvaluator:
    def __init__(self, cfg: BabyJoeyConfig, checkpoint_dir: str):
        """Initialize the evaluator with the model and tokenizer."""
        self.device = torch.device(cfg.training.device)

        # Load model and tokenizer
        self.model = BabyJoeyModel(cfg)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize DeepSpeed and load the best checkpoint
        self.baby_joey_unit = BabyJoeyUnit(
            module=self.model,
            device=self.device,
            lr=cfg.optimization.learning_rate,
            weight_decay=cfg.optimization.weight_decay,
            step_size=cfg.optimization.step_size,
            gamma=cfg.optimization.gamma,
            use_fp16=cfg.deepspeed.fp16,
            checkpoint_dir=checkpoint_dir  # Directory to load the checkpoint from
        )

        # Load the best checkpoint
        self.baby_joey_unit.load_checkpoint("best_model_checkpoint")  # Assuming the best model is saved as this

    def infer(self, user_input: str) -> str:
        """Take user input, tokenize it, run inference and return the output."""
        # Tokenize user input
        inputs = self.tokenizer(user_input, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Generate output from the model
        with torch.no_grad():
            output_logits = self.model(inputs['input_ids'])

        # Get the predicted tokens
        predicted_tokens = torch.argmax(output_logits, dim=-1)

        # Decode the output tokens to text
        output_text = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

        return output_text


def main():
    # Load the configuration using Hydra
    initialize(config_path=None)
    cfg = compose(config_name="baby_joey_config")

    # Create the evaluator object
    checkpoint_dir = './checkpoints'  # Define the directory where checkpoints are saved
    evaluator = BabyJoeyEvaluator(cfg, checkpoint_dir)

    # Define the Gradio interface
    def gradio_inference(user_input):
        return evaluator.infer(user_input)

    # Create a Gradio interface with a text input and output
    gr.Interface(
        fn=gradio_inference,
        inputs="text",
        outputs="text",
        title="BabyJoey Model Inference",
        description="Enter a text and see the model's response."
    ).launch()


if __name__ == "__main__":
    main()
