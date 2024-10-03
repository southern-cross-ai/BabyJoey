import torch
import wandb
from wandb.errors import AuthenticationError, UsageError
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TPredictUnit, TTrainUnit

from src.config.config import WandBConfig

# FIXME: Accuracy is not working
# FIXME: Process bar has issues. See the PR on pytorch/tnt created by Matty.
# TODO: Current WandB will generate logs in terminal. Maybe changing logger to suppress some? Some info is helpful tho.

class WandB(Callback):
    def __init__(self, config: WandBConfig) -> None:
        # Login WandB thorough an API key
        try:
            wandb.login(key=config.api_key, relogin=True)
        except AuthenticationError:
            raise AuthenticationError("WandB API key fails verification with the server")
        except UsageError:
            raise UsageError("WandB API key cannot be configured and no tty")
        # Initialise a project in WandB
        wandb.init(
            project=config.project_name,  # Project name
            config={
                "dataset": config.dataset,
                "num_attention_head": config.num_attention_head,
                "num_decoder_layer": config.num_decoder_layer,
                "max_epochs": config.max_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "step_size": config.step_size,
                "gamma": config.gamma
            }
        )
        
    ######################## Training state #########################
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        pass

    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        self.train_epoch_loss = []
        self.train_epoch_correct = 0
        self.train_epoch_total = 0

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        pass

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if state.train_state and state.train_state.step_output:
            current_loss = state.train_state.step_output[0].item()  # Ensure step_output is valid
            self.train_epoch_loss.append(current_loss)
            wandb.log({"train_step_loss": current_loss})
            # Assuming the step_output contains logits and labels (you may need to adjust this based on your actual model)
            if len(state.train_state.step_output) > 2:
                logits = state.train_state.step_output[1]
                labels = state.train_state.step_output[2]
                # Get the predicted token (highest probability from logits)
                _, preds = torch.max(logits, dim=-1)
                # Calculate correct predictions and total tokens
                correct = torch.sum(preds == labels).item()
                total = labels.numel()  # Total number of tokens
                # Update epoch totals for accuracy
                self.train_epoch_correct += correct
                self.train_epoch_total += total
        else:
            # Print the "No valid loss" message on a new line
            print("No valid loss available at this end of train step")

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        # Calculate and log average loss for the epoch
        if self.train_epoch_loss:
            avg_loss = sum(self.train_epoch_loss) / len(self.train_epoch_loss)
        else:
            avg_loss = None
            print("No valid loss available at this end of train epoch")
        wandb.log({"train_epoch_loss": avg_loss})
        # Calculate and log accuracy for the epoch
        if self.train_epoch_total > 0:
            accuracy = (self.train_epoch_correct / self.train_epoch_total) * 100
        else:
            accuracy = None
            print("No valid loss accuracy at this end of train epoch")
        wandb.log({"train_epoch_acc": accuracy})

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        # print("Training ended!")
        pass

    ######################## Evaluation state #########################
    def on_eval_start(self, state: State, unit: TEvalUnit) -> None:
        pass

    def on_eval_epoch_start(self, state: State, unit: TEvalUnit) -> None:
        self.eval_epoch_loss = []
        self.eval_epoch_correct = 0
        self.eval_epoch_total = 0
        
    def on_eval_get_next_batch_end(self, state: State, unit: TEvalUnit) -> None:
        pass
    
    def on_eval_step_start(self, state: State, unit: TEvalUnit) -> None:
        pass

    def on_eval_step_end(self, state: State, unit: TEvalUnit) -> None:
        if state.eval_state and state.eval_state.step_output:
            current_loss = state.eval_state.step_output[0].item()  # Ensure step_output is valid
            self.eval_epoch_loss.append(current_loss)
            wandb.log({"eval_step_loss": current_loss})
            # Assuming the step_output contains logits and labels (you may need to adjust this based on your actual model)
            if len(state.eval_state.step_output) > 2:
                logits = state.eval_state.step_output[1]
                labels = state.eval_state.step_output[2]
                # Get the predicted token (highest probability from logits)
                _, preds = torch.max(logits, dim=-1)
                # Calculate correct predictions and total tokens
                correct = torch.sum(preds == labels).item()
                total = labels.numel()  # Total number of tokens
                # Update epoch totals for accuracy
                self.eval_epoch_correct += correct
                self.eval_epoch_total += total
        else:
            # Print the "No valid loss" message on a new line
            print("No valid loss available at this end of eval step")
    
    def on_eval_epoch_end(self, state: State, unit: TEvalUnit) -> None:
        # Calculate and log average loss for the epoch
        if self.eval_epoch_loss:
            avg_loss = sum(self.eval_epoch_loss) / len(self.eval_epoch_loss)
        else:
            avg_loss = None
            print("No valid loss available at this end of eval epoch")
        wandb.log({"eval_epoch_loss": avg_loss})
        # Calculate and log accuracy for the epoch
        if self.eval_epoch_total > 0:
            accuracy = (self.eval_epoch_correct / self.eval_epoch_total) * 100
        else:
            accuracy = None
            print("No valid loss accuracy at this end of eval epoch")
        wandb.log({"eval_epoch_acc": accuracy})
    
    def on_eval_end(self, state: State, unit: TEvalUnit) -> None:
        pass


# ------------- TODO Items for BabyJoey -------------:
# Training Loss: wandb.log({"train_loss": loss})
# Validation Loss: wandb.log({"val_loss": val_loss})
# Training Accuracy (for classification tasks): wandb.log({"train_accuracy": train_accuracy})
# Validation Accuracy (for classification tasks): wandb.log({"val_accuracy": val_accuracy})
# Perplexity (for language modeling): wandb.log({"perplexity": perplexity})
# Learning Rate: wandb.log({"learning_rate": lr})
# Epoch Number: wandb.log({"epoch": current_epoch})
# Step Number: wandb.log({"step": current_step})
# Gradient Norm: wandb.log({"gradient_norm": grad_norm})
# Optional Items for Transformers:
# Attention Weights/Maps: wandb.log({"attention_maps": attention_weights}) (if you want to visualize attention layers)
# Validation Metrics (e.g., F1, precision, recall for classification tasks):
# python
# Copy code
# wandb.log({
#     "val_f1": f1_score, 
#     "val_precision": precision, 
#     "val_recall": recall
# })
# Sample Predictions: wandb.log({"predictions": sample_predictions})
# Confusion Matrix (for classification tasks): wandb.log({"confusion_matrix": confusion_matrix})
# Throughput (Samples per second): wandb.log({"throughput": samples_per_sec})
# GPU/CPU Utilization: wandb.log({"gpu_usage": gpu_usage, "cpu_usage": cpu_usage})
# Model Checkpoints (e.g., every few epochs): wandb.save('model_checkpoint.pt')
# Tokenization Stats (optional, if interested in tokenization performance): wandb.log({"avg_tokens_per_sample": avg_tokens})
# For Language Models (additional logs):
# Perplexity (for language modeling): wandb.log({"perplexity": perplexity})
# Log Likelihood: wandb.log({"log_likelihood": log_likelihood})
