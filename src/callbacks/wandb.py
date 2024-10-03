import wandb
from torchtnt.framework.callback import Callback
import torch

class WandbLogger(Callback):
    def __init__(self, project_name: str):
        """Initialize W&B logger for tracking loss and accuracy."""
        self.project_name = project_name
        self.epoch_losses = []  # To store losses for the epoch
        self.epoch_correct = 0  # To track correct predictions for accuracy
        self.epoch_total = 0    # To track total predictions for accuracy
        
        # Initialize W&B project
        wandb.init(project=self.project_name)

    ######################## Training state #########################
    def on_train_start(self, state, unit) -> None:
        print("Training started!\n")

    def on_train_epoch_start(self, state, unit) -> None:
        print(f"Epoch {unit.train_progress.num_epochs_completed}!")
        self.epoch_losses = []  # Reset losses at the start of each epoch
        self.epoch_correct = 0  # Reset correct predictions
        self.epoch_total = 0    # Reset total predictions

    def on_train_step_end(self, state, unit) -> None:
        if state.train_state and state.train_state.step_output:
            current_loss = state.train_state.step_output[0].item()  # Extract the loss
            print(f"\rBatch {unit.train_progress.num_steps_completed_in_epoch}: Current Loss: {current_loss:.6f}", end="")
            self.epoch_losses.append(current_loss)

            # Assuming step_output contains logits and labels (based on your model)
            if len(state.train_state.step_output) > 2:
                logits = state.train_state.step_output[1]
                labels = state.train_state.step_output[2]
                
                # Get the predicted token (highest probability from logits)
                _, preds = torch.max(logits, dim=-1)

                # Calculate correct predictions and total tokens
                correct = torch.sum(preds == labels).item()
                total = labels.numel()  # Total number of tokens

                # Update epoch totals for accuracy
                self.epoch_correct += correct
                self.epoch_total += total

            # Log current loss and accuracy to W&B
            wandb.log({
                "train_loss": current_loss,
                "train_accuracy": (self.epoch_correct / self.epoch_total) * 100 if self.epoch_total > 0 else 0,
                "step": unit.train_progress.num_steps_completed
            })

    def on_train_epoch_end(self, state, unit) -> None:
        # Calculate and log average loss for the epoch
        if self.epoch_losses:
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        else:
            avg_loss = None

        # Calculate and log accuracy for the epoch
        if self.epoch_total > 0:
            accuracy = (self.epoch_correct / self.epoch_total) * 100
        else:
            accuracy = None

        # Log average loss and accuracy to W&B at the end of the epoch
        wandb.log({
            "epoch_train_loss": avg_loss,
            "epoch_train_accuracy": accuracy,
            "epoch": unit.train_progress.num_epochs_completed
        })

    ######################## Evaluation state #########################
    def on_eval_epoch_start(self, state, unit) -> None:
        print(f"\nEvaluation epoch {unit.eval_progress.num_epochs_completed} started.\n")
        self.epoch_correct = 0  # Reset correct predictions for evaluation
        self.epoch_total = 0    # Reset total predictions for evaluation

    def on_eval_step_end(self, state, unit) -> None:
        if state.eval_state and state.eval_state.step_output:
            current_loss = state.eval_state.step_output[0].item()  # Extract validation loss

            # Assuming step_output contains logits and labels
            if len(state.eval_state.step_output) > 2:
                logits = state.eval_state.step_output[1]
                labels = state.eval_state.step_output[2]
                
                # Get the predicted token (highest probability from logits)
                _, preds = torch.max(logits, dim=-1)

                # Calculate correct predictions and total tokens
                correct = torch.sum(preds == labels).item()
                total = labels.numel()

                # Update epoch totals for accuracy during evaluation
                self.epoch_correct += correct
                self.epoch_total += total

            # Log validation loss to W&B
            wandb.log({
                "val_loss": current_loss,
                "step": unit.eval_progress.num_steps_completed
            })

    def on_eval_epoch_end(self, state, unit) -> None:
        # Calculate and log accuracy for the evaluation epoch
        if self.epoch_total > 0:
            accuracy = (self.epoch_correct / self.epoch_total) * 100
        else:
            accuracy = None

        # Log evaluation accuracy to W&B at the end of the epoch
        wandb.log({
            "epoch_val_accuracy": accuracy,
            "epoch": unit.eval_progress.num_epochs_completed
        })

    def on_train_end(self, state, unit) -> None:
        """Finish W&B logging at the end of training."""
        wandb.finish()

    ######################## Exception Handling #########################
    def on_exception(self, state, unit, exc: BaseException) -> None:
        print(f"Exception occurred: {exc}")
        wandb.log({"exception": str(exc)})
        wandb.finish()

