import wandb
from torchtnt.framework.callback import Callback
import torch
import logging
import sys

# Set up logging
logger = logging.getLogger(__name__)

class WandbLogger(Callback):
    def __init__(self, project_name: str, log_steps: bool = True, log_accuracy: bool = True):
        """Initialize W&B logger for tracking loss and accuracy."""
        self.project_name = project_name
        self.log_steps = log_steps
        self.log_accuracy = log_accuracy
        self.epoch_losses = []
        self.epoch_correct = 0
        self.epoch_total = 0
        
        # Initialize W&B project
        wandb.init(project=self.project_name)

    ######################## Helper Methods #########################
    def _log_metrics(self, metrics: dict, phase: str):
        """Helper to log metrics to WandB with phase prefix."""
        wandb.log({f"{phase}_{key}": value for key, value in metrics.items()})

    def _calculate_accuracy(self):
        """Helper to calculate accuracy."""
        if self.epoch_total > 0:
            return (self.epoch_correct / self.epoch_total) * 100
        else:
            logger.warning("No valid accuracy at the end of epoch.")
            return None

    ######################## Training state #########################
    def on_train_epoch_start(self, state, unit) -> None:
        logger.info(f"Epoch {unit.train_progress.num_epochs_completed} started.")
        self.epoch_losses = []
        self.epoch_correct = 0
        self.epoch_total = 0

    def on_train_step_end(self, state, unit) -> None:
        if state.train_state and state.train_state.step_output:
            current_loss = state.train_state.step_output[0].item()
            self.epoch_losses.append(current_loss)

            # Keep the current output on the same line using sys.stdout
            sys.stdout.write(f"\rBatch {unit.train_progress.num_steps_completed_in_epoch}: Current Loss: {current_loss:.6f}")
            sys.stdout.flush()

            # Log current loss to W&B
            wandb.log({
                "train_loss": current_loss,
                "step": unit.train_progress.num_steps_completed
            })

            # Handle accuracy
            if len(state.train_state.step_output) > 2 and self.log_accuracy:
                logits, labels = state.train_state.step_output[1:3]
                _, preds = torch.max(logits, dim=-1)
                correct = torch.sum(preds == labels).item()
                total = labels.numel()
                self.epoch_correct += correct
                self.epoch_total += total

                accuracy = self._calculate_accuracy()
                wandb.log({"train_accuracy": accuracy})

    def on_train_epoch_end(self, state, unit) -> None:
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else None
        accuracy = self._calculate_accuracy()
        self._log_metrics({
            "epoch_loss": avg_loss,
            "epoch_accuracy": accuracy,
        }, "train")
        logger.info(f"Epoch {unit.train_progress.num_epochs_completed} ended.")

    ######################## Evaluation state #########################
    def on_eval_epoch_start(self, state, unit) -> None:
        logger.info(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} started.")
        self.epoch_correct = 0
        self.epoch_total = 0

    def on_eval_step_end(self, state, unit) -> None:
        if state.eval_state and state.eval_state.step_output:
            current_loss = state.eval_state.step_output[0].item()

            # Log evaluation step loss
            if self.log_steps:
                wandb.log({"val_loss": current_loss, "step": unit.eval_progress.num_steps_completed})

            if len(state.eval_state.step_output) > 2 and self.log_accuracy:
                logits, labels = state.eval_state.step_output[1:3]
                _, preds = torch.max(logits, dim=-1)
                correct = torch.sum(preds == labels).item()
                total = labels.numel()
                self.epoch_correct += correct
                self.epoch_total += total

    def on_eval_epoch_end(self, state, unit) -> None:
        accuracy = self._calculate_accuracy()
        self._log_metrics({
            "epoch_val_accuracy": accuracy,
        }, "eval")
        logger.info(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} ended.")

    ######################## Training End #########################
    def on_train_end(self, state, unit) -> None:
        """Finish W&B logging at the end of training."""
        wandb.finish()
        logger.info("Training finished. W&B session closed.")

    ######################## Exception Handling #########################
    def on_exception(self, state, unit, exc: BaseException) -> None:
        logger.error(f"Exception occurred: {exc}")
        wandb.log({"exception": str(exc)})
        wandb.finish()
