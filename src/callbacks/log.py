import logging
from torchtnt.framework.callback import Callback
import torch
import os
from datetime import datetime

# Generate the log filename with current date and time
log_filename = datetime.now().strftime('logs/training_%Y-%m-%d_%H-%M-%S.log')

# Ensure logs folder exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging configuration
logging.basicConfig(
    filename=log_filename,  
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Log(Callback):
    def __init__(self):
        self.epoch_losses = []  # To store losses for the epoch
        self.epoch_correct = 0  # To track correct predictions for accuracy
        self.epoch_total = 0    # To track total predictions for accuracy

    ######################## Training state #########################
    def on_train_start(self, state, unit) -> None:
        logging.info("Training started!")

    def on_train_epoch_start(self, state, unit) -> None:
        logging.info(f"Epoch {unit.train_progress.num_epochs_completed} started.")
        self.epoch_losses = []  # Reset losses at the start of each epoch
        self.epoch_correct = 0  # Reset correct predictions
        self.epoch_total = 0    # Reset total predictions

    def on_train_step_end(self, state, unit) -> None:
        if state.train_state and state.train_state.step_output:
            current_loss = state.train_state.step_output[0].item()  # Ensure step_output is valid
            # Print the batch number and the current loss, updating on the same line
            print(f"\rBatch {unit.train_progress.num_steps_completed_in_epoch}: Current Loss: {current_loss:.6f}", end="")
            # Collect loss for the epoch
            self.epoch_losses.append(current_loss)

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
                self.epoch_correct += correct
                self.epoch_total += total

        else:
            # Print the "No valid loss" message on a new line
            print(f"Batch {unit.train_progress.num_steps_completed_in_epoch}: No valid loss available at this step.")
            logging.warning(f"Batch {unit.train_progress.num_steps_completed_in_epoch}: No valid loss available.")
    
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

        # Log both average loss and accuracy at the end of the epoch, handling None values
        logging.info(f"Epoch {unit.train_progress.num_epochs_completed} ended. "
                     f"{'Average Loss: ' + f'{avg_loss:.6f}' if avg_loss is not None else 'Average Loss: N/A'}, "
                     f"{'Accuracy: ' + f'{accuracy:.2f}%' if accuracy is not None else 'Accuracy: N/A'}")

    ######################## Evaluation state #########################
    def on_eval_epoch_start(self, state, unit) -> None:
        logging.info(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} started.")
        self.epoch_correct = 0  # Reset correct predictions for evaluation
        self.epoch_total = 0    # Reset total predictions for evaluation

    def on_eval_step_end(self, state, unit) -> None:
        step_input = unit.get_next_eval_batch(state, iter(state.eval_state.dataloader))
        
        if 'input_ids' in step_input:
            input_ids = step_input['input_ids']
            labels = input_ids[:, 1:].clone()  # Shift input_ids to create labels
            input_ids = input_ids[:, :-1]
            logits = state.eval_state.step_output[1]
            logits = logits[:, :labels.size(1), :]  # Match logits size to labels
            
            _, preds = torch.max(logits, dim=-1)
            
            if preds.size(0) != labels.size(0):
                min_size = min(preds.size(0), labels.size(0))
                preds = preds[:min_size]
                labels = labels[:min_size]
            
            correct = torch.sum(preds == labels).item()
            total = labels.numel()
            
            # Update epoch totals for accuracy during evaluation
            self.epoch_correct += correct
            self.epoch_total += total
            
            print(f"\rEvaluation Accuracy: {self.epoch_correct / self.epoch_total * 100:.2f}%", end="", flush=True)
        else:
            logging.warning("'input_ids' not found in the step_input.")

    def on_eval_end(self, state, unit) -> None:
        # Calculate and log accuracy for the evaluation epoch
        if self.epoch_total > 0:
            accuracy = (self.epoch_correct / self.epoch_total) * 100
        else:
            accuracy = None

        logging.info(f"Evaluation ended. Accuracy: {accuracy:.2f}%" if accuracy is not None else "Evaluation ended. Accuracy: N/A")
        print("\n")

    ######################## Exception Handling #########################
    def on_exception(self, state, unit, exc: BaseException) -> None:
        logging.error(f"Exception occurred: {exc}", exc_info=True)