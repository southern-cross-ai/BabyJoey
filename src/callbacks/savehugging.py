from torchtnt.framework.callback import Callback

class SaveHugging(Callback):
    ######################## Training state #########################
    def on_train_start(self, state, unit) -> None:
        # Log or use the state object to get details like total epochs or steps
        # print(f"Training started with {state.train_state.max_epochs} epochs and {state.train_state.max_steps} steps")
        pass

    def on_train_epoch_start(self, state, unit) -> None:
        # Log the epoch number or other training details
        # print(f"Training epoch {unit.train_progress.num_epochs_completed} started!")
        pass

    def on_train_step_start(self, state, unit) -> None:
        # Log batch number or other training step details
        # print(f"Training batch {unit.train_progress.num_steps_completed} started!")
        pass

    def on_train_step_end(self, state, unit) -> None:
        # Example: Log the current loss or step output at the end of the step
        # if state.train_state and state.train_state.step_output:
        #     current_loss = state.train_state.step_output[0].item()
        #     print(f"Current Loss: {current_loss}")
        pass

    def on_train_epoch_end(self, state, unit) -> None:
        # Log the completion of an epoch
        # print(f"Training epoch {unit.train_progress.num_epochs_completed} ended!")
        pass

    def on_train_end(self, state, unit) -> None:
        # Summarize the training process, for example, log total epochs or steps
        # print("Training ended!")
        pass
    
    ######################## Evaluation state #########################
    def on_eval_start(self, state, unit) -> None:
        # Log the start of the evaluation phase
        # print("Evaluation started!")
        pass

    def on_eval_epoch_start(self, state, unit) -> None:
        # Log the start of an evaluation epoch
        # print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} started!")
        pass

    def on_eval_step_start(self, state, unit) -> None:
        # Log the start of an evaluation step (batch)
        # print(f"Evaluation batch {unit.eval_progress.num_steps_completed} started!")
        pass

    def on_eval_step_end(self, state, unit) -> None:
        # Example: Log evaluation metrics or step outputs
        # if state.eval_state and state.eval_state.step_output:
        #     print(f"Evaluation step output: {state.eval_state.step_output}")
        pass

    def on_eval_epoch_end(self, state, unit) -> None:
        # Log the end of an evaluation epoch
        # print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} ended!")
        pass

    def on_eval_end(self, state, unit) -> None:
        # Summarize the evaluation process
        # print("Evaluation ended!")
        pass
    
    ######################## Exception Handling #########################
    def on_exception(self, state, unit, exc: BaseException) -> None:
        # Log the exception details for debugging
        # print(f"Exception occurred: {exc}")
        pass
