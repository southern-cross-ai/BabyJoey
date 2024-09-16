from torchtnt.framework.callback import Callback

class TestingCallback(Callback):
    def on_train_start(self, state, unit):
        print("Training started!")
    def on_train_end(self, state, unit):
        print("Training ended!")
    def on_train_epoch_start(self, state, unit):
        print(f"Training epoch {unit.train_progress.num_epochs_completed} started!")
    def on_train_epoch_end(self, state, unit):
        print(f"Training epoch {unit.train_progress.num_epochs_completed} ended!")
    def on_train_step_start(self, state, unit):
        print(f"Training batch {unit.train_progress.num_steps_completed} started!")
    def on_train_step_end(self, state, unit):
        print(f"Training batch {unit.train_progress.num_steps_completed} ended!")
    def on_eval_start(self, state, unit):
        print("Evaluation started!")
    def on_eval_end(self, state, unit):
        print("Evaluation ended!")
    def on_eval_epoch_start(self, state, unit):
        print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} started!")
    def on_eval_epoch_end(self, state, unit):
        print(f"Evaluation epoch {unit.eval_progress.num_epochs_completed} ended!")
    def on_eval_step_start(self, state, unit):
        print(f"Evaluation batch {unit.eval_progress.num_steps_completed} started!")
    def on_eval_step_end(self, state, unit):
        print(f"Evaluation batch {unit.eval_progress.num_steps_completed} ended!")
    def on_exception(self, state, unit, exc: BaseException):
        print(f"Exception occurred: {exc}")
