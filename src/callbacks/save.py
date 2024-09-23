import torch

class Template(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    # Other methods...

    def on_train_epoch_end(self, state, unit) -> None:
        # Example: Save the model at the end of each epoch
        self.save_model(state.model, epoch=unit.train_progress.num_epochs_completed)

    def on_train_end(self, state, unit) -> None:
        # Example: Save the final model
        self.save_model(state.model, epoch='final')

    def save_model(self, model, epoch):
        # Construct a file path with the epoch number
        path = f"{self.save_path}/model_epoch_{epoch}.pth"
        # Save the model state
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
