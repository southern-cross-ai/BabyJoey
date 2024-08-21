import torchtnt as tnt
from torchtnt.train import TrainLoop
from torchtnt.callbacks import EarlyStopping, ModelCheckpoint
from torchtnt.engine import Engine, State

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, max_epochs=10, checkpoint_path='best_model.pth'):
        # Initialize the state with model, optimizer, and data loaders
        self.state = State(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            max_epochs=max_epochs
        )

        # Initialize the training loop
        self.train_loop = TrainLoop(self.state)

        # Set up callbacks
        # self.early_stopping = tnt.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # self.checkpoint = tnt.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss')

        # Initialize the Engine
        self.engine = Engine(
            train_loop=self.train_loop, 
            eval_loop=None,  # You could add an evaluation loop here if needed
            callbacks=[self.early_stopping, self.checkpoint]
        )

    def fit(self):
        # Start the training process
        self.engine.run()

    def resume_training(self, checkpoint_path):
        # Optionally, you can add functionality to resume training from a checkpoint
        # Load the checkpoint and update the state
        # Example (pseudo-code):
        # checkpoint = torch.load(checkpoint_path)
        # self.state.model.load_state_dict(checkpoint['model_state_dict'])
        # self.state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.engine.run(start_epoch=checkpoint['epoch'] + 1)
        pass

    def evaluate(self):
        # If you want a separate method to run only the evaluation loop
        pass

# Example usage
# if __name__ == "__main__":
#     # Import your model, optimizer, data loaders, etc.
#     from models.my_model import MyModel
#     from data.dataloader import get_train_loader, get_val_loader
#     from torch.optim import Adam

#     model = MyModel()
#     optimizer = Adam(model.parameters())
#     train_loader = get_train_loader()
#     val_loader = get_val_loader()

#     # Create an instance of the Trainer class
#     trainer = Trainer(model, optimizer, train_loader, val_loader, max_epochs=10)

#     # Start training
#     trainer.fit()
