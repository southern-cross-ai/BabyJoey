import torch
import torch.optim as optim
from torchtnt.framework import fit, init_train_state
from torchtnt.utils import ProgressBar
import logging

class Loop:
    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

    def create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train_step(self, state, data):
        model = state.model
        optimizer = state.optimizers[0]
        loss_fn = self.get_loss_function()

        model.train()
        inputs, targets = self.to_device(data)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def val_step(self, state, data):
        model = state.model
        loss_fn = self.get_loss_function()

        model.eval()
        with torch.no_grad():
            inputs, targets = self.to_device(data)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        return {"val_loss": loss.item()}

    def get_loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def to_device(self, data):
        inputs, targets = data
        return inputs.to(self.device), targets.to(self.device)

    def create(self):
        optimizer = self.create_optimizer()
        state = init_train_state(dataloader=self.train_loader, model=self.model, optimizer=optimizer, max_epochs=self.config.epochs)
        return state

    def run(self, state):
        try:
            fit(
                state,
                self.train_loader,
                self.val_loader,
                self.train_step,
                self.val_step,
                callbacks=[ProgressBar()]
            )
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise e
