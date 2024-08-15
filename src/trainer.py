import torch
import torch.optim as optim
from torchtnt.framework import fit, State
import logging

class Loop:
    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

    def create_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def train_step(self, state, data):
        model = state.model
        optimizer = state.optimizers[0]
        loss_fn = self.get_loss_function()

        model.train()
        inputs, targets = self.to_device(data)
        outputs = model(inputs)

        # Shift the outputs and targets for language modeling (similar to GPT-2)
        shift_outputs = outputs[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        loss = loss_fn(shift_outputs.view(-1, shift_outputs.size(-1)), shift_targets.view(-1))

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

            # Shift the outputs and targets for evaluation
            shift_outputs = outputs[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            loss = loss_fn(shift_outputs.view(-1, shift_outputs.size(-1)), shift_targets.view(-1))

        return {"val_loss": loss.item()}

    def get_loss_function(self):
        return torch.nn.CrossEntropyLoss()

    def to_device(self, data):
        inputs = data['input_ids']
        return inputs.to(self.device), inputs.to(self.device)

    def create(self):
        optimizer = self.create_optimizer()
        state = State(dataloader=self.train_loader, model=self.model, optimizer=optimizer, max_epochs=self.config.epochs)
        return state

    def run(self, state):
        try:
            # You can add a progress bar or integrate W&B here
            fit(
                state,
                self.train_loader,
                self.val_loader,
                self.train_step,
                self.val_step,
                # Add progress bar or W&B callback here
                callbacks=[]
            )
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise e
