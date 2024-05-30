import torch
import torch.optim as optim
from model.model import GPTModel
from data.dataloader import get_dataloader
from utils.utils import save_checkpoint

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = GPTModel(config)
        self.dataloader = get_dataloader(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.config['num_epochs']):
            for batch in self.dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}, Loss: {loss.item()}')
            save_checkpoint(self.model, self.config['checkpoint_dir'], epoch)

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
