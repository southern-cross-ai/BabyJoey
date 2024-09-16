from torch.utils.data import DataLoader

class BabyJoeyDataLoader:
    def __init__(self, training_dataset, validation_dataset, batch_size):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size

    def get_dataloaders(self):
        training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
        return training_dataloader, validation_dataloader
