import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def __getattr__(self, item):
        return self.config.get(item, None)
