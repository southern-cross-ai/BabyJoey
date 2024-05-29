import argparse
from config import Config
from training import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train a GPT-1 like model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Initialize trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
