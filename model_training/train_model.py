import argparse

from omegaconf import OmegaConf

from rnn_trainer import BrainToTextDecoder_Trainer


def main():
    parser = argparse.ArgumentParser(description='Train a brain-to-text decoder.')
    parser.add_argument('--config', type=str, default='rnn_args.yaml', help='Path to the training configuration YAML file.')
    parsed_args = parser.parse_args()

    args = OmegaConf.load(parsed_args.config)
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
