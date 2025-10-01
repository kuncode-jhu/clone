from omegaconf import OmegaConf

from dcond_trainer import DCoNDTrainer

args = OmegaConf.load('dcond_args.yaml')
trainer = DCoNDTrainer(args)
metrics = trainer.train()
