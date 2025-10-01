"""Entry point for training the DCoND diphone-aware decoder."""
import argparse

from omegaconf import OmegaConf

from rnn_trainer import BrainToTextDecoder_Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DCoND diphone-aware decoder")
    parser.add_argument(
        "--config",
        default="rnn_args.yaml",
        help="Path to the base training configuration",
    )
    parser.add_argument(
        "--output-dir",
        default="trained_models/dcond_rnn",
        help="Directory to store training artefacts",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="trained_models/dcond_rnn/checkpoint",
        help="Directory used for intermediate checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.output_dir = args.output_dir
    cfg.checkpoint_dir = args.checkpoint_dir

    if "dcond" not in cfg or cfg.dcond is None:
        cfg.dcond = {}
    cfg.dcond.use_diphone = True
    cfg.dcond.setdefault("alpha_initial", 0.2)
    cfg.dcond.setdefault("alpha_final", 0.6)
    cfg.dcond.setdefault("alpha_warmup_steps", 0)
    cfg.dcond.setdefault("alpha_ramp_steps", 40000)
    cfg.dcond.setdefault("blank_id", 0)
    cfg.dcond.setdefault("sil_id", 1)
    cfg.dcond.setdefault("include_terminal_sil", True)

    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
