import argparse
import json
import logging
from logging import info
from time import time

import pytorch_lightning as pl
from pytorch_lightning import strategies
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoTokenizer

from src.datamodule import BookathonDataModule
from src.datamodule_input_target import BookathonDataModuleInputTarget
from src.datamodule_keyword import BookathonDataModuleWithKeywords
from src.datamodule_sliding_window import BookathonDataModuleWithSlidingWindow
from src.model import BookathonGPTTrinity

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# Paths
parser.add_argument("--checkpoint", type=str, default=None, help="Path to previously trained model")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Path to checkpoint directory")
parser.add_argument("--dataset_path", type=str, default="./data", help="Path to dataset directory")

parser.add_argument("--datamodule", type=str, required=True, help="Datamodule to use",
                    choices=["vanilla", "sliding_window", "keywords", "input_target"])

# Length constraints
parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of tokens")

# Sliding window
parser.add_argument("--sw_sentences", type=int, default=6, help="Number of sentences in sliding window")
parser.add_argument("--sw_step", type=int, default=4, help="Step size of sliding window")

# Keyword
parser.add_argument("--input_loss", action="store_true", help="Calculate input prompt loss")
parser.add_argument("--train_sep_token", action="store_true",
                    help="Train the separator token separating the keywords and the input prompt")

# Input/target
parser.add_argument("--input_target", action="store_true", help="Use input/target in keywords datamodule")
parser.add_argument("--num_input_sentences", type=int, default=3, help="Number of input sentences")

# Training
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--fp32", action="store_true", help="Use 32-bit floating point precision")
parser.add_argument("--bf16", action="store_true", help="Use bf16 floating point precision")
parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator to use")
parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
parser.add_argument("--strategy", type=str, default=None, help="Strategy to use")
parser.add_argument("--gradient_accumulation", type=int, default=8, help="Number of gradient accumulation steps")


def main():
    args = parser.parse_args()

    if args.datamodule != args.keywords and args.input_target:
        raise ValueError("input_target argument is only supported in keywords datamodule. "
                         "Use --datamodule input_target to train with input/target without keywords.")

    # Save arguments to file
    args_file_name = f"{args.checkpoint_path}/{int(time())}.txt"
    json.dump(vars(args), open(args_file_name, "w"), indent=2)
    info(f"Arguments saved to {args_file_name}")

    if args.gradient_accumulation == 1:
        args.gradient_accumulation = None

    # pl.cli_lightning_logo()
    pl.seed_everything(args.seed, workers=True)

    # Log arguments
    logger = CSVLogger("logs", name="gpt2", flush_logs_every_n_steps=10)
    logger.log_hyperparams(args)

    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")

    model = BookathonGPTTrinity(args.checkpoint, args.lr)

    match args.datamodule:
        case "vanilla":
            datamodule = BookathonDataModule(
                    args.dataset_path,
                    tokenizer,
                    False,
                    dataloader_num_workers=args.dataloader_num_workers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
            )

        case "sliding_window":
            datamodule = BookathonDataModuleWithSlidingWindow(
                    dataset_path=args.dataset_path,
                    tokenizer=tokenizer,
                    dataloader_num_workers=args.dataloader_num_workers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    num_sentences=args.sw_sentences,
                    step=args.sw_step,
            )

        case "keywords":
            datamodule = BookathonDataModuleWithKeywords(
                    dataset_path=args.dataset_path,
                    tokenizer=tokenizer,
                    sentence_input_target=args.input_target,
                    dataloader_num_workers=args.dataloader_num_workers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    num_sentences=args.sw_sentences,
                    ignore_input_loss=not args.input_loss,
                    step=args.sw_step,
                    num_input_sentences=args.num_input_sentences,
                    train_sep_token=args.train_sep_token,
            )

        case "input_target":
            datamodule = BookathonDataModuleInputTarget(
                    dataset_path=args.dataset_path,
                    tokenizer=tokenizer,
                    dataloader_num_workers=args.dataloader_num_workers,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    num_sentences=args.sw_sentences,
                    step=args.sw_step,
                    num_input_sentences=args.num_input_sentences,
            )

        case _:
            raise ValueError(f"Invalid datamodule {args.datamodule}")

    callbacks = [
        # pl.callbacks.BatchSizeFinder(steps_per_trial=20, mode="binsearch"),
        pl.callbacks.RichModelSummary(max_depth=2),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=args.checkpoint_path,
                filename="gpt-trinity-" + args.datamodule + "-{epoch:02d}-{val_loss:.2f}",
        ),
        pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=3),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    match args.strategy:
        case "ddp":
            strategy = strategies.DDPStrategy(find_unused_parameters=True)
        case _:
            strategy = None

    if args.fp32:
        precision = 32
    elif args.bf16:
        precision = "bf16"
    else:
        precision = 16

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_epochs,
                         callbacks=callbacks, enable_model_summary=False, precision=precision,
                         accumulate_grad_batches=args.gradient_accumulation, strategy=strategy, logger=logger)
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
