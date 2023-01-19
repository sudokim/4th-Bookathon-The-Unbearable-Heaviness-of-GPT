import os
import time
from logging import info

import pytorch_lightning as pl
from transformers import AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration
from transformers.optimization import Adafactor


class BookathonGPT2(pl.LightningModule):
    def __init__(self, checkpoint=None, learning_rate=1e-4):
        super().__init__()

        self.learning_rate = learning_rate

        if checkpoint is None:
            # Load new model
            info("Loading new model")
            self.model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
        else:
            # Load from pretrained checkpoint
            info(f"Loading model from checkpoint: {checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        # Save model
        if self.trainer.is_global_zero:
            while checkpoint_path := f"checkpoints/gpt2-{self.current_epoch:02d}-{int(time.time())}":
                if os.path.exists(checkpoint_path):
                    continue
                else:
                    break

            self.model.save_pretrained(checkpoint_path)
            info(f"Model saved to {checkpoint_path}")

    def validation_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), lr=self.learning_rate, scale_parameter=False, relative_step=False)
        return optimizer

        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000,
        #                                             num_training_steps=15000 * 30)
        #
        # return [optimizer], [scheduler]


class BookathonT5(pl.LightningModule):
    def __init__(self, checkpoint=None, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        if checkpoint is None:
            # Load new model
            info("Loading new model")
            self.model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-large-ko")

        else:
            # Load from pretrained checkpoint
            info(f"Loading model from checkpoint: {checkpoint}")
            self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        # Save model
        if self.trainer.is_global_zero:
            while checkpoint_path := f"checkpoints/t5-{self.current_epoch:02d}-{int(time.time())}":
                if os.path.exists(checkpoint_path):
                    continue
                else:
                    break

            self.model.save_pretrained(checkpoint_path)
            info(f"Model saved to {checkpoint_path}")

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return Adafactor(self.parameters(), lr=self.learning_rate, scale_parameter=False, relative_step=False)


class BookathonBART(pl.LightningModule):
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint is None:
            # Load new model
            info("Loading new model")
            self.model = AutoModel.from_pretrained("KETI-AIR/ke-t5-large-ko")

        else:
            # Load from pretrained checkpoint
            info(f"Loading model from checkpoint: {checkpoint}")
            self.model = AutoModel.from_pretrained(checkpoint)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        # Save model
        if self.trainer.is_global_zero:
            while checkpoint_path := f"checkpoints/bart-{self.current_epoch:02d}-{int(time.time())}":
                if os.path.exists(checkpoint_path):
                    continue
                else:
                    break

            self.model.save_pretrained(checkpoint_path)
            info(f"Model saved to {checkpoint_path}")

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return Adafactor(self.parameters(), lr=15e-5, scale_parameter=False, relative_step=False)


class BookathonGPTTrinity(pl.LightningModule):
    def __init__(self, checkpoint=None, learning_rate=1e-4):
        super().__init__()

        self.learning_rate = learning_rate

        if checkpoint is None:
            # Load new model
            info("Loading new model")
            self.model = AutoModelForCausalLM.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
        else:
            # Load from pretrained checkpoint
            info(f"Loading model from checkpoint: {checkpoint}")
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        # Save model
        if self.trainer.is_global_zero:
            while checkpoint_path := f"checkpoints/gpt_trinity-{self.current_epoch:02d}-{int(time.time())}":
                if os.path.exists(checkpoint_path):
                    continue
                else:
                    break

            self.model.save_pretrained(checkpoint_path)
            info(f"Model saved to {checkpoint_path}")

    def validation_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Batch: output from the tokenizer
        output = self.model(**batch)
        loss = output.loss

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return Adafactor(self.parameters(), lr=self.learning_rate, scale_parameter=False, relative_step=False)
