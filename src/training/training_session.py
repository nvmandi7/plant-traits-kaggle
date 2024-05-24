
import os
import random
import pandas as pd
import numpy as np
import torch
import datetime
from dotenv import load_dotenv

import lightning as L
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch.callbacks as callbacks

from src.config.training_config import TrainingConfig
from src.data.plant_traits_data_module import PlantTraitsDataModule
from src.modeling.baseline_model import BaselineModel
from src.modeling.plant_traits_model import PlantTraitsModel



class TrainingSession:
    """TrainingSession is responsible for model training setup and configuration."""

    def __init__(self, config):
        self.config = config
        if config.overfit_test:
            config.epochs *= 20

    def run(self):
        self.seed_generators()
        self.configure_device()
        self.create_datamodule()
        self.create_model()
        self.configure_logging(self.config.experiment_name)
        self.create_trainer()
        self.trainer.fit(self.model, self.datamodule)

    def seed_generators(self):
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

    def configure_device(self):
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    def create_datamodule(self):
        self.datamodule = PlantTraitsDataModule(self.config)

    def create_model(self):
        scheduler_args = {
            "steps_per_epoch": self.datamodule.train_val_split[0] // self.config.batch_size, # TODO, fix method for calculating len(train_dataloader) statically
            "epochs"         : self.config.epochs,
        }

        self.model = PlantTraitsModel(learning_rate=self.config.learning_rate, scheduler_args=scheduler_args)
        self.model_name = "BaselineModel"

    def configure_logging(self, experiment_name=""):
        load_dotenv()

        # Create ID as current datetime + random 4-letter ID
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        run_name = f"{current_datetime}_{random_id}_{experiment_name}"

        # Initialize the WandB logger with project name and run name
        wandb_logger = pl_loggers.wandb.WandbLogger(project="PlantTraits2024", name=run_name, log_model=True)
        wandb_logger.log_hyperparams({"model_name": self.model_name})

        self.wandb_logger = wandb_logger

    def create_trainer(self):
        # Define Trainer configuration (temp here)
        trainer_config = {
            'accelerator': 'gpu',
            'devices': 1,
            'min_epochs': 10,
            'max_epochs': self.config.epochs,
            'logger': self.wandb_logger,
            'precision': '16-mixed',
            'check_val_every_n_epoch': 1,
            'callbacks': [
                # Add any additional callbacks if needed
                callbacks.LearningRateMonitor(logging_interval='step'),  # Log learning rate
                callbacks.ModelCheckpoint(dirpath='./models/',  monitor="val_r2", mode="max", save_top_k=1),
            ],
            'benchmark': True,
            # TODO gradient clipping
            # 'fast_dev_run': True,
        }

        if self.config.overfit_test == True:
            trainer_config['overfit_batches'] = 3
        # else:
            # es = callbacks.EarlyStopping(monitor="average_val_loss", patience=5, mode="min"),
            # trainer_config['callbacks'].append(es)

        self.trainer = Trainer(**trainer_config)

    # TODO find appropriate way to pass in args for scheduler. 
    # Potentially pass scheduler into model separate from optimizer?
    def create_optimizers(self): 
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        self.optimizer_config = {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config.learning_rate,
                    steps_per_epoch=len(self.datamodule.train_dataloader()),
                    epochs=self.trainer.max_epochs
                ),
                'interval': 'step',
            },
        }
        return self.optimizer_config


def main():
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()


if __name__ == "__main__":
    main()
