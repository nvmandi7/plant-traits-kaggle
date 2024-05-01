
import os
import dotenv
import random
import pandas as pd
import numpy as np
import torch
import datetime

import lightning as L
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch.callbacks as callbacks

from src.config.training_config import TrainingConfig
from src.data.plant_traits_data_module import PlantTraitsDataModule
from src.modeling.baseline_model import BaselineModel



class TrainingSession:
    """TrainingSession is responsible for model training setup and configuration."""

    def __init__(self, config):
        self.config = config

    def start_experiment(self):
        pass
        # Wandb logger, set hparams
        # mlflow.set_experiment(self.config.experiment_name)
        # with mlflow.start_run():
        #     repo = Repo("/workspace")
        #     mlflow.set_tag("git_commit_hash", repo.head.object.hexsha)
        #     self.run()

    def run(self):
        self.seed_generators()
        self.create_datamodule()
        self.create_model()
        self.configure_logging()
        self.configure_device()
        self.create_trainer()
        self.trainer.fit(self.model, self.datamodule)

    def seed_generators(self):
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

    def create_datamodule(self):
        self.datamodule = PlantTraitsDataModule(self.config)

    def create_model(self):
        self.model = BaselineModel()
        self.model_name = "BaselineModel"

    def configure_logging(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        # Load dotenv
        dotenv.load_dotenv()


        # Create ID as current datetime + random 4-letter ID
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        run_name = f"{current_datetime}_{random_id}"

        # Initialize the WandB logger with project name and run name
        wandb_logger = pl_loggers.wandb.WandbLogger(project="PlantTraits2024", name=run_name, log_model=True)
        wandb_logger.log_hyperparams({"model_name": self.model_name})

        self.wandb_logger = wandb_logger

    def configure_device(self):
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # def create_optimizer(self):
    #     # self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
    #     self.optimizer = None

    def create_trainer(self):
        # Define Trainer configuration (temp here)
        trainer_config = {
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': 10,
            'logger': self.wandb_logger,
            'precision': '16-mixed',
            'check_val_every_n_epoch': 1,
            'callbacks': [
                # Add any additional callbacks if needed
                callbacks.LearningRateMonitor(logging_interval='step'),  # Log learning rate
                callbacks.ModelCheckpoint(dirpath='./models/',  monitor="val_r2", mode="max", save_top_k=1),
                callbacks.ModelCheckpoint(dirpath='./models/',  monitor="val_loss", mode="min", save_top_k=1)
                # Early stopping
            ],
            'benchmark': True,
            'fast_dev_run': True,
            'overfit_batches': 1,
        }

        self.trainer = Trainer(**trainer_config)


def main():
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()


if __name__ == "__main__":
    main()
