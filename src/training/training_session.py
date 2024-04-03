import logging
import os
import random

import mlflow
import numpy as np
import torch
from git import Repo
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig

LOGGER = logging.getLogger(__name__)


class TrainingSession:
    """TrainingSession is responsible for model training setup and configuration."""

    def __init__(self, config):
        self.config = config

    def start_experiment(self):
        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run():
            repo = Repo("/workspace")
            mlflow.set_tag("git_commit_hash", repo.head.object.hexsha)
            self.run()

    def run(self):
        self.seed_generators()
        # self.configure_logging()
        self.create_directories()
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.configure_device()
        self.create_optimizer()
        self.create_trainer()

    def seed_generators(self):
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

    def configure_logging(self):
        mlflow.log_params(vars(self.config))
        mlflow.autolog()

    def create_directories(self):
        os.makedirs(self.config.log_dir, exist_ok=True)

    def create_datasets(self):
        self.dataset = None

    def create_dataloaders(self):
        self.train_dataloader = None
        self.val_dataloader = None

    def create_model(self):
        self.model = None

    def configure_device(self):
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)

    def create_optimizer(self):
        # self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer = None

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=self.optimizer,
            config=self.config,
        )


def main():
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()


if __name__ == "__main__":
    main()
