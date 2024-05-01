import os
import time

from typing import Optional
from pydantic import Field, PositiveFloat, PositiveInt
from src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Training configuration settings."""

    data_dir: str = Field(
        default="data/raw/planttraits2024",
        description="Directory containing the training data.",
    )

    use_precomputed_embeddings: bool = Field(
        default=False,
        description="Flag that determines whether to replace images with precomputed embeddings in dataset.",
    )

    batch_size: PositiveInt = Field(
        default=64,
        description="Number of training examples used per batch.",
    )

    epochs: PositiveInt = Field(
        default=10,
        description="Number of full passes over the training dataset.",
    )

    learning_rate: PositiveFloat = Field(
        default=10,
        description="Learning rate for training.",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Seed for random number generators.",
    )

    log_dir: str = Field(
        default=os.path.join("logs", time.strftime("%Y-%m-%d_%H-%M-%S")),
        description="Directory where logs are stored.",
    )

    experiment_name: str = Field(
        default="Initial Experiment",
        description="Name of MLFlow experiment that contains MLFlow runs.",
    )

    force: bool = Field(
        default=False,
        description="Ignore a dirty git working tree (proceed with caution).",
    )

    # Possible additional args
    # - min and max epochs
    # - learning rate scheduler
    # - early stopping
    # - model checkpoint dir and params