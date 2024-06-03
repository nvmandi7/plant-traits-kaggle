import os
import time

from typing import Optional
from pydantic import Field, PositiveFloat, PositiveInt
from src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Training configuration settings."""

    data_dir: str = Field(
        default="data/processed/planttraits2024",
        description="planttraits2024 directory root containing the training tabular data with IDs.",
    )

    overfit_test: bool = Field(
        default=False,
        description="If true, sanity check model by trying to overfit on just 3 batches (wtih 20x epochs)",
    )

    use_precomputed_embeddings: bool = Field(
        default=False,
        description="Flag that determines whether to replace images with precomputed embeddings in dataset.",
    )

    batch_size: PositiveInt = Field(
        default=64,
        description="Number of training examples used per batch.",
    )

    encoder: str = Field(
        default="resnet50",
        description="Encoder used for precomputed embeddings, if used.",
    )

    epochs: PositiveInt = Field(
        default=500,
        description="Number of full passes over the training dataset.",
    )

    learning_rate: PositiveFloat = Field(
        default=3e-4,
        description="Learning rate for training.",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Seed for random number generators.",
    )

    experiment_name: str = Field(
        default="convnext_base_unfreeze_5.24_tune",
        description="Name of W&B experiment.",
    )

    force: bool = Field(
        default=False,
        description="Ignore a dirty git working tree (proceed with caution).",
    )

    # Possible additional args
    # - min and max epochs
    # - learning rate scheduler args
    # - early stopping
    # - model checkpoint dir and params
    # - drop outliers bool
    # - train-val split