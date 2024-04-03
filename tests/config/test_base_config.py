import sys

import pytest
from pydantic import Field, PositiveFloat, PositiveInt
from src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Training configuration settings."""

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

    seed: int = Field(
        default=None,
        description="Seed for random number generators.",
    )


def test_init():
    config = TrainingConfig()
    assert config.batch_size == 64
    assert config.epochs == 10
    assert config.learning_rate == 10
    assert config.seed is None


def test_parse_args():
    sys.argv = [
        "my_script.py",
        "--batch_size",
        "32",
        "--epochs",
        "20",
        "--learning_rate",
        "0.05",
        "--seed",
        "42",
    ]
    config = TrainingConfig().parse_args()
    assert config.batch_size == 32
    assert config.epochs == 20
    assert config.learning_rate == 0.05
    assert config.seed == 42


def test_parse_args_with_invalid_values():
    sys.argv = ["my_script.py", "--batch_size", "0", "--epochs", "0"]
    with pytest.raises(ValueError) as e:
        TrainingConfig().parse_args()
    assert "batch_size" in str(e.value)
    assert "epochs" in str(e.value)


def test_parse_args_with_config_file_arg():
    sys.argv = [
        "my_script.py",
        "--batch_size",
        "1",
        "--config_file",
        "tests/data/config.json",
    ]
    config = TrainingConfig().parse_args()

    # batch_size is set on CLI, so it overrides value in config file
    assert config.batch_size == 1

    # epochs is not set on CLI, so it uses value in config file
    assert config.epochs == 20

    # reamining are not set on CLI or in config file, so they use default value
    assert config.learning_rate == 10
    assert config.seed is None
