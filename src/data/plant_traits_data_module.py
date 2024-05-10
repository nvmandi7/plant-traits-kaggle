
import pandas as pd
import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from pathlib import Path

from src.config.training_config import TrainingConfig
from src.data.datasets.plant_traits_dataset import PlantTraitsDataset
from src.data.datasets.baseline_dataset import BaselineDataset
from src.data.helpers.transform_holder import TransformHolder

class PlantTraitsDataModule(L.LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.encoder = config.encoder
        self.transform = TransformHolder.get_train_transform()
        self.dataset_cls = BaselineDataset if config.use_precomputed_embeddings else PlantTraitsDataset
        self.train_val_split = [0.8, 0.2]

    def prepare_data(self):
        # Download, tokenize, preprocess, etc.
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_path = Path(self.data_dir) / Path("train.feather")
            train_df = pd.read_feather(train_path)
            if self.dataset_cls == PlantTraitsDataset:
                plant_traits_full = self.dataset_cls(train_df, stage="train", transform=self.transform, drop_outliers=True)
            else:
                plant_traits_full = self.dataset_cls(train_df, stage="train", drop_outliers=True)

            self.plant_traits_train, self.plant_traits_val = random_split(
                plant_traits_full, self.train_val_split, generator=torch.Generator()
            )

        if stage == "test":
            test_path = Path(self.data_dir) / Path("test.feather")
            test_df = pd.read_feather(test_path)
            self.plant_traits_test = self.dataset_cls(test_df, stage="test")

        if stage == "predict":
            path = Path(self.data_dir) / Path("test.feather")
            df = pd.read_feather(path)
            self.plant_traits_predict = self.dataset_cls(df, stage="test")


    def train_dataloader(self):
        return DataLoader(self.plant_traits_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.plant_traits_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.plant_traits_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.plant_traits_predict, batch_size=self.batch_size)
