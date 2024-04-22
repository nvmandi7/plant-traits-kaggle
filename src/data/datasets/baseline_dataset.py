
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.datasets.plant_traits_dataset import PlantTraitsDataset

"""
Subclass of PlantTraitsDataset that reads precomputed embeddings rather than images.
It outputs row=embeddings+tabular, and labels for plant traits.
"""

class BaselineDataset(PlantTraitsDataset):
    def __init__(self, df, stage="train", model='resnet50', drop_outliers=False):
        self._base_init(df, drop_outliers)

        # Add embeddings columns to the DataFrame
        embeddings_path = f'data/processed/planttraits2024/{model}_{stage}_embeddings.feather'
        self.df = self.df.merge(pd.read_feather(embeddings_path), on='id')
        assert self.df.shape[0] == df.shape[0]

        # Drop columns
        self.df = self.df.drop(columns = self.drop_cols, axis=1)

        # Split dataset into data and target columns to be used downstream
        self.data = self.df.drop(self.target_cols, axis=1)
        self.targets = self.df[self.target_cols]


    def __getitem__(self, idx):
        # Get the corresponding row in the DataFrame and extract target columns
        targets = torch.tensor(self.targets.iloc[idx].values.astype(float), dtype=torch.float32)
        row = torch.tensor(self.data.iloc[idx].values.astype(float), dtype=torch.float32)

        return row, targets
