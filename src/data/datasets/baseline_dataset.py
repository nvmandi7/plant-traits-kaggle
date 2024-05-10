
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
    def __init__(self, df, stage="train", encoder='resnet50', drop_outliers=False):
        self._set_common_fields(df, drop_outliers)
        self.df["id"] = self.df["id"].astype(np.int64)
        num_rows = df.shape[0]

        # Add embeddings columns to the DataFrame
        embeddings_path = f'data/processed/planttraits2024/{encoder}_{stage}_embeddings.feather'
        embeddings_df = pd.read_feather(embeddings_path)
        embeddings_df["id"] = embeddings_df["id"].astype(np.int64)

        self.df = self.df.merge(embeddings_df, on='id')
        if drop_outliers == False:
            assert self.df.shape[0] == num_rows

        self._post_init()
    
    def _post_init(self):
        # Drop columns
        self.df = self.df.drop(columns = self.drop_cols, axis=1)

        # Split dataset into data and target columns to be used downstream
        self.data = self.df.drop(self.target_cols, axis=1)
        self.targets = self.df[self.target_cols]
        self.targets = self.df[["X4_mean", "X50_mean", "X11_mean", "X18_mean"]]


    def __getitem__(self, idx):
        # Get the corresponding row in the DataFrame and extract target columns
        targets = torch.tensor(self.targets.iloc[idx].values.astype(float), dtype=torch.float32)
        row = torch.tensor(self.data.iloc[idx].values.astype(float), dtype=torch.float32)

        return row, targets
