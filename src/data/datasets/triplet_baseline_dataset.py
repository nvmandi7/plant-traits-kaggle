
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.datasets.baseline_dataset import BaselineDataset

"""
Subclass of PlantTraitsDataset that reads precomputed embeddings rather than images.
It outputs row=embeddings+tabular, and labels for plant traits.
"""

class TripletBaselineDataset(BaselineDataset):
    def __init__(self, df, stage="train", model='resnet50', drop_outliers=False):
        # Same as BaselineDataset, but keep species column
        self.drop_cols.remove('species')
        super().__init__(df, stage=stage, model=model, drop_outliers=drop_outliers)

        # Remove singleton species
        self.df = self.df[self.df['species'].map(self.df['species'].value_counts()) > 1]


    def __getitem__(self, idx):
        anchor_row = torch.tensor(self.data.iloc[idx].values.astype(float), dtype=torch.float32)

        # According the species label, randomly choose positive and negative samples
        # TODO
