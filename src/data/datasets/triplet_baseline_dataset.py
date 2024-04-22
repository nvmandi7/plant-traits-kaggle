
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
    def _post_init(self):
        # Similar to BaselineDataset, but keep species column and remove singleton species
        self.df = self.df[self.df['species'].map(self.df['species'].value_counts()) > 1]
        self.drop_cols.remove('species')

        self.df = self.df.drop(columns = self.drop_cols, axis=1)
        self.data = self.df.drop(self.target_cols, axis=1)


    def __getitem__(self, idx):
        anchor_row = self._row_to_tensor(self.data.iloc[idx])

        # According the species label, randomly choose positive and negative samples
        species = self.data.iloc[idx]['species']
        positive_df = self.data[self.data['species'] == species]
        negative_df = self.data[self.data['species'] != species]

        positive_row = self._row_to_tensor(positive_df.sample(1).iloc[0])
        negative_row = self._row_to_tensor(negative_df.sample(1).iloc[0])

        return anchor_row, positive_row, negative_row, species



    @staticmethod
    def _row_to_tensor(row: pd.Series):
        row = row.drop('species')
        return torch.tensor(row.values.astype(float), dtype=torch.float32)





