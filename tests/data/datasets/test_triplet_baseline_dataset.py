
from src.data.datasets.triplet_baseline_dataset import TripletBaselineDataset
import pandas as pd
import torch


def test_init():
    train_path = "data/processed/planttraits2024/train.feather"
    print("Reading train data from: ", train_path)

    train_df = pd.read_feather(train_path)
    print("Train data shape: ", train_df.shape)

    dataset = TripletBaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    assert "species" in dataset.df.columns

    # Assert no singletons
    species_counts = dataset.df['species'].value_counts()
    assert (species_counts == 1).sum() == 0

    return


def test_get_item():
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    dataset = TripletBaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    # Replace _row_to_tensor so it keeps species column, for testing
    dataset._row_to_tensor = lambda x: x

    for i in range(5):
        anchor, positive, negative, species = dataset[i]
        print("Anchor size: ", anchor.shape)
        assert anchor.shape[0] == positive.shape[0] == negative.shape[0]

        assert anchor['species'] == positive['species']
        assert anchor['species'] != negative['species']
        assert anchor['species'] == species

    return


if __name__ == "__main__":
    pass