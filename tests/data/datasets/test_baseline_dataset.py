
from src.data.datasets.baseline_dataset import BaselineDataset
import pandas as pd


def test_init():
    # train_path = "data/raw/planttraits2024/train.csv"
    train_path = "data/processed/planttraits2024/train.feather"
    print("Reading train data from: ", train_path)

    train_df = pd.read_feather(train_path)
    print("Train data shape: ", train_df.shape)

    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))
    return


def test_get_item():
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    print(train_df.columns)
    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    for i in range(5):
        img, row, targets = dataset[i]
        print("Image size: ", img.size)
        print("Row size: ", row.shape)
        print("Targets size: ", targets.shape)
        print("\n")
    return


if __name__ == "__main__":
    train_path = "data/raw/planttraits2024/train.csv"
    print("Reading train data from: ", train_path)

    train_df = pd.read_csv(train_path)
    print("Train data shape: ", train_df.shape)

    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    for i in range(5):
        img, row, targets = dataset[i]
        print("Image size: ", img.size)
        print("Row size: ", row.shape)
        print("Targets size: ", targets.shape)
        print("\n")