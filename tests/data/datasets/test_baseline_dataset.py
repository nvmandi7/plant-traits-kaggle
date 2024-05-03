
from src.data.datasets.baseline_dataset import BaselineDataset
import pandas as pd


def test_init():
    # train_path = "data/raw/planttraits2024/train.csv"
    train_path = "data/processed/planttraits2024/train.feather"
    print("Reading train data from: ", train_path)

    train_df = pd.read_feather(train_path)
    print("Train data shape: ", train_df.shape)
    print("Train data columns: ", train_df.columns)
    print("species" in train_df.columns)

    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))
    # print("Dataset columns: ", dataset.data.columns)

    assert len(dataset) == train_df.shape[0]
    assert "WORLDCLIM_BIO1_annual_mean_temperature" in dataset.data.columns
    assert "species" not in dataset.data.columns
    assert "2047" in dataset.data.columns # Embeddings

    return


def test_get_item():
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    for i in range(5):
        row, targets = dataset[i]
        print("Row size: ", row.shape)
        print("Targets size: ", targets.shape)
        print("\n")
    return

def _test_id_dtype():
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    embeddings_path = f'data/processed/planttraits2024/resnet50_train_embeddings.feather'
    embeddings_df = pd.read_feather(embeddings_path)
    print("Embeddings ID dtype: ", embeddings_df["id"].dtype)
    print("Train ID dtype: ", train_df["id"].dtype)


if __name__ == "__main__":
    train_path = "data/raw/planttraits2024/train.csv"
    print("Reading train data from: ", train_path)

    train_df = pd.read_csv(train_path)
    print("Train data shape: ", train_df.shape)

    dataset = BaselineDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    for i in range(5):
        row, targets = dataset[i]
        print("Image size: ", img.size)
        print("Row size: ", row.shape)
        print("Targets size: ", targets.shape)
        print("\n")