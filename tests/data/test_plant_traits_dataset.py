
from src.data.plant_traits_dataset import PlantTraitsDataset
import pandas as pd


def test_init():
    train_path = "data/raw/planttraits2024/train.csv"
    print("Reading train data from: ", train_path)

    train_df = pd.read_csv(train_path)
    print("Train data shape: ", train_df.shape)

    dataset = PlantTraitsDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))
    return


def test_get_item():
    train_path = "data/raw/planttraits2024/train.csv"
    train_df = pd.read_csv(train_path)
    dataset = PlantTraitsDataset(train_df, stage="train")
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

    dataset = PlantTraitsDataset(train_df, stage="train")
    print("Dataset length: ", len(dataset))

    for i in range(5):
        img, row, targets = dataset[i]
        print("Image size: ", img.size)
        print("Row size: ", row.shape)
        print("Targets size: ", targets.shape)
        print("\n")