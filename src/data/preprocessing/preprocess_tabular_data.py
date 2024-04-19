
import pandas as pd

from src.data.helpers.dataset_scaler import Dataset_Scaler

"""
Scale dataset and convert to feather file
"""

def main():
    scaler = Dataset_Scaler()
    train_path = "data/raw/planttraits2024/train.csv"

    train_df = pd.read_csv(train_path)
    train_df = scaler.scale_df(train_df)
    print("Train data shape: ", train_df.shape)

    train_df.to_feather("data/processed/planttraits2024/train.feather")
    return



if __name__ == "__main__":
    main()